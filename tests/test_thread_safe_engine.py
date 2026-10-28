"""Unit and concurrency tests for ThreadSafeInferenceWrapper."""

import concurrent.futures
import time
import numpy as np

from pc_infernoguard.capture.stream_types import FrameData
from pc_infernoguard.inference import (
    InferenceConfig,
    InferenceEngine,
    ThreadSafeInferenceWrapper,
)


def test_thread_safe_wrapper_delegation():
    cfg = InferenceConfig(
        model_path="models/inferno.pt",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    engine = InferenceEngine(cfg)
    wrapper = ThreadSafeInferenceWrapper(engine)

    assert wrapper.resolved_device == "cpu"
    assert wrapper.engine is engine
    assert wrapper.cfg.imgsz == 320
    assert 0 in wrapper.class_names
    assert wrapper.call_count == 0


def test_concurrent_multi_threaded_inference():
    """Verify that multiple concurrent threads calling predict_frame on a single wrapper

    execute without race conditions, corruption, or PyTorch exceptions.
    """
    cfg = InferenceConfig(
        model_path="models/inferno.pt",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    engine = InferenceEngine(cfg)
    wrapper = ThreadSafeInferenceWrapper(engine)

    num_threads = 4
    num_requests_per_thread = 3

    def worker_task(thread_id: int):
        results = []
        for i in range(num_requests_per_thread):
            img = np.ones((240, 320, 3), dtype=np.uint8) * ((thread_id * 30 + i * 10) % 255)
            frame = FrameData(
                source_id=f"thread_cam_{thread_id}",
                frame_id=i,
                timestamp_utc=time.time(),
                timestamp_mono=time.monotonic(),
                image=img,
                width=320,
                height=240,
            )
            res = wrapper.predict_frame(frame)
            results.append(res)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, tid) for tid in range(num_threads)]
        all_results = [f.result() for f in futures]

    # Verify each thread received expected results
    assert len(all_results) == num_threads
    for tid, res_list in enumerate(all_results):
        assert len(res_list) == num_requests_per_thread
        for idx, res in enumerate(res_list):
            assert res.source_id == f"thread_cam_{tid}"
            assert res.frame_id == idx
            assert res.device_used == "cpu"
            assert res.inference_ms > 0

    assert wrapper.call_count == num_threads * num_requests_per_thread


def test_thread_safe_wrapper_predict_batch():
    """Verify ThreadSafeInferenceWrapper.predict_batch executes with Sequence[FrameData]."""
    cfg = InferenceConfig(model_path="models/inferno.pt", device="cpu", imgsz=320, warmup_iterations=0)
    engine = InferenceEngine(cfg)
    wrapper = ThreadSafeInferenceWrapper(engine)

    img = np.zeros((240, 320, 3), dtype=np.uint8)
    f = FrameData(
        source_id="batch_cam",
        frame_id=1,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic(),
        image=img,
        width=320,
        height=240,
    )
    res = wrapper.predict_batch([f])
    assert len(res) == 1
    assert res[0].source_id == "batch_cam"
    assert res[0].end_to_end_ms > 0
    assert wrapper.call_count == 1
