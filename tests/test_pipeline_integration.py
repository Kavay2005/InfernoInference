"""Integration tests verifying end-to-end multi-stream ingestion coupled with InferenceEngine."""

import time
from pathlib import Path
import pytest

from pc_infernoguard.capture.multi_stream_manager import MultiStreamManager
from pc_infernoguard.config import CameraConfig, MultiStreamConfig
from pc_infernoguard.inference import InferenceConfig, InferenceEngine
from pc_infernoguard.utils.streams_file import to_ultralytics_batch
from tests.synthetic_stream_fixture import create_synthetic_video_file


@pytest.fixture(scope="module")
def synthetic_stream_pair():
    v1 = create_synthetic_video_file(duration_sec=3.0, fps=30, prefix="integ_stream_1_")
    v2 = create_synthetic_video_file(duration_sec=3.0, fps=30, prefix="integ_stream_2_")
    yield v1, v2
    for p in (v1, v2):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def test_end_to_end_ingestion_and_inference_pipeline(synthetic_stream_pair):
    v1, v2 = synthetic_stream_pair

    # 1. Setup multi-stream capture manager
    cams = [
        CameraConfig(source_id="furnace_cam_0", uri=v1, reconnect_enabled=False),
        CameraConfig(source_id="exhaust_cam_1", uri=v2, reconnect_enabled=False),
    ]
    multi_cfg = MultiStreamConfig(cameras=cams)
    manager = MultiStreamManager(multi_cfg)

    # 2. Setup inference engine on CPU with lightweight resolution for fast test
    infer_cfg = InferenceConfig(
        model_path="models/inferno.pt",
        device="cpu",
        imgsz=320,
        conf_threshold=0.2,
        warmup_iterations=1,
    )
    engine = InferenceEngine(infer_cfg)
    assert engine.is_warm

    # 3. Execute combined pipeline
    with manager:
        time.sleep(0.3)
        batch = manager.read_batch()
        assert len(batch) >= 1

        frames = list(batch.values())
        results = engine.predict_batch(frames)

        assert len(results) == len(frames)
        for idx, res in enumerate(results):
            assert res.source_id == frames[idx].source_id
            assert res.frame_id == frames[idx].frame_id
            assert res.device_used == "cpu"
            assert res.inference_ms > 0
            assert res.total_latency_ms > 0
            assert res.end_to_end_ms > 0
            assert res.image_shape == (240, 320)
