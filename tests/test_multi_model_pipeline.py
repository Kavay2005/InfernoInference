"""Unit tests for MultiModelPipeline orchestration and model attribution."""

import time
import numpy as np
import pytest

from pc_infernoguard.capture.stream_types import FrameData
from pc_infernoguard.inference import (
    InferenceConfig,
    InferenceEngine,
    MultiModelPipeline,
    ThreadSafeInferenceWrapper,
)


def test_multi_model_pipeline_registration():
    pipeline = MultiModelPipeline()

    cfg1 = InferenceConfig(
        model_path="models/inferno.pt",
        model_name="detector_primary",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    engine1 = InferenceEngine(cfg1)

    cfg2 = InferenceConfig(
        model_path="models/inferno.pt",
        model_name="detector_secondary",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    engine2 = InferenceEngine(cfg2)
    wrapper2 = ThreadSafeInferenceWrapper(engine2)

    pipeline.register_model("detector_primary", engine1)
    pipeline.register_model("detector_secondary", wrapper2)

    assert set(pipeline.list_models()) == {"detector_primary", "detector_secondary"}
    assert pipeline.get_model("detector_primary") is engine1
    assert pipeline.get_model("detector_secondary") is wrapper2

    # Duplicate registration raises ValueError
    with pytest.raises(ValueError):
        pipeline.register_model("detector_primary", engine1)

    # Unregister
    pipeline.unregister_model("detector_secondary")
    assert pipeline.list_models() == ["detector_primary"]


def test_multi_model_pipeline_register_model_name_mismatch_raises():
    """Verify register_model requires engine config model_name to equal registered name."""
    pipeline = MultiModelPipeline()
    cfg = InferenceConfig(
        model_path="models/inferno.pt",
        model_name="original_name",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    engine = InferenceEngine(cfg)

    with pytest.raises(ValueError) as exc_info:
        pipeline.register_model("different_name", engine)
    assert "Model name mismatch" in str(exc_info.value)


def test_multi_model_pipeline_predict_combined():
    pipeline = MultiModelPipeline()

    cfg1 = InferenceConfig(
        model_path="models/inferno.pt",
        model_name="detector_primary",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    cfg2 = InferenceConfig(
        model_path="models/inferno.pt",
        model_name="verifier_secondary",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )

    pipeline.register_model("detector_primary", InferenceEngine(cfg1))
    pipeline.register_model("verifier_secondary", InferenceEngine(cfg2))

    img = np.ones((240, 320, 3), dtype=np.uint8) * 100
    frame = FrameData(
        source_id="cam_main",
        frame_id=1,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic(),
        image=img,
        width=320,
        height=240,
    )

    # Individual model results
    results = pipeline.predict_frame(frame)
    assert "detector_primary" in results
    assert "verifier_secondary" in results
    assert results["detector_primary"].source_id == "cam_main"
    assert results["verifier_secondary"].source_id == "cam_main"
    assert results["detector_primary"].end_to_end_ms > 0
    assert results["verifier_secondary"].end_to_end_ms > 0

    # Combined aggregated result
    combined = pipeline.predict_combined(frame)
    assert combined.source_id == "cam_main"
    assert combined.frame_id == 1
    assert combined.image_shape == (240, 320)
    assert combined.total_latency_ms > 0
    assert combined.end_to_end_ms > 0
    # Any detection generated will have its model_name tagged
    for det in combined.detections:
        assert det.model_name in ("detector_primary", "verifier_secondary")


def test_multi_model_pipeline_predict_combined_no_models_ran():
    """Verify predict_combined reports device_used='none' and end_to_end_ms=0.0 when no models run."""
    pipeline = MultiModelPipeline()
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    frame = FrameData(
        source_id="empty_cam",
        frame_id=1,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic(),
        image=img,
        width=320,
        height=240,
    )
    res = pipeline.predict_combined(frame)
    assert res.device_used == "none"
    assert res.end_to_end_ms == 0.0
    assert len(res.detections) == 0


def test_multi_model_pipeline_predict_batch():
    """Verify predict_batch calls each engine once for the whole frame batch."""
    pipeline = MultiModelPipeline()
    cfg1 = InferenceConfig(
        model_path="models/inferno.pt",
        model_name="batch_model_1",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    cfg2 = InferenceConfig(
        model_path="models/inferno.pt",
        model_name="batch_model_2",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    pipeline.register_model("batch_model_1", InferenceEngine(cfg1))
    pipeline.register_model("batch_model_2", InferenceEngine(cfg2))

    img1 = np.ones((240, 320, 3), dtype=np.uint8) * 10
    img2 = np.ones((240, 320, 3), dtype=np.uint8) * 20
    frames = [
        FrameData(
            source_id="cam_0",
            frame_id=1,
            timestamp_utc=time.time(),
            timestamp_mono=time.monotonic(),
            image=img1,
            width=320,
            height=240,
        ),
        FrameData(
            source_id="cam_1",
            frame_id=1,
            timestamp_utc=time.time(),
            timestamp_mono=time.monotonic(),
            image=img2,
            width=320,
            height=240,
        ),
    ]

    batch_res = pipeline.predict_batch(frames)
    assert "batch_model_1" in batch_res
    assert "batch_model_2" in batch_res
    assert len(batch_res["batch_model_1"]) == 2
    assert len(batch_res["batch_model_2"]) == 2
    assert batch_res["batch_model_1"][0].source_id == "cam_0"
    assert batch_res["batch_model_1"][1].source_id == "cam_1"


def test_multi_model_pipeline_failure_isolation():
    """Verify that if one model fails, its error is logged, other models succeed, and pipeline does not crash."""
    pipeline = MultiModelPipeline()

    # 1. Healthy model
    cfg_healthy = InferenceConfig(
        model_path="models/inferno.pt",
        model_name="healthy_model",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    pipeline.register_model("healthy_model", InferenceEngine(cfg_healthy))

    # 2. Failing model mock
    class FaultyEngine:
        cfg = InferenceConfig(model_name="faulty_model")

        def predict_batch(self, frames):
            raise RuntimeError("Simulated GPU out-of-memory or corrupt tensor in faulty_model")

    pipeline.register_model("faulty_model", FaultyEngine())

    img = np.ones((240, 320, 3), dtype=np.uint8) * 50
    frame = FrameData(
        source_id="cam_fault_test",
        frame_id=1,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic(),
        image=img,
        width=320,
        height=240,
    )

    # predict_batch should NOT raise
    results = pipeline.predict_batch([frame])

    # Healthy model results are intact
    assert "healthy_model" in results
    assert len(results["healthy_model"]) == 1
    assert results["healthy_model"][0].source_id == "cam_fault_test"

    # Faulty model omitted from results, and error recorded in last_errors
    assert "faulty_model" not in results
    assert pipeline.last_errors["faulty_model"] is not None
    assert "Simulated GPU out-of-memory" in pipeline.last_errors["faulty_model"]
    assert pipeline.last_errors["healthy_model"] is None


def test_multi_model_pipeline_from_manifest(tmp_path):
    """Verify constructing MultiModelPipeline from a models.yaml manifest file."""
    import yaml

    manifest_file = tmp_path / "models.yaml"
    manifest_data = {
        "models": [
            {
                "name": "manifest_model_1",
                "path": "models/inferno.pt",
                "device": "cpu",
                "imgsz": 320,
                "warmup_iterations": 0,
            }
        ]
    }
    manifest_file.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

    pipeline = MultiModelPipeline.from_manifest(manifest_file)
    assert pipeline.list_models() == ["manifest_model_1"]
    engine = pipeline.get_model("manifest_model_1")
    assert engine.cfg.model_name == "manifest_model_1"

