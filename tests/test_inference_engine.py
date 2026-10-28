"""Unit and integration tests for InferenceEngine."""

import time
import numpy as np
import pytest

from pc_infernoguard.capture.stream_types import FrameData
from pc_infernoguard.inference import (
    DetectionBox,
    InferenceConfig,
    InferenceEngine,
    InferenceResult,
)


def test_inference_config_validation():
    # Valid config
    cfg = InferenceConfig(conf_threshold=0.3, iou_threshold=0.5, imgsz=640)
    assert cfg.conf_threshold == 0.3
    assert cfg.imgsz == 640

    # Invalid conf_threshold
    with pytest.raises(ValueError):
        InferenceConfig(conf_threshold=1.5)

    # Invalid imgsz (not multiple of 32)
    with pytest.raises(ValueError):
        InferenceConfig(imgsz=641)

    # Negative warmup
    with pytest.raises(ValueError):
        InferenceConfig(warmup_iterations=-1)


def test_detection_box_validation():
    # Valid box
    box = DetectionBox(
        cls_id=0,
        cls_name="fire",
        confidence=0.88,
        xyxy=(10.0, 15.0, 100.0, 150.0),
        xyxy_norm=(0.05, 0.05, 0.5, 0.5),
        model_name="inferno",
    )
    assert box.cls_name == "fire"
    assert box.confidence == 0.88
    assert box.model_name == "inferno"

    # Invalid confidence
    with pytest.raises(ValueError):
        DetectionBox(
            cls_id=0,
            cls_name="fire",
            confidence=1.2,
            xyxy=(10.0, 15.0, 100.0, 150.0),
            xyxy_norm=(0.05, 0.05, 0.5, 0.5),
            model_name="inferno",
        )

    # Invalid inverted coordinates (x1 > x2)
    with pytest.raises(ValueError):
        DetectionBox(
            cls_id=0,
            cls_name="fire",
            confidence=0.5,
            xyxy=(100.0, 15.0, 10.0, 150.0),
            xyxy_norm=(0.5, 0.05, 0.05, 0.5),
            model_name="inferno",
        )


def test_inference_engine_initialization_and_warmup():
    cfg = InferenceConfig(
        model_path="models/inferno.pt",
        device="cpu",
        imgsz=320,  # small for fast test execution
        warmup_iterations=1,
    )
    engine = InferenceEngine(cfg)

    assert engine.resolved_device == "cpu"
    assert engine.is_warm is True
    assert 0 in engine.class_names
    assert engine.class_names[0] == "fire"
    assert engine.class_names[1] == "smoke"
    assert engine.class_names[2] == "spark"


def test_inference_engine_predict_frame():
    cfg = InferenceConfig(
        model_path="models/inferno.pt",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    engine = InferenceEngine(cfg)

    # Create synthetic frame
    img = np.ones((240, 320, 3), dtype=np.uint8) * 128
    frame = FrameData(
        source_id="test_cam",
        frame_id=42,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic(),
        image=img,
        width=320,
        height=240,
    )

    result = engine.predict_frame(frame)

    assert isinstance(result, InferenceResult)
    assert result.source_id == "test_cam"
    assert result.frame_id == 42
    assert result.image_shape == (240, 320)
    assert result.device_used == "cpu"
    assert result.inference_ms > 0.0
    assert result.total_latency_ms > 0.0
    assert result.end_to_end_ms > 0.0


def test_inference_engine_predict_batch():
    cfg = InferenceConfig(
        model_path="models/inferno.pt",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
    )
    engine = InferenceEngine(cfg)

    img1 = np.zeros((240, 320, 3), dtype=np.uint8)
    img2 = np.ones((240, 320, 3), dtype=np.uint8) * 200

    f1 = FrameData(
        source_id="cam_A",
        frame_id=101,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic(),
        image=img1,
        width=320,
        height=240,
    )
    f2 = FrameData(
        source_id="cam_B",
        frame_id=102,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic(),
        image=img2,
        width=320,
        height=240,
    )

    results = engine.predict_batch([f1, f2])

    assert len(results) == 2
    assert results[0].source_id == "cam_A"
    assert results[0].frame_id == 101
    assert results[0].end_to_end_ms > 0.0
    assert results[1].source_id == "cam_B"
    assert results[1].frame_id == 102
    assert results[1].end_to_end_ms > 0.0


def test_inference_engine_predict_batch_rejecting_missing_metadata():
    """Verify predict_batch rejects non-FrameData items that lack frame metadata."""
    cfg = InferenceConfig(model_path="models/inferno.pt", device="cpu", imgsz=320, warmup_iterations=0)
    engine = InferenceEngine(cfg)
    raw_image = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(TypeError) as exc_info:
        engine.predict_batch([raw_image])  # type: ignore
    assert "FrameData" in str(exc_info.value)


def test_inference_engine_strict_classes_validation():
    """Verify strict_classes=True raises ValueError on mismatch, while False warns."""
    # Strict validation failure
    cfg_strict = InferenceConfig(
        model_path="models/inferno.pt",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
        expected_classes={0: "wrong_class_name"},
        strict_classes=True,
    )
    with pytest.raises(ValueError) as exc_info:
        InferenceEngine(cfg_strict)
    assert "Model class mapping mismatch" in str(exc_info.value)

    # Permissive validation logs warning and succeeds
    cfg_permissive = InferenceConfig(
        model_path="models/inferno.pt",
        device="cpu",
        imgsz=320,
        warmup_iterations=0,
        expected_classes={0: "wrong_class_name"},
        strict_classes=False,
    )
    engine_permissive = InferenceEngine(cfg_permissive)
    assert engine_permissive is not None


def test_inference_engine_empty_batch():
    cfg = InferenceConfig(model_path="models/inferno.pt", device="cpu", imgsz=320)
    engine = InferenceEngine(cfg)
    assert engine.predict_batch([]) == []
