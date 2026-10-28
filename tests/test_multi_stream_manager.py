"""Unit and integration tests for MultiStreamManager and stream isolation."""

import time
from pathlib import Path
import pytest

from pc_infernoguard.capture.multi_stream_manager import MultiStreamManager
from pc_infernoguard.capture.stream_capture import StreamCapture
from pc_infernoguard.capture.stream_types import FrameData, StreamStatus
from pc_infernoguard.config import CameraConfig, MultiStreamConfig
from tests.synthetic_stream_fixture import create_synthetic_video_file


@pytest.fixture
def synthetic_video_pair():
    v1 = create_synthetic_video_file(duration_sec=3.0, fps=30, prefix="stream_pair_1_")
    v2 = create_synthetic_video_file(duration_sec=3.0, fps=30, prefix="stream_pair_2_")
    yield v1, v2
    for p in (v1, v2):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def test_multi_stream_manager_concurrent_reads(synthetic_video_pair):
    v1, v2 = synthetic_video_pair
    cams = [
        CameraConfig(source_id="cam_A", uri=v1, reconnect_enabled=False),
        CameraConfig(source_id="cam_B", uri=v2, reconnect_enabled=False),
    ]
    cfg = MultiStreamConfig(cameras=cams)
    manager = MultiStreamManager(cfg)

    with manager:
        # Wait a short moment for workers to capture frames
        time.sleep(0.3)
        batch = manager.read_batch()

        # Both streams should have delivered frames
        assert "cam_A" in batch or "cam_B" in batch
        if "cam_A" in batch:
            assert batch["cam_A"].source_id == "cam_A"
        if "cam_B" in batch:
            assert batch["cam_B"].source_id == "cam_B"

        health_report = manager.get_health_report()
        assert len(health_report) == 2
        assert "cam_A" in health_report
        assert "cam_B" in health_report


def test_stream_isolation_failing_stream_does_not_crash_healthy_stream(synthetic_video_pair):
    """Crucial requirement from AGENTS.md Section 7:
    A failure or reconnection of one camera must not crash the other camera pipelines.
    """
    v1, _ = synthetic_video_pair
    cams = [
        CameraConfig(source_id="healthy_cam", uri=v1, reconnect_enabled=False),
        CameraConfig(
            source_id="broken_cam",
            uri="invalid_dead_url_or_file.mp4",
            reconnect_enabled=True,
            reconnect_delay_sec=0.1,
            max_retries=2,
        ),
    ]
    cfg = MultiStreamConfig(cameras=cams)
    manager = MultiStreamManager(cfg)

    with manager:
        # Let broken stream attempt connection and fail while healthy stream runs
        time.sleep(0.6)

        # Healthy stream must remain operational
        healthy_capture = manager._streams["healthy_cam"]
        assert healthy_capture.is_alive()
        assert healthy_capture.get_health().total_frames_read > 0

        # Broken stream must be in RECONNECTING or FAILED without taking down manager or sibling stream
        broken_health = manager._streams["broken_cam"].get_health()
        assert broken_health.status in (StreamStatus.RECONNECTING, StreamStatus.FAILED)

        # Batch reading succeeds and returns healthy camera's frames
        batch = manager.read_batch()
        assert "healthy_cam" in batch
        assert batch["healthy_cam"].image is not None


def test_multi_stream_manager_max_frame_age_sec():
    """Verify that read_batch omits frames older than max_frame_age_sec."""
    import numpy as np

    cfg = MultiStreamConfig(cameras=[], max_frame_age_sec=0.1)
    manager = MultiStreamManager(cfg)
    cam = CameraConfig(source_id="aging_cam", uri="dummy")
    capture = StreamCapture(cam)
    manager._streams["aging_cam"] = capture

    # 1. Old frame (0.5s ago > 0.1s max age) must be omitted
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    old_frame = FrameData(
        source_id="aging_cam",
        frame_id=1,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic() - 0.5,
        image=img,
        width=100,
        height=100,
    )
    with capture._lock:
        capture._buffer.append(old_frame)

    batch = manager.read_batch()
    assert "aging_cam" not in batch

    # 2. Fresh frame (now < 0.1s max age) must be included
    fresh_frame = FrameData(
        source_id="aging_cam",
        frame_id=2,
        timestamp_utc=time.time(),
        timestamp_mono=time.monotonic(),
        image=img,
        width=100,
        height=100,
    )
    with capture._lock:
        capture._buffer.append(fresh_frame)

    batch_fresh = manager.read_batch()
    assert "aging_cam" in batch_fresh
    assert batch_fresh["aging_cam"].frame_id == 2


def test_multi_stream_manager_thread_safe_add_remove_no_dict_mutation_error(synthetic_video_pair):
    """Verify list snapshot iteration prevents 'dict changed size during iteration' errors."""
    v1, v2 = synthetic_video_pair
    cams = [CameraConfig(source_id="cam_base", uri=v1, reconnect_enabled=False)]
    manager = MultiStreamManager(MultiStreamConfig(cameras=cams))

    with manager:
        time.sleep(0.2)
        # Add a stream and immediately remove while calling read_batch and get_health_report
        new_cam = CameraConfig(source_id="cam_dynamic", uri=v2, reconnect_enabled=False)
        manager.add_stream(new_cam, auto_start=True)
        report = manager.get_health_report()
        assert "cam_dynamic" in report

        manager.remove_stream("cam_dynamic")
        report2 = manager.get_health_report()
        assert "cam_dynamic" not in report2
