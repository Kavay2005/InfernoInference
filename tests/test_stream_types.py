"""Unit tests for capture stream types and data structures."""

import pytest
import time
import numpy as np

from pc_infernoguard.capture.stream_types import FrameData, StreamHealth, StreamStatus


def test_stream_status_enum_values():
    assert StreamStatus.INITIALIZING.value == "INITIALIZING"
    assert StreamStatus.ACTIVE.value == "ACTIVE"
    assert StreamStatus.STALE.value == "STALE"
    assert StreamStatus.RECONNECTING.value == "RECONNECTING"
    assert StreamStatus.FAILED.value == "FAILED"
    assert StreamStatus.CLOSED.value == "CLOSED"


def test_frame_data_creation_and_immutability():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    t_utc = time.time()
    t_mono = time.monotonic()

    frame = FrameData(
        source_id="cam_0",
        frame_id=1,
        timestamp_utc=t_utc,
        timestamp_mono=t_mono,
        image=img,
        width=640,
        height=480,
        fps_estimate=30.0,
        metadata={"camera_model": "virtual"},
    )

    assert frame.source_id == "cam_0"
    assert frame.frame_id == 1
    assert frame.timestamp_utc == t_utc
    assert frame.timestamp_mono == t_mono
    assert frame.width == 640
    assert frame.height == 480
    assert frame.fps_estimate == 30.0
    assert frame.metadata["camera_model"] == "virtual"

    # FrameData is frozen (immutable)
    with pytest.raises(Exception):
        frame.frame_id = 2  # type: ignore


def test_frame_data_validation():
    # Test invalid image type
    with pytest.raises(TypeError):
        FrameData(
            source_id="test",
            frame_id=1,
            timestamp_utc=time.time(),
            timestamp_mono=time.monotonic(),
            image="not an image",  # type: ignore
            width=100,
            height=100,
        )

    # Test invalid ndim
    with pytest.raises(ValueError):
        FrameData(
            source_id="test",
            frame_id=1,
            timestamp_utc=time.time(),
            timestamp_mono=time.monotonic(),
            image=np.zeros((100, 100)),  # 2D instead of 3D
            width=100,
            height=100,
        )


def test_stream_health_defaults():
    health = StreamHealth(source_id="test_cam")
    assert health.source_id == "test_cam"
    assert health.status == StreamStatus.INITIALIZING
    assert health.total_frames_read == 0
    assert health.dropped_frames == 0
    assert health.reconnect_count == 0
    assert health.last_frame_timestamp is None
    assert health.current_fps is None
    assert health.error_message is None


def test_stream_health_is_stale_derivation():
    """Verify StreamHealth.is_stale behavior for uninitialized and timed-out states."""
    health = StreamHealth(source_id="stale_test")
    now = 100.0

    # 1. Uninitialized timestamp must be considered stale
    assert health.last_frame_timestamp is None
    assert health.is_stale(now=now, timeout_sec=5.0) is True

    # 2. Fresh timestamp within timeout is not stale
    health.last_frame_timestamp = 98.0
    assert health.is_stale(now=now, timeout_sec=5.0) is False

    # 3. Timestamp exceeding timeout is stale
    health.last_frame_timestamp = 94.0
    assert health.is_stale(now=now, timeout_sec=5.0) is True


def test_frame_data_comparison_does_not_raise_numpy_ambiguity():
    """Verify that FrameData(eq=False) uses identity comparison and does not raise ValueError."""
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)

    t_utc = time.time()
    t_mono = time.monotonic()

    frame1 = FrameData(
        source_id="cam",
        frame_id=1,
        timestamp_utc=t_utc,
        timestamp_mono=t_mono,
        image=img1,
        width=100,
        height=100,
    )
    frame2 = FrameData(
        source_id="cam",
        frame_id=1,
        timestamp_utc=t_utc,
        timestamp_mono=t_mono,
        image=img2,
        width=100,
        height=100,
    )

    # FrameData default fps_estimate is None
    assert frame1.fps_estimate is None

    # Comparison should not raise ValueError (would raise with dataclass default eq=True)
    assert not (frame1 == frame2)
    assert frame1 != frame2
    assert frame1 == frame1
