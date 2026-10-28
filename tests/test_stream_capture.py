"""Unit and integration tests for single-source StreamCapture."""

import time
from pathlib import Path
import pytest

from pc_infernoguard.capture.stream_capture import StreamCapture
from pc_infernoguard.capture.stream_types import StreamStatus
from pc_infernoguard.config import CameraConfig
from tests.synthetic_stream_fixture import create_synthetic_video_file


@pytest.fixture
def synthetic_video():
    path = create_synthetic_video_file(duration_sec=3.0, fps=30, width=320, height=240)
    yield path
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def test_stream_capture_lifecycle_with_synthetic_video(synthetic_video):
    cfg = CameraConfig(
        source_id="test_synthetic_0",
        uri=synthetic_video,
        buffer_size=2,
        reconnect_enabled=False,
    )
    capture = StreamCapture(cfg)

    # Initial health check
    health_init = capture.get_health()
    assert health_init.status == StreamStatus.INITIALIZING

    capture.start()
    assert capture.is_alive()

    # Wait up to 2 seconds for frames to arrive
    frame = None
    start_wait = time.monotonic()
    while time.monotonic() - start_wait < 2.0:
        frame = capture.read(timeout_sec=0.2)
        if frame is not None:
            break

    assert frame is not None
    assert frame.source_id == "test_synthetic_0"
    assert frame.width == 320
    assert frame.height == 240
    assert frame.image.shape == (240, 320, 3)

    # Check health after frames captured
    health = capture.get_health()
    assert health.total_frames_read >= 1

    # Graceful stop
    capture.stop(timeout_sec=2.0)
    assert not capture.is_alive()
    assert capture.get_health().status == StreamStatus.CLOSED


def test_bounded_buffer_drops_old_frames(synthetic_video):
    cfg = CameraConfig(
        source_id="test_drop",
        uri=synthetic_video,
        buffer_size=1,  # strictly newest frame only
        reconnect_enabled=False,
    )
    capture = StreamCapture(cfg)
    capture.start()

    # Sleep while frames are produced without reading them
    time.sleep(0.5)

    health = capture.get_health()
    # If the video is producing frames and buffer_size is 1, dropped_frames should be > 0
    assert health.dropped_frames > 0, "Bounded buffer should increment dropped_frames when not read"

    # Reading should return the freshest frame immediately
    frame = capture.read_latest()
    assert frame is not None
    assert frame.frame_id > 1

    capture.stop()


def test_invalid_source_isolation_and_no_crash():
    cfg = CameraConfig(
        source_id="invalid_stream",
        uri="non_existent_file_or_invalid_device_path.mp4",
        reconnect_enabled=True,
        reconnect_delay_sec=0.1,
        max_retries=2,
    )
    capture = StreamCapture(cfg)
    capture.start()

    # Worker thread should attempt reconnect and transition to FAILED or RECONNECTING without throwing
    time.sleep(0.5)

    health = capture.get_health()
    assert health.status in (StreamStatus.RECONNECTING, StreamStatus.FAILED)
    assert health.error_message is not None

    capture.stop()
    assert capture.get_health().status == StreamStatus.CLOSED


def test_stale_status_derivation_without_mutating_stored_state():
    """Verify that get_health() returns STALE if inactive, without mutating internal state."""
    cfg = CameraConfig(
        source_id="stale_test",
        uri="dummy_uri",
        stale_timeout_sec=2.0,
    )
    capture = StreamCapture(cfg)

    # 1. Simulate active stream state with an old frame timestamp (5s ago > 2s timeout)
    now = time.monotonic()
    with capture._lock:
        capture._health.status = StreamStatus.ACTIVE
        capture._health.last_frame_timestamp = now - 5.0

    health_derived = capture.get_health()
    assert health_derived.status == StreamStatus.STALE

    # Ensure internal stored state remains ACTIVE (unmutated)
    with capture._lock:
        assert capture._health.status == StreamStatus.ACTIVE

    # 2. Simulate fresh frame timestamp (0.5s ago <= 2s timeout)
    with capture._lock:
        capture._health.last_frame_timestamp = now - 0.5

    fresh_health = capture.get_health()
    assert fresh_health.status == StreamStatus.ACTIVE


def test_is_file_source_and_no_pacing_on_live_sources():
    """Verify _is_file_source accurately differentiates live feeds from local video files."""
    # Live sources
    live_webcam = StreamCapture(CameraConfig(source_id="webcam_idx", uri=0))
    assert live_webcam._is_file_source() is False

    live_str_idx = StreamCapture(CameraConfig(source_id="webcam_str", uri="0"))
    assert live_str_idx._is_file_source() is False

    live_rtsp = StreamCapture(CameraConfig(source_id="rtsp_cam", uri="rtsp://192.168.1.1:554/ch0"))
    assert live_rtsp._is_file_source() is False

    live_http = StreamCapture(CameraConfig(source_id="http_cam", uri="http://192.168.1.1:8080/video"))
    assert live_http._is_file_source() is False

    # File sources
    file_mp4 = StreamCapture(CameraConfig(source_id="file_cam_mp4", uri="recordings/test.mp4"))
    assert file_mp4._is_file_source() is True

    file_avi = StreamCapture(CameraConfig(source_id="file_cam_avi", uri="sample.avi"))
    assert file_avi._is_file_source() is True


def test_fps_smoothing_seeds_from_first_measurement_no_fake_30(synthetic_video):
    """Verify first frame has None fps_estimate and second frame seeds smoothing."""
    cfg = CameraConfig(
        source_id="fps_test",
        uri=synthetic_video,
        buffer_size=5,
        reconnect_enabled=False,
    )
    capture = StreamCapture(cfg)
    capture.start()

    frame1 = None
    start = time.monotonic()
    while time.monotonic() - start < 2.0:
        frame1 = capture.read(timeout_sec=0.2)
        if frame1 is not None:
            break

    assert frame1 is not None
    # First frame cannot measure delta from past; must be None, not 30.0
    assert frame1.fps_estimate is None

    frame2 = None
    start = time.monotonic()
    while time.monotonic() - start < 2.0:
        frame2 = capture.read(timeout_sec=0.2)
        if frame2 is not None:
            break

    assert frame2 is not None
    # Second frame has measured delta; fps_estimate must be a valid float
    assert frame2.fps_estimate is not None
    assert frame2.fps_estimate > 0.0

    capture.stop()
