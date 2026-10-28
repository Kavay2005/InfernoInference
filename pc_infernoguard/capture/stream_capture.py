"""Threaded, resilient single-camera capture with bounded buffering."""

import collections
import copy
import logging
import threading
import time
from typing import Optional, Union
import cv2

from pc_infernoguard.capture.stream_types import FrameData, StreamHealth, StreamStatus
from pc_infernoguard.config import CameraConfig

logger = logging.getLogger("pc_infernoguard.capture")


class StreamCapture:
    """Manages acquisition of video frames from a single camera source in a background thread.

    Guarantees:
    - Bounded buffering (avoids memory leaks and stale frame accumulation).
    - Isolated automatic reconnection with exponential backoff.
    - Real-time health metrics (FPS, total frames, dropped frames).
    - Thread-safe frame retrieval and non-blocking lifecycle management.
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_available = threading.Event()
        self._buffer: collections.deque[FrameData] = collections.deque(maxlen=self.cfg.buffer_size)

        self._health = StreamHealth(source_id=self.cfg.source_id)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_counter = 0

    def start(self) -> None:
        """Start the background capture worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._health.status = StreamStatus.INITIALIZING
            self._thread = threading.Thread(
                target=self._worker_loop,
                name=f"CaptureWorker-{self.cfg.source_id}",
                daemon=True,
            )
            self._thread.start()
            logger.info("Started capture worker for source: %s", self.cfg.source_id)

    def stop(self, timeout_sec: float = 2.0) -> None:
        """Stop capture worker, release OpenCV resources, and mark stream CLOSED.

        Parameters:
            timeout_sec (float): Max seconds to wait for worker thread to exit.
        """
        self._running = False
        self._frame_available.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_sec)

        with self._lock:
            # Only release capture if worker thread has cleanly terminated.
            # Avoid calling cap.release() while worker thread might be inside cap.read().
            if thread is None or not thread.is_alive():
                self._release_capture()
            else:
                logger.warning(
                    "Worker thread for %s did not terminate within %.1fs timeout; leaving capture release to worker",
                    self.cfg.source_id,
                    timeout_sec,
                )
            self._health.status = StreamStatus.CLOSED
            self._health.current_fps = None

        logger.info("Stopped capture worker for source: %s", self.cfg.source_id)

    def is_alive(self) -> bool:
        """Return True if the worker thread is running and active."""
        return self._running and (self._thread is not None) and self._thread.is_alive()

    def read(self, timeout_sec: Optional[float] = None) -> Optional[FrameData]:
        """Read the next available frame, waiting up to timeout_sec if specified.

        Returns:
            FrameData if a frame is retrieved, None otherwise.
        """
        start_time = time.monotonic()
        while self._running:
            with self._lock:
                if self._buffer:
                    frame = self._buffer.popleft()
                    if not self._buffer:
                        self._frame_available.clear()
                    return frame

            if timeout_sec is not None:
                elapsed = time.monotonic() - start_time
                remaining = timeout_sec - elapsed
                if remaining <= 0:
                    return None
                wait_time = min(remaining, 0.05)
            else:
                wait_time = 0.05

            self._frame_available.wait(timeout=wait_time)

        # Final check if stopped with items remaining
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    def read_latest(self) -> Optional[FrameData]:
        """Retrieve the single freshest frame currently in the buffer, dropping older frames."""
        with self._lock:
            if not self._buffer:
                return None
            latest = self._buffer[-1]
            self._buffer.clear()
            self._frame_available.clear()
            return latest

    def get_health(self) -> StreamHealth:
        """Return a snapshot of current stream telemetry and health.

        Returns:
            StreamHealth: Shallow copy of health metrics. If status is ACTIVE and
                          exceeds stale_timeout_sec, the returned copy is marked STALE
                          without mutating the stored internal state.
        """
        with self._lock:
            health_copy = copy.copy(self._health)

        now = time.monotonic()
        if health_copy.status == StreamStatus.ACTIVE and health_copy.is_stale(
            now, self.cfg.stale_timeout_sec
        ):
            health_copy.status = StreamStatus.STALE

        return health_copy

    def _is_file_source(self) -> bool:
        """Check whether the configured source URI is a local video file.

        Parameters:
            None (reads self.cfg.uri)

        Returns:
            bool: True if source is a local video file path, False for live devices/streams.
        """
        if not isinstance(self.cfg.uri, str):
            return False
        uri_clean = self.cfg.uri.strip()
        if uri_clean.isdigit():
            return False
        lower_uri = uri_clean.lower()
        if lower_uri.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "/dev/video")):
            return False
        known_exts = (".avi", ".mp4", ".mkv", ".mov", ".m4v", ".wmv", ".webm", ".flv")
        if lower_uri.endswith(known_exts):
            return True
        import os
        return os.path.isfile(uri_clean)

    def _redacted_uri(self) -> str:
        """Return the camera URI with credentials masked for safe logging and error reporting."""
        from pc_infernoguard.utils.streams_file import redact_uri

        return redact_uri(self.cfg.uri)

    def _parse_source(self, uri: Union[str, int]) -> Union[str, int]:
        """Convert numeric string indices to integer device index for OpenCV."""
        if isinstance(uri, int):
            return uri
        if isinstance(uri, str) and uri.strip().isdigit():
            return int(uri.strip())
        return uri

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Instantiate and configure cv2.VideoCapture."""
        source = self._parse_source(self.cfg.uri)
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                cap.release()
                return None

            # Attempt setting buffer size to 1 in backend to avoid OS-level frame lag
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if self.cfg.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
            if self.cfg.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))
            if self.cfg.fps:
                cap.set(cv2.CAP_PROP_FPS, float(self.cfg.fps))

            return cap
        except Exception as ex:
            logger.warning("Error opening capture for %s: %s", self.cfg.source_id, ex)
            return None

    def _release_capture(self) -> None:
        """Release underlying OpenCV VideoCapture object safely."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as ex:
                logger.warning("Exception during cap.release() for %s: %s", self.cfg.source_id, ex)
            finally:
                self._cap = None

    def _worker_loop(self) -> None:
        """Background thread executing the capture loop with fault isolation."""
        retry_count = 0
        current_delay = self.cfg.reconnect_delay_sec
        fps_smoothing = 0.9
        last_frame_mono: Optional[float] = None

        while self._running:
            # 1. Ensure capture source is opened
            if self._cap is None or not self._cap.isOpened():
                self._release_capture()
                self._cap = self._open_capture()

                if self._cap is None or not self._cap.isOpened():
                    with self._lock:
                        if self.cfg.reconnect_enabled:
                            if self.cfg.max_retries is not None and retry_count >= self.cfg.max_retries:
                                self._health.status = StreamStatus.FAILED
                                self._health.error_message = (
                                    f"Max retries ({self.cfg.max_retries}) exceeded"
                                )
                                logger.error(
                                    "Stream %s failed after %d retries.",
                                    self.cfg.source_id,
                                    retry_count,
                                )
                                break
                            self._health.status = StreamStatus.RECONNECTING
                            self._health.reconnect_count += 1
                            self._health.error_message = (
                                f"Failed to open source {self._redacted_uri()} (attempt {retry_count + 1})"
                            )
                        else:
                            self._health.status = StreamStatus.FAILED
                            self._health.error_message = f"Failed to open source {self._redacted_uri()}"
                            logger.error(
                                "Stream %s failed to open (reconnect disabled).",
                                self.cfg.source_id,
                            )
                            break

                    retry_count += 1
                    # Exponential backoff sleep
                    sleep_until = time.monotonic() + current_delay
                    while self._running and time.monotonic() < sleep_until:
                        time.sleep(0.05)
                    current_delay = min(
                        current_delay * self.cfg.reconnect_backoff_factor,
                        self.cfg.max_reconnect_delay_sec,
                    )
                    continue

                # Successfully opened
                with self._lock:
                    self._health.status = StreamStatus.ACTIVE
                    self._health.error_message = None
                retry_count = 0
                current_delay = self.cfg.reconnect_delay_sec

            # 2. Read frame from OpenCV
            t_read_start = time.monotonic()
            ret, raw_image = self._cap.read()
            t_utc = time.time()
            t_mono = time.monotonic()

            if not ret or raw_image is None or raw_image.size == 0:
                # Check if source is a local video file that reached EOF; loop only if cfg.loop_video is True
                if self.cfg.loop_video and self._is_file_source() and self._cap is not None:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_image = self._cap.read()
                    t_utc = time.time()
                    t_mono = time.monotonic()

            if not ret or raw_image is None or raw_image.size == 0:
                with self._lock:
                    if self.cfg.reconnect_enabled:
                        if self.cfg.max_retries is not None and retry_count >= self.cfg.max_retries:
                            self._health.status = StreamStatus.FAILED
                            self._health.error_message = "Stream interrupted, max retries reached"
                            break
                        self._health.status = StreamStatus.RECONNECTING
                        self._health.reconnect_count += 1
                        self._health.error_message = "Stream read returned empty frame"
                    else:
                        self._health.status = StreamStatus.FAILED
                        self._health.error_message = "Stream read failed (reconnect disabled)"
                        break

                self._release_capture()
                retry_count += 1
                sleep_until = time.monotonic() + current_delay
                while self._running and time.monotonic() < sleep_until:
                    time.sleep(0.05)
                current_delay = min(
                    current_delay * self.cfg.reconnect_backoff_factor,
                    self.cfg.max_reconnect_delay_sec,
                )
                continue

            # 3. Process valid frame and update health
            retry_count = 0
            current_delay = self.cfg.reconnect_delay_sec
            h, w = raw_image.shape[:2]

            # Calculate FPS: no fake 30.0 on the first frame.
            # Seed smoothing with the first measured delta once 2 frames exist.
            smoothed_fps: Optional[float] = None
            if last_frame_mono is not None:
                delta = t_mono - last_frame_mono
                if delta > 0:
                    inst_fps = 1.0 / delta
                    with self._lock:
                        prev_fps = self._health.current_fps
                    if prev_fps is not None:
                        smoothed_fps = (fps_smoothing * prev_fps) + ((1.0 - fps_smoothing) * inst_fps)
                    else:
                        smoothed_fps = inst_fps
            last_frame_mono = t_mono

            self._frame_counter += 1
            frame_data = FrameData(
                source_id=self.cfg.source_id,
                frame_id=self._frame_counter,
                timestamp_utc=t_utc,
                timestamp_mono=t_mono,
                image=raw_image,
                width=w,
                height=h,
                fps_estimate=smoothed_fps,
            )

            with self._lock:
                if len(self._buffer) >= self.cfg.buffer_size:
                    self._health.dropped_frames += 1
                self._buffer.append(frame_data)
                self._health.total_frames_read += 1
                self._health.last_frame_timestamp = t_mono
                self._health.current_fps = smoothed_fps
                self._health.status = StreamStatus.ACTIVE
                self._health.error_message = None
                self._frame_available.set()

            # Pacing: sleep to a target FPS ONLY when cfg.pace_fps is True AND the source is a local video file.
            # For live cameras (RTSP/USB) never sleep to pace; read continuously so buffers cannot back up.
            if self.cfg.pace_fps and self._is_file_source():
                src_fps = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
                target_fps: Optional[float] = None
                if self.cfg.fps is not None and 0 < self.cfg.fps < 1000:
                    target_fps = float(self.cfg.fps)
                elif 0 < src_fps < 1000:
                    target_fps = float(src_fps)

                if target_fps is not None:
                    target_interval = 1.0 / target_fps
                    elapsed = time.monotonic() - t_read_start
                    sleep_needed = target_interval - elapsed
                    if sleep_needed > 0:
                        time.sleep(sleep_needed)

        self._release_capture()
