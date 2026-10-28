"""Multi-stream manager providing stream orchestration and failure isolation."""

import logging
import threading
import time
from typing import Dict, Generator, Optional
from pc_infernoguard.capture.stream_capture import StreamCapture
from pc_infernoguard.capture.stream_types import FrameData, StreamHealth
from pc_infernoguard.config import CameraConfig, MultiStreamConfig

logger = logging.getLogger("pc_infernoguard.manager")


class MultiStreamManager:
    """Orchestrates multiple StreamCapture instances with independent fault isolation.

    Architectural properties:
    - Batches are NOT synchronized across cameras; they represent the freshest available frame
      per camera captured at poll time.
    - Zero cascade failure: a failure or disconnection in stream A never interrupts stream B.
    - Consolidated health reporting and telemetry across all cameras.
    - Thread-safe stream registration and lifecycle management via threading.Lock.
    - Clean resource management via Python context manager interface.
    """

    def __init__(self, cfg: MultiStreamConfig):
        """Initialize MultiStreamManager with configured camera definitions.

        Parameters:
            cfg (MultiStreamConfig): Configuration containing camera list and polling options.
        """
        self.cfg = cfg
        self._lock = threading.Lock()
        self._streams: Dict[str, StreamCapture] = {}
        for cam_cfg in self.cfg.cameras:
            self._streams[cam_cfg.source_id] = StreamCapture(cam_cfg)

    def __enter__(self) -> "MultiStreamManager":
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_all()

    def start_all(self) -> None:
        """Start all configured camera streams."""
        with self._lock:
            stream_items = list(self._streams.items())
        for source_id, capture in stream_items:
            capture.start()
        logger.info("Started all %d camera streams", len(stream_items))

    def stop_all(self, timeout_sec: float = 2.0) -> None:
        """Stop all streams and wait for threads to exit cleanly.

        Parameters:
            timeout_sec (float): Max seconds to wait per stream worker join.
        """
        with self._lock:
            stream_items = list(self._streams.items())
        for source_id, capture in stream_items:
            capture.stop(timeout_sec=timeout_sec)
        logger.info("Stopped all camera streams")

    def is_any_alive(self) -> bool:
        """Return True if at least one stream capture thread is running."""
        with self._lock:
            stream_items = list(self._streams.items())
        return any(capture.is_alive() for _, capture in stream_items)

    def read_batch(self) -> Dict[str, FrameData]:
        """Fetch the freshest frame from each stream.

        Batches are NOT synchronized; they are the latest frame per camera at poll time.

        Parameters:
            None (reads from managed streams)

        Returns:
            Dict[str, FrameData]: Dict mapping source_id to FrameData. Streams that have
            no new frame or are reconnecting are omitted without blocking.
            If max_frame_age_sec is set in MultiStreamConfig, frames older than that
            threshold relative to the current monotonic clock are omitted.
        """
        batch: Dict[str, FrameData] = {}
        now_mono = time.monotonic()
        with self._lock:
            stream_items = list(self._streams.items())

        for source_id, capture in stream_items:
            try:
                frame = capture.read_latest()
                if frame is not None:
                    if (
                        self.cfg.max_frame_age_sec is not None
                        and (now_mono - frame.timestamp_mono) > self.cfg.max_frame_age_sec
                    ):
                        logger.debug(
                            "Omitted frame from %s: age %.3fs exceeds max_frame_age_sec %.3fs",
                            source_id,
                            now_mono - frame.timestamp_mono,
                            self.cfg.max_frame_age_sec,
                        )
                        continue
                    batch[source_id] = frame
            except Exception as ex:
                logger.error("Unexpected error reading from stream %s: %s", source_id, ex)
        return batch

    def iter_batches(
        self, poll_interval_sec: Optional[float] = None
    ) -> Generator[Dict[str, FrameData], None, None]:
        """Yield batches of latest frames while streams are running.

        Parameters:
            poll_interval_sec (Optional[float]): Sleep interval between polling passes.
        """
        interval = (
            poll_interval_sec
            if poll_interval_sec is not None
            else self.cfg.poll_interval_sec
        )
        while self.is_any_alive():
            batch = self.read_batch()
            if batch:
                yield batch
            time.sleep(interval)

    def get_health_report(self) -> Dict[str, StreamHealth]:
        """Return real-time health snapshot for all managed streams."""
        with self._lock:
            stream_items = list(self._streams.items())
        return {
            source_id: capture.get_health()
            for source_id, capture in stream_items
        }

    def add_stream(self, cam_cfg: CameraConfig, auto_start: bool = True) -> None:
        """Dynamically attach and optionally start a new camera stream.

        Parameters:
            cam_cfg (CameraConfig): Configuration for the new stream.
            auto_start (bool): Whether to immediately spawn the worker thread.
        """
        with self._lock:
            if cam_cfg.source_id in self._streams:
                raise ValueError(f"Stream {cam_cfg.source_id} already exists")
            capture = StreamCapture(cam_cfg)
            self._streams[cam_cfg.source_id] = capture

        if auto_start:
            capture.start()
        logger.info("Added new stream: %s", cam_cfg.source_id)

    def remove_stream(self, source_id: str, timeout_sec: float = 2.0) -> None:
        """Stop and remove a camera stream.

        Parameters:
            source_id (str): Stream identifier to remove.
            timeout_sec (float): Max seconds to wait for worker join.
        """
        with self._lock:
            capture = self._streams.pop(source_id, None)

        if capture is not None:
            capture.stop(timeout_sec=timeout_sec)
            logger.info("Removed stream: %s", source_id)
