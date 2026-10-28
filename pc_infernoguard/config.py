"""Configuration data models for camera inputs and multi-stream manager."""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class CameraConfig:
    """Configuration for an individual camera stream.

    Attributes:
        source_id: Unique logical identifier for this camera stream (e.g., 'webcam_0').
        uri: Camera stream URI. Can be an integer device index (e.g. 0), string integer ("0"),
             RTSP/HTTP URL ("rtsp://..."), or path to a video file.
        buffer_size: Maximum size of the internal capture queue. Default is 1 to guarantee
                     retrieval of the freshest frame and prevent latency buildup.
        reconnect_enabled: Whether to attempt automatic reconnection when the stream drops.
        reconnect_delay_sec: Initial reconnection delay in seconds.
        reconnect_backoff_factor: Exponential multiplier for reconnection delay.
        max_reconnect_delay_sec: Maximum ceiling for exponential reconnection backoff.
        max_retries: Maximum number of reconnection attempts before marking stream as FAILED.
                     None means infinite retries.
        stale_timeout_sec: Inactivity threshold in seconds beyond which stream is marked STALE.
        width: Optional requested capture width.
        height: Optional requested capture height.
        fps: Optional requested capture frame rate.
    """

    source_id: str
    uri: Union[str, int]
    buffer_size: int = 1
    reconnect_enabled: bool = True
    reconnect_delay_sec: float = 1.0
    reconnect_backoff_factor: float = 2.0
    max_reconnect_delay_sec: float = 30.0
    max_retries: Optional[int] = None
    stale_timeout_sec: float = 5.0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    loop_video: bool = False
    pace_fps: bool = True

    def __post_init__(self):
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError(f"source_id must be a non-empty string, got {self.source_id!r}")
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {self.buffer_size}")


@dataclass
class MultiStreamConfig:
    """Configuration for orchestrating multiple camera streams simultaneously.

    Attributes:
        cameras: List of CameraConfig specifications to initialize and manage.
        poll_interval_sec: Sleep interval in seconds between polling passes in iter_batches.
        max_frame_age_sec: Optional maximum permissible age of a frame (now - timestamp_mono) in seconds.
                           When configured, read_batch omits frames older than this threshold.
    """

    cameras: List[CameraConfig] = field(default_factory=list)
    poll_interval_sec: float = 0.01
    max_frame_age_sec: Optional[float] = None


# Re-export InferenceConfig for top-level convenience
from pc_infernoguard.inference.inference_types import InferenceConfig  # noqa: E402

__all__ = ["CameraConfig", "MultiStreamConfig", "InferenceConfig"]
