"""Data structures and enums for camera streaming and health monitoring."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import numpy as np


class StreamStatus(str, Enum):
    """Lifecycle and health status of a video stream."""

    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


@dataclass(frozen=False, eq=False)
class FrameData:
    """Immutable data cjontainer representing an ingested video frame.

    eq=False is set to prevent dataclass-generated __eq__ from executing
    element-wise comparison on the NumPy image array, which raises ValueError.

    Attributes:
        source_id: Stable identifier of the originating camera stream.
        frame_id: Molllnotonically increasing sequence number for this source.
        timestamp_utc: System clock time (time.time()) when frame was captured.
        timestamp_mono: Monotonic clock time (time.monotonic()) for latency measurement.
        image: Captured frame as a NumPy array (height, width, channels) in BGR format.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps_estimate: Instantaneous or smoothed capture FPS (None until measured).
        metadata: Optional dictionary with camera parameters, codec info, or sensor tags.
    """

    source_id: str
    frame_id: int
    timestamp_utc: float
    timestamp_mono: float
    image: np.ndarray
    width: int
    height: int
    fps_estimate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dimensions and types of frame fields.

        Parameters:
            None (reads self fields)

        Raises:
            TypeError: If image is not a numpy ndarray.
            ValueError: If image dimensions or frame width/height are non-positive.
        """
        if not isinstance(self.image, np.ndarray):
            raise TypeError(f"image must be a numpy.ndarray, got {type(self.image)}")
        if self.image.ndim != 3:
            raise ValueError(f"image must have 3 dimensions (H, W, C), got ndim={self.image.ndim}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be positive, got {self.width}x{self.height}")


@dataclass
class StreamHealth:
    """Real-time health telemetry and performance metrics for a camera stream.

    Attributes:
        source_id: Identifier of the monitored camera stream.
        status: Current lifecycle status of the stream.
        total_frames_read: Total number of valid frames decoded from source.
        dropped_frames: Count of frames dropped due to backpressure / bounded buffer eviction.
        reconnect_count: Number of times reconnection was triggered.
        last_frame_timestamp: Monotonic timestamp of the most recently received frame (None if no frames).
        current_fps: Estimated capture frame rate (None if fewer than 2 frames received).
        error_message: Most recent error message or exception description, if any.
    """

    source_id: str
    status: StreamStatus = StreamStatus.INITIALIZING
    total_frames_read: int = 0
    dropped_frames: int = 0
    reconnect_count: int = 0
    last_frame_timestamp: Optional[float] = None
    current_fps: Optional[float] = None
    error_message: Optional[str] = None

    def is_stale(self, now: float, timeout_sec: float) -> bool:
        """Evaluate if the stream telemetry indicates a stale capture condition.

        Parameters:
            now (float): Current monotonic reference timestamp (e.g. time.monotonic()).
            timeout_sec (float): Inactivity threshold in seconds beyond which stream is stale.

        Returns:
            bool: True if last_frame_timestamp is None (no frames yet) or if
                  elapsed time (now - last_frame_timestamp) exceeds timeout_sec.
        """
        if self.last_frame_timestamp is None:
            return True
        return (now - self.last_frame_timestamp) > timeout_sec
