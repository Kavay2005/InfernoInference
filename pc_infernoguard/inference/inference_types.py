"""Data models and contracts for inference configuration and detection outputs."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DeviceInfo:
    """Information describing the compute device used for inference.

    Attributes:
        device_str: Normalized PyTorch/Ultralytics device string (e.g., 'cpu', 'cuda:0').
        device_type: Primary device category ('cpu' or 'cuda').
        is_cuda: True if execution uses NVIDIA CUDA acceleration.
        device_name: Descriptive name of the device (e.g., 'CPU' or 'NVIDIA RTX 4090').
        torch_version: PyTorch version string (required fact, no default).
        total_memory_mb: Total VRAM in megabytes for CUDA devices, or None for CPU.
        cuda_version: CUDA runtime version if available.
    """

    device_str: str
    device_type: str
    is_cuda: bool
    device_name: str
    torch_version: str
    total_memory_mb: Optional[float] = None
    cuda_version: Optional[str] = None


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration options for model inference and execution device.

    Attributes:
        model_path: Path to the YOLO weights file (default: 'models/inferno.pt').
        model_name: Model identifier used to tag detections (default: 'inferno').
        device: Device selection string ('cpu', 'cuda', 'cuda:0', or 'auto').
        conf_threshold: Minimum detection confidence threshold (0.0 to 1.0).
        iou_threshold: Intersection over Union (IoU) threshold for NMS.
        imgsz: Inference input resolution in pixels (default: 640).
        half_precision: Enable FP16 half precision (only valid on CUDA).
        max_det: Maximum number of detections allowed per image.
        warmup_iterations: Number of dummy forward passes during startup initialization.
        fallback_to_cpu: Whether to fall back to CPU if CUDA is requested but unavailable.
        expected_classes: Optional mapping of class indices to class names for validation.
        strict_classes: If True, class mismatch raises ValueError; otherwise logs warning.
    """

    model_path: str = "models/inferno.pt"
    model_name: str = "inferno"
    device: str = "auto"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    half_precision: bool = False
    max_det: int = 300
    warmup_iterations: int = 1
    fallback_to_cpu: bool = True
    expected_classes: Optional[Dict[int, str]] = None
    strict_classes: bool = False

    def __post_init__(self):
        """Validate configuration thresholds, stride compatibility, and warmup iterations."""
        if self.conf_threshold < 0.0 or self.conf_threshold > 1.0:
            raise ValueError(f"conf_threshold must be between 0.0 and 1.0, got {self.conf_threshold}")
        if self.iou_threshold < 0.0 or self.iou_threshold > 1.0:
            raise ValueError(f"iou_threshold must be between 0.0 and 1.0, got {self.iou_threshold}")
        if self.imgsz <= 0 or self.imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a positive multiple of 32 (YOLO stride), got {self.imgsz}")
        if self.warmup_iterations < 0:
            raise ValueError(f"warmup_iterations must be non-negative, got {self.warmup_iterations}")


@dataclass(frozen=True)
class DetectionBox:
    """Structured representation of a single detected hazard instance.

    Attributes:
        cls_id: Class integer index (e.g. 0: fire, 1: smoke, 2: spark).
        cls_name: Human-readable class name string ('fire', 'smoke', 'spark').
        confidence: Detection probability score between 0.0 and 1.0.
        xyxy: Bounding box coordinates in absolute pixels: (x1, y1, x2, y2).
        xyxy_norm: Bounding box coordinates normalized to [0.0, 1.0]: (x1_norm, y1_norm, x2_norm, y2_norm).
        model_name: Name of the model that generated this detection (required, no default).
    """

    cls_id: int
    cls_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]
    xyxy_norm: Tuple[float, float, float, float]
    model_name: str

    def __post_init__(self):
        """Validate confidence bounds and non-inverted bounding box coordinates."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        x1, y1, x2, y2 = self.xyxy
        if x1 > x2 or y1 > y2:
            raise ValueError(f"Invalid xyxy bounding box coordinates: {self.xyxy}")


@dataclass(frozen=True)
class InferenceResult:
    """Comprehensive structured result of model inference for a single frame.

    Attributes:
        source_id: Originating camera or stream identifier.
        frame_id: Sequence frame number matching the ingested FrameData.
        timestamp_utc: System UTC timestamp associated with the source frame.
        timestamp_mono: Monotonic clock timestamp associated with the source frame.
        detections: Tuple of DetectionBox objects found in this frame.
        preprocess_ms: Time spent preprocessing (letterboxing, normalization) in ms.
        inference_ms: Time spent executing model forward pass in ms.
        postprocess_ms: Time spent on non-maximum suppression (NMS) in ms.
        total_latency_ms: Ultralytics per-image average latency (prep + inf + post) in ms, not wall time.
        device_used: String identifier of the device that processed this frame.
        image_shape: (height, width) of the original source image.
        end_to_end_ms: Measured wall time in ms: (time.monotonic() - timestamp_mono) * 1000 when result is built.
    """

    source_id: str
    frame_id: int
    timestamp_utc: float
    timestamp_mono: float
    detections: Tuple[DetectionBox, ...]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_latency_ms: float
    device_used: str
    image_shape: Tuple[int, int]
    end_to_end_ms: float
