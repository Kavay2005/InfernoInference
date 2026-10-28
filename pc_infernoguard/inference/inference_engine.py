"""Unified InferenceEngine wrapping Ultralytics YOLO with explicit CPU/CUDA selection."""

import logging
import time
from typing import Dict, List, Optional, Sequence
import numpy as np
from ultralytics import YOLO

from pc_infernoguard.capture.stream_types import FrameData
from pc_infernoguard.inference.device_manager import DeviceManager
from pc_infernoguard.inference.inference_types import (
    DetectionBox,
    DeviceInfo,
    InferenceConfig,
    InferenceResult,
)

logger = logging.getLogger("pc_infernoguard.inference")


class InferenceEngine:
    """Production inference engine for pc-infernoguard-inference.

    Adheres strictly to AGENTS.md Section 8:
    - Explicit device selection and configuration (no hidden constants).
    - Model loaded once at startup.
    - Explicit runtime warmup pass before streaming starts.
    - Unified application-level detection interface across CPU and GPU backends.
    - Structured output isolating model execution from downstream safety state logic.
    """

    def __init__(self, cfg: Optional[InferenceConfig] = None):
        """Initialize InferenceEngine, load YOLO model, validate classes, and execute warmup.

        Parameters:
            cfg (Optional[InferenceConfig]): Configuration options for device, model, and thresholds.
        """
        self.cfg = cfg or InferenceConfig()

        # 1. Resolve and validate compute device
        self.resolved_device = DeviceManager.resolve_device(
            self.cfg.device,
            fallback_to_cpu=self.cfg.fallback_to_cpu,
        )
        self.device_info: DeviceInfo = DeviceManager.get_device_info(self.resolved_device)
        logger.info(
            "Initializing InferenceEngine on device: %s (%s)",
            self.resolved_device,
            self.device_info.device_name,
        )

        # 2. Precision check: FP16 is only valid and beneficial on CUDA
        self.half_precision = self.cfg.half_precision
        if self.half_precision and not self.device_info.is_cuda:
            logger.warning("FP16 half_precision requested on CPU; disabling FP16 to avoid errors.")
            self.half_precision = False

        # Check whether installed Ultralytics supports 'quantize' in predict mode
        # Ultralytics 8.4+ introduces 'quantize=16' in place of deprecated 'half=True'
        try:
            from ultralytics.cfg import get_cfg
            self._supports_quantize = hasattr(get_cfg(), "quantize")
        except Exception:
            self._supports_quantize = False

        # 3. Load model weights once during startup
        t0 = time.monotonic()
        logger.info("Loading YOLO model from: %s", self.cfg.model_path)
        self.model = YOLO(self.cfg.model_path)
        load_time_ms = (time.monotonic() - t0) * 1000.0
        logger.info("YOLO model loaded in %.1f ms", load_time_ms)

        # 4. Validate model classes and task
        self.class_names: Dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        self._validate_model_metadata()

        # 5. Model Warmup
        self.is_warm = False
        if self.cfg.warmup_iterations > 0:
            self._warmup(self.cfg.warmup_iterations)

    def _validate_model_metadata(self) -> None:
        """Verify model task and validate class mappings against cfg.expected_classes.

        Parameters:
            None (reads self.model, self.class_names, and self.cfg)

        Raises:
            ValueError: If expected_classes is configured and mismatches actual classes,
                        and cfg.strict_classes is True.
        """
        if hasattr(self.model, "task") and self.model.task != "detect":
            logger.warning(
                "Model task is '%s', expected 'detect'. Inference may produce unexpected outputs.",
                self.model.task,
            )

        if self.cfg.expected_classes is None:
            logger.debug("No expected_classes specified; skipping class mapping validation.")
            return

        missing_classes = []
        for cid, expected_name in self.cfg.expected_classes.items():
            actual = self.class_names.get(cid)
            if actual != expected_name:
                missing_classes.append(f"{cid}: expected '{expected_name}', got '{actual}'")

        if missing_classes:
            msg = (
                f"Model class mapping mismatch with expected_classes: {missing_classes}. "
                f"Current classes: {self.class_names}"
            )
            if self.cfg.strict_classes:
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info("Model class mapping verified against expected_classes: %s", self.class_names)

    def _warmup(self, iterations: int) -> None:
        """Perform warmup forward passes using dummy inputs to initialize runtime and JIT caches.

        Parameters:
            iterations (int): Number of forward passes to execute.
        """
        logger.info(
            "Running %d warmup pass(es) at resolution %dx%d on %s...",
            iterations,
            self.cfg.imgsz,
            self.cfg.imgsz,
            self.resolved_device,
        )
        dummy_img = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        t_start = time.monotonic()

        warmup_kwargs = {
            "source": dummy_img,
            "device": self.resolved_device,
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.conf_threshold,
            "iou": self.cfg.iou_threshold,
            "verbose": False,
        }
        if self.half_precision:
            if self._supports_quantize:
                # Ultralytics 8.4+ uses 'quantize=16' for FP16 half precision
                warmup_kwargs["quantize"] = 16
            else:
                # Pre-8.4 Ultralytics uses deprecated 'half=True'
                warmup_kwargs["half"] = True

        for idx in range(iterations):
            _ = self.model.predict(**warmup_kwargs)

        warmup_ms = (time.monotonic() - t_start) * 1000.0
        self.is_warm = True
        logger.info("Model warmup completed in %.1f ms", warmup_ms)

    def predict_frame(self, frame: FrameData) -> InferenceResult:
        """Run detection on a single FrameData container.

        Parameters:
            frame (FrameData): Ingested FrameData packet from StreamCapture.

        Returns:
            InferenceResult: Structured InferenceResult containing detections and latency metrics.
        """
        results = self.predict_batch([frame])
        return results[0]

    def predict_batch(self, frames: Sequence[FrameData]) -> List[InferenceResult]:
        """Execute batched model prediction across a sequence of FrameData packets.

        Parameters:
            frames (Sequence[FrameData]): Sequence of ingested FrameData objects.

        Returns:
            List[InferenceResult]: List of InferenceResult dataclasses matching input order.

        Raises:
            TypeError: If any item in frames is not an instance of FrameData.
            RuntimeError: If Ultralytics result lacks required speed profiling dictionary.
        """
        if not frames:
            return []

        for f in frames:
            if not isinstance(f, FrameData):
                raise TypeError(f"All elements in frames must be FrameData instances, got {type(f)}")

        images = [f.image for f in frames]
        predict_kwargs = {
            "source": images,
            "device": self.resolved_device,
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.conf_threshold,
            "iou": self.cfg.iou_threshold,
            "max_det": self.cfg.max_det,
            "verbose": False,
        }
        if self.half_precision:
            if self._supports_quantize:
                # Ultralytics 8.4+ uses 'quantize=16' for FP16 half precision
                predict_kwargs["quantize"] = 16
            else:
                # Pre-8.4 Ultralytics uses deprecated 'half=True'
                predict_kwargs["half"] = True

        raw_results = self.model.predict(**predict_kwargs)

        now_mono = time.monotonic()
        output_results: List[InferenceResult] = []
        for idx, res in enumerate(raw_results):
            frame = frames[idx]
            h, w = frame.image.shape[:2]

            # Read res.speed keys directly; raise if speed is missing
            speed = getattr(res, "speed", None)
            if speed is None:
                raise RuntimeError(
                    f"Ultralytics result for source '{frame.source_id}' is missing required 'speed' telemetry."
                )

            prep_ms = float(speed["preprocess"])
            inf_ms = float(speed["inference"])
            post_ms = float(speed["postprocess"])
            total_lat = prep_ms + inf_ms + post_ms

            # Extract detections
            boxes = res.boxes
            detection_boxes: List[DetectionBox] = []
            if boxes is not None and len(boxes) > 0:
                xyxy_tensor = boxes.xyxy.cpu().numpy()
                xyxyn_tensor = boxes.xyxyn.cpu().numpy()
                conf_tensor = boxes.conf.cpu().numpy()
                cls_tensor = boxes.cls.cpu().numpy()

                for b_idx in range(len(boxes)):
                    cid = int(cls_tensor[b_idx])
                    conf_val = float(conf_tensor[b_idx])
                    b_xyxy = tuple(float(v) for v in xyxy_tensor[b_idx])
                    b_xyxyn = tuple(float(v) for v in xyxyn_tensor[b_idx])

                    detection_boxes.append(
                        DetectionBox(
                            cls_id=cid,
                            cls_name=self.class_names.get(cid, f"class_{cid}"),
                            confidence=conf_val,
                            xyxy=b_xyxy,  # type: ignore
                            xyxy_norm=b_xyxyn,  # type: ignore
                            model_name=self.cfg.model_name,
                        )
                    )

            e2e_ms = (now_mono - frame.timestamp_mono) * 1000.0

            output_results.append(
                InferenceResult(
                    source_id=frame.source_id,
                    frame_id=frame.frame_id,
                    timestamp_utc=frame.timestamp_utc,
                    timestamp_mono=frame.timestamp_mono,
                    detections=tuple(detection_boxes),
                    preprocess_ms=prep_ms,
                    inference_ms=inf_ms,
                    postprocess_ms=post_ms,
                    total_latency_ms=total_lat,
                    device_used=self.resolved_device,
                    image_shape=(h, w),
                    end_to_end_ms=e2e_ms,
                )
            )

        return output_results
