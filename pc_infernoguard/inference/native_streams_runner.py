"""Runner for Ultralytics native .streams inference comparison mode."""

import logging
import time
from pathlib import Path
from typing import Generator, List, Optional, Union

from ultralytics import YOLO

from pc_infernoguard.inference.inference_types import DetectionBox, InferenceResult

logger = logging.getLogger("pc_infernoguard.native_streams")


class NativeStreamsRunner:
    """Runs inference directly using Ultralytics native .streams mode for baseline comparison.

    ARCHITECTURAL LIMITATIONS & TRADE-OFFS:
    As documented in AGENTS.md Section 6 & 7 and ADR 0001, this runner relies directly on
    Ultralytics LoadStreams with the following known edge failure modes:
    1. No reconnect: If any single stream drops, stalls, or terminates, Ultralytics terminates
       all concurrent streams simultaneously.
    2. No per-camera health: Does not track individual stream health transitions (INITIALIZING,
       ACTIVE, STALE, RECONNECTING, FAILED) or frame drop counts.
    3. No capture timestamps: Frames are not timestamped at camera capture; arrival monotonic
       time is used as a fallback.
    4. Black frames on dropped streams: When a network stream hiccups, OpenCV/FFmpeg backend
       yields empty or black frames without state signaling.
    5. Startup failure: Startup fails immediately and aborts the process if any camera URI in
       the .streams file is unreachable.

    This mode is provided exclusively for benchmarking and profiling against the resilient
    InfernoGuard multi-stream ingestion architecture.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/inferno.pt",
        streams_file: Union[str, Path] = "cameras.streams",
        device: str = "auto",
        conf: float = 0.25,
        imgsz: int = 640,
        model_name: str = "native_ultralytics",
    ):
        """Initialize NativeStreamsRunner with model weights and target .streams file.

        Parameters:
            model_path (Union[str, Path]): Path to YOLO weights file (.pt).
            streams_file (Union[str, Path]): Path to an Ultralytics-compatible .streams file.
            device (str): Compute device ('auto', 'cpu', 'cuda', etc.).
            conf (float): Detection confidence threshold.
            imgsz (int): Input inference resolution.
            model_name (str): Identifier used to tag detections.
        """
        from pc_infernoguard.utils.streams_file import (
            parse_streams_file,
            redact_uri,
            source_id_for,
        )

        self.streams_path = Path(streams_file)
        if not self.streams_path.is_file():
            raise FileNotFoundError(f"Streams file not found: {streams_file}")

        self.sources = parse_streams_file(self.streams_path)
        if not self.sources:
            raise ValueError(f"Streams file '{streams_file}' is empty or contains no valid URIs.")

        self.num_cameras = len(self.sources)
        ids_so_far = set()
        self.source_ids: List[str] = []
        for idx, uri in enumerate(self.sources):
            sid = source_id_for(uri, idx, taken=ids_so_far)
            ids_so_far.add(sid)
            self.source_ids.append(sid)


        logger.info(
            "Initializing NativeStreamsRunner for %d streams: %s",
            self.num_cameras,
            [redact_uri(s) for s in self.sources],
        )

        self.model_path = str(model_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model_name = model_name
        self.frame_count = 0
        self._model: Optional[YOLO] = None

    def _get_model(self) -> YOLO:
        """Lazily load YOLO model weights on first stream invocation."""
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def stream(
        self,
        max_batches: Optional[int] = None,
    ) -> Generator[InferenceResult, None, None]:
        """Stream inference results directly from Ultralytics predictor generator.

        Maps each Results object to cameras in strict file order (camera 0, camera 1, ..., camera N-1).

        Parameters:
            max_batches (Optional[int]): Optional limit on total number of results to yield.

        Yields:
            InferenceResult: Structured detection result for each camera in order.
        """
        model = self._get_model()

        # Execute streaming inference via Ultralytics generator
        stream_gen = model.predict(
            source=str(self.streams_path),
            device=self.device,
            conf=self.conf,
            imgsz=self.imgsz,
            stream=True,
            stream_buffer=False,
            verbose=False,
        )

        for res in stream_gen:
            t_arrival_utc = time.time()
            t_arrival_mono = time.monotonic()

            # Map Results to camera by order
            cam_idx = self.frame_count % self.num_cameras
            source_id = self.source_ids[cam_idx]
            self.frame_count += 1

            # Extract detections
            det_boxes: List[DetectionBox] = []
            h, w = (
                res.orig_shape
                if hasattr(res, "orig_shape") and res.orig_shape is not None
                else (self.imgsz, self.imgsz)
            )

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy_cpu = res.boxes.xyxy.cpu().numpy()
                conf_cpu = res.boxes.conf.cpu().numpy()
                cls_cpu = res.boxes.cls.cpu().numpy()
                names = res.names if hasattr(res, "names") else {}

                for i in range(len(xyxy_cpu)):
                    x1, y1, x2, y2 = xyxy_cpu[i]
                    c = float(conf_cpu[i])
                    class_id = int(cls_cpu[i])
                    class_name = names.get(class_id, f"class_{class_id}")
                    det_boxes.append(
                        DetectionBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=c,
                            class_id=class_id,
                            class_name=class_name,
                            model_name=self.model_name,
                            xyxy_norm=(
                                float(x1) / w if w > 0 else 0.0,
                                float(y1) / h if h > 0 else 0.0,
                                float(x2) / w if w > 0 else 0.0,
                                float(y2) / h if h > 0 else 0.0,
                            ),
                        )
                    )

            speed = getattr(res, "speed", {}) or {}
            prep_ms = float(speed.get("preprocess", 0.0))
            inf_ms = float(speed.get("inference", 0.0))
            post_ms = float(speed.get("postprocess", 0.0))
            tot_ms = prep_ms + inf_ms + post_ms

            device_used = str(getattr(res, "device", self.device))

            yield InferenceResult(
                source_id=source_id,
                frame_id=self.frame_count,
                timestamp_utc=t_arrival_utc,
                timestamp_mono=t_arrival_mono,
                detections=tuple(det_boxes),
                preprocess_ms=prep_ms,
                inference_ms=inf_ms,
                postprocess_ms=post_ms,
                total_latency_ms=tot_ms,
                device_used=device_used,
                image_shape=(int(h), int(w)),
                end_to_end_ms=tot_ms,
            )

            if max_batches is not None and self.frame_count >= max_batches:
                break
