"""Thread-safe synchronization wrapper for InferenceEngine."""

import logging
import threading
from typing import Dict, List, Sequence

from pc_infernoguard.capture.stream_types import FrameData
from pc_infernoguard.inference.inference_engine import InferenceEngine
from pc_infernoguard.inference.inference_types import (
    DeviceInfo,
    InferenceConfig,
    InferenceResult,
)

logger = logging.getLogger("pc_infernoguard.thread_safe")


class ThreadSafeInferenceWrapper:
    """Thread-safe locking wrapper around an InferenceEngine instance.

    Ultralytics YOLO model instances maintain internal mutable state (predictors,
    tensor buffers, preprocessing contexts) that are NOT thread-safe when accessed
    concurrently by multiple threads.

    Reference: https://docs.ultralytics.com/guides/yolo-thread-safe-inference/
    As documented in the official Ultralytics thread-safety guide, a shared model instance
    must be serialized with a lock to guarantee thread isolation during forward passes.

    This wrapper provides:
    1. Standard mutual exclusion lock (threading.Lock) serialization for all forward-pass calls.
    2. Transparent pass-through of the exact InferenceEngine interface.
    3. Thread-safe call count diagnostics.
    """

    def __init__(self, engine: InferenceEngine):
        """Wrap an InferenceEngine with thread-safe lock serialization.

        Parameters:
            engine (InferenceEngine): Underlying InferenceEngine instance to serialize.
        """
        self._engine = engine
        self._lock = threading.Lock()
        self._call_count = 0

    @property
    def engine(self) -> InferenceEngine:
        """Access the underlying unwrapped InferenceEngine."""
        return self._engine

    @property
    def cfg(self) -> InferenceConfig:
        """Inference configuration of the underlying engine."""
        return self._engine.cfg

    @property
    def resolved_device(self) -> str:
        """Compute device string (e.g., 'cpu', 'cuda:0')."""
        return self._engine.resolved_device

    @property
    def device_info(self) -> DeviceInfo:
        """Hardware device info metadata."""
        return self._engine.device_info

    @property
    def class_names(self) -> Dict[int, str]:
        """Class name mapping for detected hazards."""
        return self._engine.class_names

    @property
    def is_warm(self) -> bool:
        """True if the model has undergone startup warmup passes."""
        return self._engine.is_warm

    @property
    def call_count(self) -> int:
        """Total number of inference calls executed through this wrapper."""
        with self._lock:
            return self._call_count

    def predict_frame(self, frame: FrameData) -> InferenceResult:
        """Thread-safely predict hazard detections for a single frame.

        Parameters:
            frame (FrameData): Ingested FrameData packet.

        Returns:
            InferenceResult: Structured InferenceResult containing detections and latency metrics.
        """
        with self._lock:
            self._call_count += 1
            return self._engine.predict_frame(frame)

    def predict_batch(self, frames: Sequence[FrameData]) -> List[InferenceResult]:
        """Thread-safely execute batched model prediction across a sequence of FrameData packets.

        Parameters:
            frames (Sequence[FrameData]): Sequence of ingested FrameData packets.

        Returns:
            List[InferenceResult]: List of InferenceResult objects.
        """
        with self._lock:
            self._call_count += len(frames)
            return self._engine.predict_batch(frames)

    def __enter__(self) -> "ThreadSafeInferenceWrapper":
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._lock.release()
