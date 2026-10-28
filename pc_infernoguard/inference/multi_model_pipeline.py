"""Multi-model pipeline orchestrating multiple specialized models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pc_infernoguard.capture.stream_types import FrameData
from pc_infernoguard.inference.inference_engine import InferenceEngine
from pc_infernoguard.inference.inference_types import DetectionBox, InferenceResult
from pc_infernoguard.inference.thread_safe_engine import ThreadSafeInferenceWrapper

logger = logging.getLogger("pc_infernoguard.multi_model")


class MultiModelPipeline:
    """Orchestrates execution of multiple specialized YOLO models.

    Adheres to AGENTS.md Section 8 and ADR 0003:
    - Maintains explicit model identity and class mappings per model.
    - Isolates execution while retaining source stream and frame identity.
    - Tags each detection with its originating model name.
    - Isolates model failures: a runtime exception in one model does not crash the loop
      and preserves results from healthy sibling models.
    - Runs models sequentially in the single inference worker. Each engine owns its own
      YOLO instance. ThreadSafeInferenceWrapper is used only when an engine is shared across threads.
    """

    def __init__(self):
        """Initialize empty registry for specialized model engines and error tracking."""
        self._models: Dict[str, Union[InferenceEngine, ThreadSafeInferenceWrapper]] = {}
        self.last_errors: Dict[str, Optional[str]] = {}

    @classmethod
    def from_manifest(cls, path: Union[str, Path]) -> "MultiModelPipeline":
        """Construct a MultiModelPipeline from a models.yaml manifest file.

        Loads each model entry, instantiates an InferenceEngine, and registers it.

        Parameters:
            path (Union[str, Path]): Path to the models.yaml file.

        Returns:
            MultiModelPipeline: Initialized pipeline with registered engines.
        """
        from pc_infernoguard.inference.models_manifest import load_models_manifest

        configs = load_models_manifest(path)
        pipeline = cls()
        for cfg in configs:
            engine = InferenceEngine(cfg)
            pipeline.register_model(cfg.model_name, engine)
        return pipeline

    def register_model(
        self,
        name: str,
        engine: Union[InferenceEngine, ThreadSafeInferenceWrapper],
    ) -> None:
        """Register a named model engine into the pipeline.

        Requires that engine was already built with an InferenceConfig whose model_name
        equals the registered name; does not mutate engine.cfg.

        Parameters:
            name (str): Unique identifier for this model (e.g., 'primary_inferno', 'spark_verifier').
            engine (Union[InferenceEngine, ThreadSafeInferenceWrapper]): Configured engine or wrapper.

        Raises:
            ValueError: If model name is already registered or if engine.cfg.model_name != name.
        """
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered in the pipeline.")

        engine_model_name = getattr(getattr(engine, "cfg", None), "model_name", None)
        if engine_model_name != name:
            raise ValueError(
                f"Model name mismatch: registration name '{name}' does not match engine config model_name '{engine_model_name}'"
            )

        self._models[name] = engine
        self.last_errors[name] = None
        logger.info("Registered model '%s' in MultiModelPipeline", name)

    def unregister_model(self, name: str) -> None:
        """Remove a model from the pipeline.

        Parameters:
            name (str): Name of the model to unregister.
        """
        if name in self._models:
            del self._models[name]
            self.last_errors.pop(name, None)
            logger.info("Unregistered model '%s' from MultiModelPipeline", name)

    def get_model(self, name: str) -> Union[InferenceEngine, ThreadSafeInferenceWrapper]:
        """Retrieve a registered model engine by name.

        Parameters:
            name (str): Registered model identifier.

        Returns:
            Union[InferenceEngine, ThreadSafeInferenceWrapper]: Underlying engine or wrapper.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered.")
        return self._models[name]

    def list_models(self) -> List[str]:
        """Return a list of all registered model names."""
        return list(self._models.keys())

    def predict_batch(
        self,
        frames: Sequence[FrameData],
        model_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[InferenceResult]]:
        """Run batched inference across registered models, calling each engine's predict_batch once.

        Isolates model failures: if one model raises an exception, logs the failure with the
        model name, records the error string in self.last_errors[name], keeps the other models'
        results, and never allows one model failure to crash the pipeline loop.

        Parameters:
            frames (Sequence[FrameData]): Sequence of FrameData instances to predict on.
            model_names (Optional[Sequence[str]]): Optional subset of model names to execute (default: all).

        Returns:
            Dict[str, List[InferenceResult]]: Mapping of model_name to list of InferenceResults.
        """
        targets = list(model_names) if model_names is not None else list(self._models.keys())
        results: Dict[str, List[InferenceResult]] = {}

        for name in targets:
            if name not in self._models:
                raise KeyError(f"Model '{name}' is not registered.")
            engine = self._models[name]
            try:
                model_results = engine.predict_batch(frames)
                results[name] = model_results
                self.last_errors[name] = None
            except Exception as ex:
                err_msg = f"Inference execution failed for model '{name}': {ex}"
                logger.error(err_msg, exc_info=True)
                self.last_errors[name] = str(ex)
                # Failure isolation: do not let this model failure take down the loop or other models

        return results

    def predict_frame(
        self,
        frame: FrameData,
        model_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, InferenceResult]:
        """Run inference on a single frame across registered models.

        Parameters:
            frame (FrameData): Ingested FrameData container.
            model_names (Optional[Sequence[str]]): Optional subset of model names to execute.

        Returns:
            Dict[str, InferenceResult]: Dict mapping model_name to its structured InferenceResult.
        """
        batch_results = self.predict_batch([frame], model_names=model_names)
        return {name: res_list[0] for name, res_list in batch_results.items() if res_list}

    def predict_combined(
        self,
        frame: FrameData,
        model_names: Optional[Sequence[str]] = None,
    ) -> InferenceResult:
        """Execute across models and aggregate all detections into a single InferenceResult.

        Aggregates end_to_end_ms as the max across executed models, sums per-image
        timing components, and reports device_used="none" if no models ran.

        Parameters:
            frame (FrameData): Ingested FrameData packet.
            model_names (Optional[Sequence[str]]): Optional subset of model names to run.

        Returns:
            InferenceResult: Aggregated InferenceResult with combined DetectionBox instances,
                             summed latency breakdowns, max end_to_end_ms, and combined devices.
        """
        model_results = self.predict_frame(frame, model_names=model_names)

        all_detections: List[DetectionBox] = []
        total_prep = 0.0
        total_inf = 0.0
        total_post = 0.0
        devices_used: List[str] = []

        for name, res in model_results.items():
            all_detections.extend(res.detections)
            total_prep += res.preprocess_ms
            total_inf += res.inference_ms
            total_post += res.postprocess_ms
            if res.device_used not in devices_used:
                devices_used.append(res.device_used)

        max_e2e = max((res.end_to_end_ms for res in model_results.values()), default=0.0)
        device_str = ",".join(devices_used) if devices_used else "none"

        return InferenceResult(
            source_id=frame.source_id,
            frame_id=frame.frame_id,
            timestamp_utc=frame.timestamp_utc,
            timestamp_mono=frame.timestamp_mono,
            detections=tuple(all_detections),
            preprocess_ms=total_prep,
            inference_ms=total_inf,
            postprocess_ms=total_post,
            total_latency_ms=total_prep + total_inf + total_post,
            device_used=device_str,
            image_shape=(frame.height, frame.width),
            end_to_end_ms=max_e2e,
        )
