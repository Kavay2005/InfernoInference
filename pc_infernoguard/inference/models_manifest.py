"""Schema validation and loader for models.yaml manifest into InferenceConfig instances."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml

from pc_infernoguard.inference.inference_types import InferenceConfig


def load_models_manifest(path: Union[str, Path]) -> List[InferenceConfig]:
    """Load and validate specialized model configurations from a models.yaml manifest.

    Adheres to AGENTS.md Section 8 and ADR 0003:
    - Model identities, paths, device selection, thresholds, and class mappings
      are explicitly specified in configuration.
    - Validates that model names are unique.
    - Verifies that model weight files physically exist on disk before pipeline initialization.
    - Emits clear, descriptive error messages identifying any offending entries.

    Parameters:
        path (Union[str, Path]): Path to the models.yaml file.

    Returns:
        List[InferenceConfig]: Validated list of InferenceConfig instances.

    Raises:
        FileNotFoundError: If the manifest file or any referenced model weights file does not exist.
        ValueError: If the schema is invalid, required fields are missing, or duplicate names occur.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Models manifest file not found: {path}")

    try:
        content = manifest_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except Exception as ex:
        raise ValueError(f"Failed to parse YAML manifest '{path}': {ex}") from ex

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid manifest structure in '{path}': expected a dictionary mapping at root, got {type(data).__name__}"
        )

    if "models" not in data:
        raise ValueError(
            f"Invalid manifest in '{path}': missing required top-level 'models' list"
        )

    models_list = data["models"]
    if not isinstance(models_list, list):
        raise ValueError(
            f"Invalid manifest in '{path}': 'models' must be a list, got {type(models_list).__name__}"
        )

    if len(models_list) == 0:
        raise ValueError(
            f"Invalid manifest in '{path}': 'models' list is empty; at least one model configuration is required"
        )

    configs: List[InferenceConfig] = []
    seen_names: Set[str] = set()

    for idx, entry in enumerate(models_list):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Offending entry {idx} in models manifest '{path}': expected a dictionary, got {type(entry).__name__}"
            )

        # 1. Validate required fields: name and path
        name = entry.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Offending entry {idx} in models manifest '{path}': missing or empty required field 'name'"
            )
        name = name.strip()

        if name in seen_names:
            raise ValueError(
                f"Offending entry {idx} in models manifest '{path}': duplicate model name '{name}' already defined"
            )
        seen_names.add(name)

        raw_path = entry.get("path") or entry.get("model_path")
        if not raw_path or not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"Offending entry {idx} (model '{name}') in models manifest '{path}': missing or empty required field 'path'"
            )
        raw_path = raw_path.strip()

        # 2. Verify weights file exists
        weights_path = Path(raw_path)
        if not weights_path.is_file():
            # Check relative to manifest file directory if not found relative to CWD
            manifest_dir = manifest_path.parent
            candidate_path = manifest_dir / weights_path
            if candidate_path.is_file():
                weights_path = candidate_path
            else:
                raise FileNotFoundError(
                    f"Offending entry {idx} (model '{name}') in models manifest '{path}': "
                    f"weights file '{raw_path}' does not exist (checked '{weights_path}' and '{candidate_path}')"
                )

        # 3. Optional fields and type conversions
        device = str(entry.get("device", "auto")).strip()
        conf = float(entry.get("conf_threshold", 0.25))
        iou = float(entry.get("iou_threshold", 0.45))
        imgsz = int(entry.get("imgsz", 640))
        half = bool(entry.get("half_precision", False))
        max_det = int(entry.get("max_det", 300))
        warmup = int(entry.get("warmup_iterations", 1))
        fallback_cpu = bool(entry.get("fallback_to_cpu", True))
        strict_cls = bool(entry.get("strict_classes", False))

        expected_classes: Optional[Dict[int, str]] = None
        raw_classes = entry.get("expected_classes")
        if raw_classes is not None:
            if not isinstance(raw_classes, dict):
                raise ValueError(
                    f"Offending entry {idx} (model '{name}') in models manifest '{path}': "
                    f"'expected_classes' must be a dictionary, got {type(raw_classes).__name__}"
                )
            expected_classes = {int(k): str(v) for k, v in raw_classes.items()}

        config = InferenceConfig(
            model_path=str(weights_path),
            model_name=name,
            device=device,
            conf_threshold=conf,
            iou_threshold=iou,
            imgsz=imgsz,
            half_precision=half,
            max_det=max_det,
            warmup_iterations=warmup,
            fallback_to_cpu=fallback_cpu,
            expected_classes=expected_classes,
            strict_classes=strict_cls,
        )
        configs.append(config)

    return configs
