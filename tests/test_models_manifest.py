"""Unit tests for models.yaml manifest parsing and schema validation."""

import tempfile
from pathlib import Path
import pytest
import yaml

from pc_infernoguard.inference.models_manifest import load_models_manifest


def test_load_models_manifest_valid():
    """Verify loading a valid models.yaml manifest with multiple model entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        w1 = tmp_path / "model1.pt"
        w2 = tmp_path / "model2.pt"
        w1.write_bytes(b"dummy_weights_1")
        w2.write_bytes(b"dummy_weights_2")

        manifest_content = {
            "models": [
                {
                    "name": "fire_primary",
                    "path": str(w1),
                    "device": "cpu",
                    "conf_threshold": 0.30,
                    "iou_threshold": 0.50,
                    "imgsz": 640,
                    "half_precision": False,
                    "max_det": 100,
                    "warmup_iterations": 2,
                    "expected_classes": {0: "fire", 1: "smoke", 2: "spark"},
                    "strict_classes": True,
                },
                {
                    "name": "spark_verifier",
                    "path": str(w2),
                    "conf_threshold": 0.15,
                },
            ]
        }
        manifest_file = tmp_path / "models.yaml"
        manifest_file.write_text(yaml.safe_dump(manifest_content), encoding="utf-8")

        configs = load_models_manifest(manifest_file)
        assert len(configs) == 2

        # Check first config
        c1 = configs[0]
        assert c1.model_name == "fire_primary"
        assert c1.model_path == str(w1)
        assert c1.device == "cpu"
        assert c1.conf_threshold == 0.30
        assert c1.iou_threshold == 0.50
        assert c1.imgsz == 640
        assert c1.max_det == 100
        assert c1.warmup_iterations == 2
        assert c1.expected_classes == {0: "fire", 1: "smoke", 2: "spark"}
        assert c1.strict_classes is True

        # Check second config defaults
        c2 = configs[1]
        assert c2.model_name == "spark_verifier"
        assert c2.model_path == str(w2)
        assert c2.conf_threshold == 0.15
        assert c2.iou_threshold == 0.45  # default
        assert c2.imgsz == 640  # default


def test_load_models_manifest_nonexistent_manifest_raises():
    """Verify missing manifest file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_models_manifest("non_existent_models_manifest.yaml")


def test_load_models_manifest_nonexistent_weights_raises():
    """Verify non-existent weights file raises FileNotFoundError with entry context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "models.yaml"
        manifest_content = {
            "models": [
                {
                    "name": "ghost_model",
                    "path": "does_not_exist_weights.pt",
                }
            ]
        }
        manifest_file.write_text(yaml.safe_dump(manifest_content), encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="weights file 'does_not_exist_weights.pt' does not exist"):
            load_models_manifest(manifest_file)


def test_load_models_manifest_duplicate_model_names_raises():
    """Verify duplicate model names raise ValueError naming the duplicate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        w = tmp_path / "model.pt"
        w.write_bytes(b"dummy")

        manifest_file = tmp_path / "models.yaml"
        manifest_content = {
            "models": [
                {"name": "duplicate_name", "path": str(w)},
                {"name": "duplicate_name", "path": str(w)},
            ]
        }
        manifest_file.write_text(yaml.safe_dump(manifest_content), encoding="utf-8")

        with pytest.raises(ValueError, match="duplicate model name 'duplicate_name'"):
            load_models_manifest(manifest_file)


def test_load_models_manifest_missing_name_raises():
    """Verify entry missing 'name' raises ValueError naming the offending entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        w = tmp_path / "model.pt"
        w.write_bytes(b"dummy")

        manifest_file = tmp_path / "models.yaml"
        manifest_content = {
            "models": [
                {"path": str(w)}  # missing name
            ]
        }
        manifest_file.write_text(yaml.safe_dump(manifest_content), encoding="utf-8")

        with pytest.raises(ValueError, match="missing or empty required field 'name'"):
            load_models_manifest(manifest_file)


def test_load_models_manifest_empty_models_list_raises():
    """Verify manifest with empty 'models' list raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "models.yaml"
        manifest_file.write_text("models: []", encoding="utf-8")

        with pytest.raises(ValueError, match="'models' list is empty"):
            load_models_manifest(manifest_file)
