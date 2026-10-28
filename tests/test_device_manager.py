"""Unit tests for DeviceManager device resolution, inspection, and fallback."""

import pytest
from unittest.mock import patch

from pc_infernoguard.inference.device_manager import DeviceManager


def test_cpu_device_resolution():
    assert DeviceManager.resolve_device("cpu") == "cpu"
    assert DeviceManager.resolve_device("CPU") == "cpu"


def test_auto_device_resolution_on_current_host():
    # On the current host without CUDA, "auto" must resolve to "cpu"
    is_cuda = DeviceManager.is_cuda_available()
    expected = "cuda:0" if is_cuda else "cpu"
    assert DeviceManager.resolve_device("auto") == expected


def test_cuda_fallback_behavior_when_cuda_unavailable():
    with patch.object(DeviceManager, "is_cuda_available", return_value=False):
        # 1. With fallback_to_cpu=True (default), should cleanly return "cpu"
        dev = DeviceManager.resolve_device("cuda", fallback_to_cpu=True)
        assert dev == "cpu"

        dev_idx = DeviceManager.resolve_device("cuda:0", fallback_to_cpu=True)
        assert dev_idx == "cpu"

        # 2. With fallback_to_cpu=False, must raise RuntimeError per AGENTS.md
        with pytest.raises(RuntimeError) as exc_info:
            DeviceManager.resolve_device("cuda:0", fallback_to_cpu=False)
        assert "torch.cuda.is_available() is False" in str(exc_info.value)


def test_cuda_resolution_when_cuda_available():
    with patch.object(DeviceManager, "is_cuda_available", return_value=True), \
         patch.object(DeviceManager, "get_cuda_device_count", return_value=2):
        assert DeviceManager.resolve_device("cuda") == "cuda:0"
        assert DeviceManager.resolve_device("cuda:0") == "cuda:0"
        assert DeviceManager.resolve_device("cuda:1") == "cuda:1"
        assert DeviceManager.resolve_device("auto") == "cuda:0"

        # Index out of bounds with fallback
        dev_fallback = DeviceManager.resolve_device("cuda:5", fallback_to_cpu=True)
        assert dev_fallback == "cpu"

        # Index out of bounds with strict fail
        with pytest.raises(RuntimeError):
            DeviceManager.resolve_device("cuda:5", fallback_to_cpu=False)


def test_invalid_device_name_raises():
    with pytest.raises(ValueError):
        DeviceManager.resolve_device("tpu:0")

    with pytest.raises(ValueError):
        DeviceManager.resolve_device("metal")


def test_get_device_info_cpu():
    info = DeviceManager.get_device_info("cpu")
    assert info.device_str == "cpu"
    assert info.device_type == "cpu"
    assert info.is_cuda is False
    assert info.device_name == "CPU"
    assert info.total_memory_mb is None
    assert len(info.torch_version) > 0


def test_list_available_devices_includes_cpu():
    devices = DeviceManager.list_available_devices()
    assert len(devices) >= 1
    cpu_dev = next((d for d in devices if d.device_str == "cpu"), None)
    assert cpu_dev is not None
