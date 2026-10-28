"""Device management and hardware capability resolution for PyTorch/CUDA."""

import logging
from typing import List
import torch

from pc_infernoguard.inference.inference_types import DeviceInfo

logger = logging.getLogger("pc_infernoguard.device")


class DeviceManager:
    """Manages execution device detection, capability inspection, and resolution.

    Adheres to AGENTS.md Section 5:
    - Explicit selectable CPU option.
    - Inspects GPU availability and runtime compatibility rather than assuming CUDA is present.
    - Provides configurable fallback when requested GPU hardware is absent.

    Note:
        If the process will fork or use multiprocessing, set the environment variable
        PYTORCH_NVML_BASED_CUDA_CHECK=1 before importing torch, because
        torch.cuda.is_available() initializes CUDA by default.
    """

    @staticmethod
    def is_cuda_available() -> bool:
        """Return True if PyTorch has CUDA support and a compatible GPU is available.

        Note:
            If the process will fork or use multiprocessing, set PYTORCH_NVML_BASED_CUDA_CHECK=1
            before importing torch, because torch.cuda.is_available() initializes CUDA.
        """
        return bool(torch.cuda.is_available())

    @staticmethod
    def get_cuda_device_count() -> int:
        """Return the number of available CUDA devices."""
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @classmethod
    def resolve_device(
        cls,
        target_device: str = "auto",
        fallback_to_cpu: bool = True,
    ) -> str:
        """Resolve a device specification string into a validated execution device.

        Args:
            target_device: Requested device string ('auto', 'cpu', 'cuda', 'cuda:0', '0').
            fallback_to_cpu: If True and CUDA is unavailable, log warning and fallback to 'cpu'.
                             If False, raise RuntimeError.

        Returns:
            Normalized device string (e.g., 'cpu' or 'cuda:0').

        Raises:
            RuntimeError: If CUDA is requested, unavailable, and fallback_to_cpu is False.
            ValueError: If target_device specification format is unrecognized.
        """
        target = str(target_device).strip().lower()

        if target == "auto":
            if cls.is_cuda_available():
                resolved = "cuda:0"
                logger.info("Auto device resolution: CUDA available, selected '%s'", resolved)
                return resolved
            logger.info("Auto device resolution: CUDA not available, selected 'cpu'")
            return "cpu"

        if target == "cpu":
            return "cpu"

        # Check for CUDA specifications: "cuda", "cuda:0", "cuda:1", or numeric string "0"
        is_cuda_spec = target.startswith("cuda") or target.isdigit()
        if is_cuda_spec:
            # Normalize target to e.g. "cuda:0"
            if target.isdigit():
                normalized = f"cuda:{target}"
                dev_idx = int(target)
            elif target == "cuda":
                normalized = "cuda:0"
                dev_idx = 0
            else:
                normalized = target
                try:
                    dev_idx = int(target.split(":")[1])
                except (IndexError, ValueError):
                    dev_idx = 0

            if cls.is_cuda_available():
                count = cls.get_cuda_device_count()
                if dev_idx >= count:
                    msg = (
                        f"Requested CUDA device index {dev_idx} exceeds available count ({count})."
                    )
                    if fallback_to_cpu:
                        logger.warning("%s Falling back to CPU.", msg)
                        return "cpu"
                    raise RuntimeError(msg)
                return normalized
            else:
                msg = (
                    f"CUDA execution was requested ('{target_device}'), but "
                    f"torch.cuda.is_available() is False on this host (PyTorch: {torch.__version__})."
                )
                if fallback_to_cpu:
                    logger.warning("%s Falling back to CPU execution.", msg)
                    return "cpu"
                raise RuntimeError(msg)

        raise ValueError(
            f"Unrecognized device specification: '{target_device}'. "
            f"Expected 'auto', 'cpu', 'cuda', or 'cuda:<idx>'."
        )

    @classmethod
    def get_device_info(cls, device_str: str) -> DeviceInfo:
        """Inspect and return comprehensive metadata for the specified device."""
        norm_dev = cls.resolve_device(device_str, fallback_to_cpu=True)
        is_cuda = norm_dev.startswith("cuda")

        if is_cuda and cls.is_cuda_available():
            try:
                idx = int(norm_dev.split(":")[1]) if ":" in norm_dev else 0
                name = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                mem_mb = props.total_memory / (1024.0 * 1024.0)
                cuda_ver = getattr(torch.version, "cuda", None)
            except Exception as ex:
                logger.warning("Failed querying CUDA properties for %s: %s", norm_dev, ex)
                name = "CUDA Device"
                mem_mb = None
                cuda_ver = None

            return DeviceInfo(
                device_str=norm_dev,
                device_type="cuda",
                is_cuda=True,
                device_name=name,
                total_memory_mb=mem_mb,
                cuda_version=cuda_ver,
                torch_version=torch.__version__,
            )

        return DeviceInfo(
            device_str="cpu",
            device_type="cpu",
            is_cuda=False,
            device_name="CPU",
            total_memory_mb=None,
            cuda_version=None,
            torch_version=torch.__version__,
        )

    @classmethod
    def list_available_devices(cls) -> List[DeviceInfo]:
        """Enumerate all available execution devices on the current host."""
        devices = [cls.get_device_info("cpu")]
        if cls.is_cuda_available():
            for idx in range(cls.get_cuda_device_count()):
                devices.append(cls.get_device_info(f"cuda:{idx}"))
        return devices
