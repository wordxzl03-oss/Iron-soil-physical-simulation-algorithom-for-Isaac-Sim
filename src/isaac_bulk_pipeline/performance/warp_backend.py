"""Optional NVIDIA Warp backend discovery and transfer accounting.

This module deliberately does not add Isaac Sim extension paths to ``sys.path``.
Isaac/Kit owns extension loading; standalone benchmarks must expose the shipped
``omni.warp.core`` Python directory through ``PYTHONPATH``.  A missing Warp
runtime or CUDA device is an explicit unavailable backend, never a CPU pass
reported as GPU work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from time import perf_counter
from typing import Any

import numpy as np


class WarpBackendUnavailable(RuntimeError):
    """Raised when a requested Warp execution device is unavailable."""


@dataclass(frozen=True)
class WarpBackendStatus:
    status: str
    device: str
    warp_version: str | None
    cuda_available: bool
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "AVAILABLE"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class WarpTransferTelemetry:
    """Actual explicit host/device traffic performed by one backend instance."""

    h2d_bytes: int = 0
    d2h_bytes: int = 0
    h2d_transfer_count: int = 0
    d2h_transfer_count: int = 0
    synchronization_count: int = 0
    kernel_launch_count: int = 0
    kernel_wall_time_ms: float = 0.0

    def record_h2d(self, value: np.ndarray) -> None:
        self.h2d_bytes += int(np.asarray(value).nbytes)
        self.h2d_transfer_count += 1

    def record_d2h(self, value: np.ndarray) -> None:
        self.d2h_bytes += int(np.asarray(value).nbytes)
        self.d2h_transfer_count += 1

    def record_sync(self) -> None:
        self.synchronization_count += 1

    def record_kernel(self, wall_time_ms: float) -> None:
        self.kernel_launch_count += 1
        self.kernel_wall_time_ms += float(wall_time_ms)

    def reset(self) -> None:
        for name in (
            "h2d_bytes",
            "d2h_bytes",
            "h2d_transfer_count",
            "d2h_transfer_count",
            "synchronization_count",
            "kernel_launch_count",
        ):
            setattr(self, name, 0)
        self.kernel_wall_time_ms = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_warp() -> Any | None:
    """Return the imported Warp module, or ``None`` without masking errors."""

    try:
        return import_module("warp")
    except (ImportError, ModuleNotFoundError):
        return None


def probe_warp(device: str = "cuda:0") -> WarpBackendStatus:
    wp = load_warp()
    if wp is None:
        return WarpBackendStatus(
            status="UNAVAILABLE",
            device=device,
            warp_version=None,
            cuda_available=False,
            reason="WARP_PYTHON_MODULE_NOT_IMPORTABLE",
        )
    version = str(getattr(wp, "__version__", "UNKNOWN"))
    try:
        wp.init()
        cuda_available = bool(wp.is_cuda_available())
        available = bool(wp.is_device_available(device))
    except Exception as exc:  # backend/driver failures must remain diagnostic
        return WarpBackendStatus(
            status="UNAVAILABLE",
            device=device,
            warp_version=version,
            cuda_available=False,
            reason=f"WARP_INITIALIZATION_FAILED:{type(exc).__name__}:{exc}",
        )
    if not available:
        return WarpBackendStatus(
            status="UNAVAILABLE",
            device=device,
            warp_version=version,
            cuda_available=cuda_available,
            reason=f"WARP_DEVICE_UNAVAILABLE:{device}",
        )
    return WarpBackendStatus(
        status="AVAILABLE",
        device=device,
        warp_version=version,
        cuda_available=cuda_available,
    )


class WarpRuntime:
    """Small ownership layer for resident arrays and measured Warp launches."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.status = probe_warp(device)
        if not self.status.available:
            raise WarpBackendUnavailable(self.status.reason or "WARP_UNAVAILABLE")
        self.wp = load_warp()
        assert self.wp is not None
        self.device = device
        self.telemetry = WarpTransferTelemetry()
        self.arrays: dict[str, Any] = {}

    def upload(self, name: str, value: np.ndarray, *, dtype: Any = None) -> Any:
        host = np.ascontiguousarray(value)
        array = self.wp.from_numpy(host, dtype=dtype, device=self.device)
        self.arrays[name] = array
        self.telemetry.record_h2d(host)
        return array

    def zeros(self, name: str, shape: int | tuple[int, ...], *, dtype: Any) -> Any:
        array = self.wp.zeros(shape, dtype=dtype, device=self.device)
        self.arrays[name] = array
        return array

    def empty(self, name: str, shape: int | tuple[int, ...], *, dtype: Any) -> Any:
        array = self.wp.empty(shape, dtype=dtype, device=self.device)
        self.arrays[name] = array
        return array

    def download(self, name: str) -> np.ndarray:
        host = np.asarray(self.arrays[name].numpy())
        self.telemetry.record_d2h(host)
        return host

    def release(self, name: str) -> None:
        """Release this runtime's reference to a transient resident array.

        Warp's allocator may retain the allocation in its device pool for
        reuse, but removing the final Python/runtime reference makes the
        buffer reusable instead of growing the live working set forever.
        Callers must synchronize the work that consumes the array first.
        """

        self.arrays.pop(name, None)

    def synchronize(self) -> None:
        self.wp.synchronize_device(self.device)
        self.telemetry.record_sync()

    def launch(self, kernel: Any, *, dim: int, inputs: list[Any]) -> None:
        start = perf_counter()
        self.wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)
        self.telemetry.record_kernel((perf_counter() - start) * 1_000.0)

    def diagnostics(self, backend_name: str) -> dict[str, object]:
        return {
            "backend": backend_name,
            "execution_status": "GPU_OPTIMIZED",
            "warp": self.status.to_dict(),
            **self.telemetry.to_dict(),
            "resident_array_names": sorted(self.arrays),
        }
