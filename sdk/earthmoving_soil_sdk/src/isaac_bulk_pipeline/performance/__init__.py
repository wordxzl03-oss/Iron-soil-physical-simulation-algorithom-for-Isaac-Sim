"""Computability primitives retained by the production soil core."""

from .active_domain import ActiveDomainManager, ActiveDomainSnapshot, ActiveReason, GridWindow
from .profiler import PerformanceProfiler, PerformanceSummary, TransferCounters
from .warp_backend import WarpBackendStatus, WarpBackendUnavailable, WarpRuntime, WarpTransferTelemetry, probe_warp

__all__ = [
    "ActiveDomainManager", "ActiveDomainSnapshot", "ActiveReason", "GridWindow",
    "PerformanceProfiler", "PerformanceSummary", "TransferCounters",
    "WarpBackendStatus", "WarpBackendUnavailable", "WarpRuntime",
    "WarpTransferTelemetry", "probe_warp",
]
