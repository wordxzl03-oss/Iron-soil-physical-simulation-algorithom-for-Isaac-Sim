"""Computability primitives shared by headless and interactive V2 runtimes."""

from .active_domain import (
    ActiveDomainManager,
    ActiveDomainSnapshot,
    ActiveReason,
    GridWindow,
)
from .profiler import PerformanceProfiler, PerformanceSummary, TransferCounters
from .terrain_hierarchy_acceptance import (
    TerrainAcceptancePath,
    TerrainAcceptanceStatus,
    TerrainHierarchyAcceptanceReport,
    TerrainPathMetrics,
)
from .warp_backend import (
    WarpBackendStatus,
    WarpBackendUnavailable,
    WarpRuntime,
    WarpTransferTelemetry,
    probe_warp,
)

__all__ = [
    "ActiveDomainManager",
    "ActiveDomainSnapshot",
    "ActiveReason",
    "GridWindow",
    "PerformanceProfiler",
    "PerformanceSummary",
    "TransferCounters",
    "TerrainAcceptancePath",
    "TerrainAcceptanceStatus",
    "TerrainHierarchyAcceptanceReport",
    "TerrainPathMetrics",
    "WarpBackendStatus",
    "WarpBackendUnavailable",
    "WarpRuntime",
    "WarpTransferTelemetry",
    "probe_warp",
]
