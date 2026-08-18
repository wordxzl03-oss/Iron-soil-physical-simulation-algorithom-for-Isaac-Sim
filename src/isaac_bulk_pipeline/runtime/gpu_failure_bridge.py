"""Compact CAD FailureZone bridge for the device-authoritative runtime.

The established CAD sweep, intersection and FailureZone equations remain CPU
geometry code.  Only their terrain input is changed: a physical, dynamically
expanded local region is downloaded from :class:`DeviceBulkState`, activation
is calculated there, and only that changed region is scattered back to device
authority.  No full ``TerrainState`` is reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from ..bulk_interaction import (
    DeviceToolMobileContactSupport,
    FailureZone,
    FailureZoneModel,
    ToolMobileContactSupport,
    ToolTerrainIntersection,
    ToolTerrainIntersectionModel,
    WarpExactToolMobileContactGeometry,
)
from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..interaction import SweepResult
from ..terrain import TerrainGrid
from ..tools import ToolDescriptor, ToolState
from .bulk_state_authority import DeviceBulkState, HostBulkStatePatch


@dataclass(frozen=True)
class DeviceFailureZoneResult:
    """Compact FailureZone result and direct resting→mobile transaction data."""

    patch_bbox_yx: tuple[int, int, int, int]
    requested_bbox_yx: tuple[int, int, int, int]
    expansion_count: int
    patch_cell_count: int
    patch_fraction_of_terrain: float
    intersection: ToolTerrainIntersection
    failure_zone: FailureZone
    activated_height_m: np.ndarray
    activated_volume_m3: float
    activation_tool_impulse_on_mobile_terrain_ns: np.ndarray
    activation_mode: str
    forcing_flat_indices: np.ndarray
    tool_mobile_contact: ToolMobileContactSupport | DeviceToolMobileContactSupport
    timings_ms: dict[str, float]


class DeviceFailureZoneBridge:
    """Run the real FailureZone equation on a physically sufficient patch."""

    backend_identity = "GPU_RUNTIME_COMPACT_HOST_FAILUREZONE_DEVICE_CONTACT_BRIDGE"

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": self.backend_identity,
            "failure_surface_execution": "COMPACT_HOST_REFERENCE_UNCHANGED",
            "tool_mobile_contact": (
                {"backend": "DISABLED_NO_DESCRIPTOR"}
                if self.tool_mobile_geometry is None
                else self.tool_mobile_geometry.diagnostics()
            ),
        }

    def __init__(
        self,
        state: DeviceBulkState,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        *,
        descriptor: ToolDescriptor | None = None,
        intersection_model: ToolTerrainIntersectionModel | None = None,
        failure_model: FailureZoneModel | None = None,
    ) -> None:
        if state.grid is not grid:
            raise ValueError("[GpuFailureBridge] state/grid identity mismatch")
        self.state = state
        self.material = material
        self.grid = grid
        self.integrator = integrator
        self.descriptor = descriptor
        self.intersection_model = intersection_model or ToolTerrainIntersectionModel()
        self.failure_model = failure_model or FailureZoneModel()
        self.tool_mobile_geometry = (
            None
            if descriptor is None or descriptor.bucket_geometry is None
            else WarpExactToolMobileContactGeometry(state=state, descriptor=descriptor)
        )
        # Acceptance-only observer.  It is deliberately unset in production
        # and observes the two real FailureSurface/R->M commit boundaries
        # without changing either geometry or transfer semantics.
        self.audit_state_observer: Callable[[str], None] | None = None

    def execute(self, sweep: SweepResult, tool_state: ToolState) -> DeviceFailureZoneResult:
        requested = self._physical_request_bbox(sweep)
        bbox = requested
        expansions = 0
        query_ms = geometry_ms = intersection_ms = failure_surface_ms = 0.0
        while True:
            start = perf_counter()
            patch = self.state.download_region(bbox, source="failure_zone_geometry")
            query_ms += (perf_counter() - start) * 1_000.0
            local_grid = self._local_grid(bbox)
            local_integrator = TerrainVolumeIntegrator.from_grid(local_grid)
            local_sweep = self._local_sweep(sweep, bbox)
            start = perf_counter()
            intersection_start = perf_counter()
            intersection = self.intersection_model.compute(
                patch.H_resting_m + patch.H_mobile_m,
                local_sweep, tool_state, local_grid, local_integrator
            )
            intersection_ms += (perf_counter() - intersection_start) * 1_000.0
            failure_start = perf_counter()
            failure = self.failure_model.compute(
                intersection,
                patch.H_resting_m,
                self.material,
                local_grid,
                local_integrator,
                fallback_approach_direction_xy=tool_state.pose_terrain[:2, 1],
                H_free_m=patch.H_resting_m + patch.H_mobile_m,
            )
            failure_surface_ms += (perf_counter() - failure_start) * 1_000.0
            geometry_ms += (perf_counter() - start) * 1_000.0
            if not self._touches_patch_boundary(failure.active_thickness_m) or self._at_global_boundary(bbox):
                break
            expanded = self._expand_bbox(bbox)
            if expanded == bbox:
                break
            bbox = expanded
            expansions += 1

        available_static = patch.h_resting_derived_m
        activated = np.minimum(failure.active_thickness_m, available_static)
        mode = "FEE_FAILURE_ZONE"
        if failure.applicability_status in {"OUTSIDE_FEE_DOMAIN", "PARTIAL_OUTSIDE_FEE_DOMAIN"}:
            activated = np.maximum(
                activated, np.minimum(intersection.penetration_depth_m, available_static)
            )
            mode = "GEOMETRIC_SWEEP_ONLY_OUTSIDE_FEE_FORCE_DOMAIN"
        elif not np.any(activated > 0.0):
            mode = "NONE"
        if self.audit_state_observer is not None:
            self.audit_state_observer("AFTER_FAILURE_SURFACE_GEOMETRY")
        # Failure activation transfers mass only.  Geometry and donor limiting
        # are committed by DeviceBulkState's single authoritative entrainment
        # primitive; q is not modified by state activation.
        activation_total_start = perf_counter()
        start = perf_counter()
        self.state.capture_surface_for_dirty_tracking()
        rows_all, cols_all = np.indices(activated.shape, dtype=np.int32)
        flat_all = ((rows_all + bbox[0]) * self.grid.nx + (cols_all + bbox[2])).ravel()
        volume = self.state.entrain_host_indices(
            flat_all, activated.ravel(), reason="failure_zone_activation"
        )
        if self.audit_state_observer is not None:
            self.audit_state_observer("AFTER_R2M_ACTIVATION")
        activation_commit_ms = (perf_counter() - start) * 1_000.0
        # The physical failure patch is only a spatial restriction.  Wet
        # candidate generation and exact CAD proof now happen against the
        # authoritative device fields; material_mask is an output, never proof.
        contact_support = (
            ToolMobileContactSupport.empty()
            if self.tool_mobile_geometry is None
            else self.tool_mobile_geometry.compute(
                bbox_yx=bbox, tool_state=tool_state
            )
        )
        contact_diagnostics = dict(contact_support.performance_diagnostics)
        forcing = np.empty(0, dtype=np.int32)
        start = perf_counter()
        self.state.collect_surface_dirty_tiles()
        dirty_surface_sync_ms = (perf_counter() - start) * 1_000.0
        activation_ms = (perf_counter() - activation_total_start) * 1_000.0
        contact_timing_and_counts = {
            "tool_mobile_candidate_build_ms": float(
                contact_diagnostics.get("candidate_build_ms", 0.0)
            ),
            "tool_mobile_broadphase_ms": float(
                contact_diagnostics.get("broadphase_ms", 0.0)
            ),
            "tool_mobile_narrowphase_ms": float(
                contact_diagnostics.get("narrowphase_ms", 0.0)
            ),
            "tool_mobile_closest_point_ms": float(
                contact_diagnostics.get("closest_point_ms", 0.0)
            ),
            "tool_mobile_geometry_prepare_ms": float(
                contact_diagnostics.get(
                    "cad_transform_ms", contact_diagnostics.get("geometry_prepare_ms", 0.0)
                )
            ),
            "tool_mobile_gpu_geometry_ms": float(
                contact_diagnostics.get("gpu_geometry_ms", 0.0)
            ),
            "tool_mobile_contact_support_compaction_ms": float(
                contact_diagnostics.get("contact_support_compaction_ms", 0.0)
            ),
            "tool_mobile_geometry_total_ms": float(
                contact_diagnostics.get("tool_mobile_total_ms", 0.0)
            ),
            "tool_mobile_host_device_sync_ms": float(
                contact_diagnostics.get("scalar_d2h_ms", 0.0)
            ),
            "failure_activation_dirty_surface_sync_ms": float(dirty_surface_sync_ms),
            "tool_mobile_contact_h2d_ms": float(
                contact_diagnostics.get("contact_h2d_ms", 0.0)
            ),
            "tool_mobile_contact_h2d_bytes": float(
                contact_diagnostics.get("contact_h2d_bytes", 0)
            ),
            "mobile_candidate_count": float(
                contact_diagnostics.get("mobile_candidate_count", 0)
            ),
            "geometry_candidate_count": float(
                contact_diagnostics.get("geometry_candidate_count", 0)
            ),
            "cad_triangle_count": float(
                contact_diagnostics.get("cad_triangle_count", 0)
            ),
            "broadphase_pair_count": float(
                contact_diagnostics.get("broadphase_pair_count", 0)
            ),
            "triangle_aabb_test_count": float(
                contact_diagnostics.get("triangle_aabb_test_count", 0)
            ),
            "ray_triangle_test_count": float(
                contact_diagnostics.get("ray_triangle_test_count", 0)
            ),
            "containment_test_count": float(
                contact_diagnostics.get("containment_test_count", 0)
            ),
            "exact_test_count": float(
                contact_diagnostics.get("exact_test_count", 0)
            ),
            "closest_point_query_count": float(
                contact_diagnostics.get("closest_point_query_count", 0)
            ),
            "accepted_contact_count": float(contact_support.cell_count),
        }
        return DeviceFailureZoneResult(
            patch_bbox_yx=bbox,
            requested_bbox_yx=requested,
            expansion_count=expansions,
            patch_cell_count=int(np.prod(activated.shape)),
            patch_fraction_of_terrain=float(activated.size / self.state.size),
            intersection=intersection,
            failure_zone=failure,
            activated_height_m=np.ascontiguousarray(activated),
            activated_volume_m3=float(volume),
            activation_tool_impulse_on_mobile_terrain_ns=np.zeros(3, dtype=np.float64),
            activation_mode=mode,
            forcing_flat_indices=forcing,
            tool_mobile_contact=contact_support,
            timings_ms={
                "terrain_patch_query": query_ms,
                "bucket_geometry_and_failure_zone": geometry_ms,
                "tool_terrain_intersection_ms": intersection_ms,
                "failure_surface_ms": failure_surface_ms,
                "failure_activation_commit_ms": activation_commit_ms,
                "activation_scatter": activation_ms,
                **contact_timing_and_counts,
            },
        )

    def clear_tool_forcing(
        self,
        contact: ToolMobileContactSupport | DeviceToolMobileContactSupport,
    ) -> None:
        if getattr(contact, "device_resident", False):
            assert self.tool_mobile_geometry is not None
            self.tool_mobile_geometry.clear()
            return
        self.state.apply_host_indices(
            "material_mask", np.asarray(contact.flat_indices, dtype=np.int32), 0,
            reason="failure_zone_tool_forcing_complete",
        )

    def _physical_request_bbox(self, sweep: SweepResult) -> tuple[int, int, int, int]:
        row0, col0, row1, col1 = sweep.affected_bbox_grid
        if row1 <= row0 or col1 <= col0:
            # No CAD sweep: a one-cell compact query is sufficient and cannot
            # accidentally become a global terrain request.
            return (0, 1, 0, 1)
        runout_rows = int(np.ceil(self.failure_model.config.maximum_wedge_length_m / self.grid.dy))
        runout_cols = int(np.ceil(self.failure_model.config.maximum_wedge_length_m / self.grid.dx))
        return (
            max(0, row0 - runout_rows), min(self.grid.ny, row1 + runout_rows),
            max(0, col0 - runout_cols), min(self.grid.nx, col1 + runout_cols),
        )

    def _expand_bbox(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        # Expansion distance derives solely from the implemented physical
        # maximum wedge length, never a fixed performance halo.
        row_pad = int(np.ceil(self.failure_model.config.maximum_wedge_length_m / self.grid.dy))
        col_pad = int(np.ceil(self.failure_model.config.maximum_wedge_length_m / self.grid.dx))
        return (
            max(0, bbox[0] - row_pad), min(self.grid.ny, bbox[1] + row_pad),
            max(0, bbox[2] - col_pad), min(self.grid.nx, bbox[3] + col_pad),
        )

    def _at_global_boundary(self, bbox: tuple[int, int, int, int]) -> bool:
        return bbox[0] == 0 or bbox[1] == self.grid.ny or bbox[2] == 0 or bbox[3] == self.grid.nx

    @staticmethod
    def _touches_patch_boundary(field: np.ndarray) -> bool:
        active = np.asarray(field) > 1.0e-12
        return bool(active.any() and (active[0].any() or active[-1].any() or active[:, 0].any() or active[:, -1].any()))

    def _local_grid(self, bbox: tuple[int, int, int, int]) -> TerrainGrid:
        row0, row1, col0, col1 = bbox
        return TerrainGrid(
            col1 - col0, row1 - row0, self.grid.dx, self.grid.dy,
            self.grid.origin_x + col0 * self.grid.dx,
            self.grid.origin_y + row0 * self.grid.dy,
            "/World/Terrain/CompactFailurePatch",
            terrain_to_world_matrix=self.grid.terrain_to_world_matrix,
        )

    @staticmethod
    def _local_sweep(sweep: SweepResult, bbox: tuple[int, int, int, int]) -> SweepResult:
        row0, row1, col0, col1 = bbox
        mask = sweep.affected_mask[row0:row1, col0:col1]
        cut = sweep.cut_surface[row0:row1, col0:col1]
        where = np.argwhere(mask)
        local_bbox = (0, 0, 0, 0) if not len(where) else (
            int(where[:, 0].min()), int(where[:, 1].min()),
            int(where[:, 0].max()) + 1, int(where[:, 1].max()) + 1,
        )
        return SweepResult(local_bbox, mask, cut, sweep.sampled_poses, sweep.diagnostics)
