from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import (
    BucketIntakeModel,
    ToolTerrainIntersectionModel,
)
from isaac_bulk_pipeline.bulk_state import PayloadState, TerrainVolumeIntegrator
from isaac_bulk_pipeline.interaction import SweepResult
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import (
    BucketGeometryDescriptor,
    GeometryQuality,
    GeometrySource,
    ToolDescriptor,
    ToolDescriptorLoader,
    ToolKinematicsAdapter,
    ToolState,
)


def _rotation_x(angle: float) -> np.ndarray:
    result = np.eye(4)
    cosine, sine = np.cos(angle), np.sin(angle)
    result[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
    return result


def _rotation_z(angle: float) -> np.ndarray:
    result = np.eye(4)
    cosine, sine = np.cos(angle), np.sin(angle)
    result[:2, :2] = [[cosine, -sine], [sine, cosine]]
    return result


def _descriptor(*, width: float = 1.6, depth: float = 2.0, height: float = 1.0):
    half = 0.5 * width
    cutting = np.asarray([[-half, 0.0, 0.0], [half, 0.0, 0.0]])
    bottom = np.asarray([[0.0, -depth, 0.0], [0.0, 0.0, 0.0]])
    interior = np.asarray(
        [
            [0.0, -depth, 0.0],
            [0.0, -depth, height],
            [0.0, 0.0, height],
            [0.0, 0.0, 0.0],
        ]
    )
    geometry = BucketGeometryDescriptor.from_extruded_profile(
        cutting_edge_local=cutting,
        bottom_profile_local=bottom,
        interior_profile_local=interior,
        top_edge_local=np.asarray([[-half, 0.0, height], [half, 0.0, height]]),
        rated_capacity_m3=None,
        geometry_source=GeometrySource.EXPLICIT_PROFILE,
        geometry_quality=GeometryQuality.REDUCED_ORDER,
    )
    descriptor = ToolDescriptor(
        tool_type="bucket",
        tool_frame_prim="/World/Bucket/ToolFrame",
        cutting_edge_local=cutting,
        bottom_profile_local=bottom,
        left_boundary_local=geometry.left_side_wall_local,
        right_boundary_local=geometry.right_side_wall_local,
        nominal_width_m=width,
        interior_profile_local=interior,
        nominal_capacity_m3=None,
        proxy_level="L1",
        actual_proxy_type="ExtrudedProfileBucket_L1",
        bucket_geometry=geometry,
    )
    return descriptor


def _grid(spacing: float, *, extent: float = 2.4) -> tuple[TerrainGrid, TerrainVolumeIntegrator]:
    count = int(round(extent / spacing)) + 1
    origin = -0.5 * spacing * (count - 1)
    grid = TerrainGrid(count, count, spacing, spacing, origin, origin, "/World/Terrain")
    return grid, TerrainVolumeIntegrator.from_grid(grid)


def _state(
    descriptor: ToolDescriptor,
    *,
    pose: np.ndarray | None = None,
    velocity: tuple[float, float, float] = (0.0, 1.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ToolState:
    transform = np.eye(4) if pose is None else np.asarray(pose, dtype=np.float64)
    geometry = descriptor.bucket_geometry
    assert geometry is not None
    return ToolState(
        timestamp=0.0,
        pose_world=transform,
        pose_terrain=transform,
        cutting_edge_terrain=geometry.transform_points(transform, descriptor.cutting_edge_local),
        bottom_profile_terrain=geometry.transform_points(transform, descriptor.bottom_profile_local),
        left_boundary_terrain=geometry.transform_points(transform, descriptor.left_boundary_local),
        right_boundary_terrain=geometry.transform_points(transform, descriptor.right_boundary_local),
        linear_velocity=np.asarray(velocity),
        angular_velocity=np.asarray(angular_velocity),
        mouth_polygon_terrain=geometry.transform_points(transform, geometry.mouth_polygon_local),
        top_edge_terrain=geometry.transform_points(transform, geometry.top_edge_local),
        separation_plane_direction_terrain=geometry.transform_direction(
            transform, geometry.separation_plane_direction_local
        ),
        separation_plane_source="BUCKET_GEOMETRY_BOTTOM_PLATE",
        tool_plus_z_separation_fallback_used=False,
    )


def _payload(descriptor: ToolDescriptor, volume: float = 0.0) -> PayloadState:
    capacity = descriptor.bucket_geometry.effective_capacity_m3  # type: ignore[union-attr]
    return PayloadState(volume, capacity, 1800.0, np.zeros(3))


def _apply(
    descriptor: ToolDescriptor,
    grid: TerrainGrid,
    integrator: TerrainVolumeIntegrator,
    height: np.ndarray,
    *,
    state: ToolState | None = None,
    momentum: np.ndarray | None = None,
    payload: PayloadState | None = None,
    dt: float = 0.01,
):
    return BucketIntakeModel().apply(
        height,
        np.zeros(grid.shape + (2,)) if momentum is None else momentum,
        _payload(descriptor) if payload is None else payload,
        _state(descriptor) if state is None else state,
        descriptor,
        grid,
        integrator,
        dt,
        terrain_surface_m=np.zeros(grid.shape),
    )


class BucketGeometryTests(unittest.TestCase):
    def test_closed_geometry_mouth_capacity_and_frames_are_consistent(self) -> None:
        descriptor = _descriptor()
        geometry = descriptor.bucket_geometry
        assert geometry is not None
        self.assertEqual(geometry.geometry_source, GeometrySource.EXPLICIT_PROFILE)
        self.assertFalse(geometry.legacy_fallback_used)
        self.assertAlmostEqual(geometry.cutting_edge_length_m, 1.6)
        self.assertAlmostEqual(geometry.mouth_area_m2, 1.6)
        self.assertLess(geometry.mouth_planarity_error_m, 1.0e-12)
        np.testing.assert_allclose(geometry.mouth_normal_local, [0.0, -1.0, 0.0])
        np.testing.assert_allclose(geometry.bottom_plate_normal_local, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(geometry.penetration_direction_local, [0.0, 1.0, 0.0])
        self.assertAlmostEqual(geometry.geometric_capacity_m3, 3.2)
        self.assertAlmostEqual(geometry.closed_mesh_volume_m3, 3.2)
        self.assertAlmostEqual(np.linalg.det(geometry.mouth_local_frame[:3, :3]), 1.0)

        transform = _rotation_z(0.37) @ _rotation_x(-0.24)
        transform[:3, 3] = [0.3, -1.1, 2.0]
        transformed = geometry.transform_points(transform, geometry.interior_vertices_local)
        roundtrip = geometry.transform_points(np.linalg.inv(transform), transformed)
        np.testing.assert_allclose(roundtrip, geometry.interior_vertices_local, atol=1.0e-12)
        before = np.linalg.norm(geometry.cutting_edge_local[-1] - geometry.cutting_edge_local[0])
        profile_count = len(geometry.interior_profile_local)
        after = np.linalg.norm(transformed[profile_count] - transformed[0])
        self.assertAlmostEqual(before, after)

    def test_invalid_mouth_and_winding_are_rejected(self) -> None:
        geometry = _descriptor().bucket_geometry
        assert geometry is not None
        warped = np.array(geometry.mouth_polygon_local, copy=True)
        warped[-1, 1] += 0.05
        with self.assertRaisesRegex(ValueError, "non-planar"):
            replace(geometry, mouth_polygon_local=warped)
        faces = np.array(geometry.interior_faces, copy=True)
        faces[0, [1, 2]] = faces[0, [2, 1]]
        with self.assertRaisesRegex(ValueError, "winding"):
            replace(geometry, interior_faces=faces)

    def test_marker_contract_builds_authoritative_geometry_without_usd(self) -> None:
        markers = {
            "ToolOrigin": [2.0, 3.0, 0.5],
            "CuttingEdgeLeft": [2.0, 4.6, 0.5],
            "CuttingEdgeCenter": [2.0, 3.0, 0.5],
            "CuttingEdgeRight": [2.0, 1.4, 0.5],
            "BottomRearLeft": [-0.4, 4.6, 0.5],
            "BottomRearRight": [-0.4, 1.4, 0.5],
            "SideTopLeft": [-0.4, 4.6, 1.75],
            "SideTopRight": [-0.4, 1.4, 1.75],
        }
        geometry = ToolDescriptorLoader.geometry_from_marker_positions(markers)
        self.assertEqual(geometry.geometry_source, GeometrySource.MARKERS)
        self.assertEqual(geometry.geometry_quality, GeometryQuality.REDUCED_ORDER)
        self.assertGreaterEqual(len(geometry.interior_profile_local), 3)
        self.assertGreater(geometry.geometric_capacity_m3, 0.0)
        self.assertFalse(geometry.legacy_fallback_used)

    def test_high_quality_descriptor_replaces_tool_plus_z_separation(self) -> None:
        descriptor = _descriptor()
        grid, integrator = _grid(0.1)
        state = ToolKinematicsAdapter(descriptor, grid).update(np.eye(4), 0.0)
        self.assertEqual(state.separation_plane_source, "BUCKET_GEOMETRY_BOTTOM_PLATE")
        self.assertFalse(state.tool_plus_z_separation_fallback_used)
        np.testing.assert_allclose(state.separation_plane_direction_terrain, [0.0, 1.0, 0.0])
        empty = np.zeros(grid.shape, dtype=bool)
        sweep = SweepResult((0, 0, 0, 0), empty, np.full(grid.shape, np.inf), (np.eye(4),))
        intersection = ToolTerrainIntersectionModel().compute(
            np.zeros(grid.shape), sweep, state, grid, integrator
        )
        self.assertEqual(intersection.separation_plane_source, "BUCKET_GEOMETRY_BOTTOM_PLATE")
        self.assertFalse(intersection.tool_plus_z_separation_fallback_used)

    def test_legacy_geometry_is_flagged_and_rejected_by_intake(self) -> None:
        explicit = _descriptor()
        legacy = ToolDescriptor(
            tool_type="bucket",
            tool_frame_prim="/World/Bucket/ToolFrame",
            cutting_edge_local=explicit.cutting_edge_local,
            bottom_profile_local=explicit.bottom_profile_local,
            left_boundary_local=explicit.left_boundary_local,
            right_boundary_local=explicit.right_boundary_local,
            nominal_width_m=explicit.nominal_width_m,
            interior_profile_local=explicit.interior_profile_local,
            nominal_capacity_m3=explicit.geometric_capacity_m3,
        )
        self.assertTrue(legacy.bucket_geometry.legacy_fallback_used)  # type: ignore[union-attr]
        grid, integrator = _grid(0.1)
        with self.assertRaisesRegex(ValueError, "LEGACY_FALLBACK"):
            _apply(
                legacy,
                grid,
                integrator,
                np.full(grid.shape, 0.2),
                payload=_payload(legacy),
            )


class ConservativeBucketIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = _descriptor()
        self.grid, self.integrator = _grid(0.1)

    def test_uniform_perpendicular_flux_matches_analytic_solution(self) -> None:
        height = np.full(self.grid.shape, 0.4)
        result = _apply(self.descriptor, self.grid, self.integrator, height, dt=0.01)
        expected = 0.4 * 1.0 * 1.6 * 0.01
        self.assertAlmostEqual(result.bucket_inflow_volume_m3, expected, places=12)
        self.assertAlmostEqual(result.mouth_overlap_length_m, 1.6, places=12)
        self.assertAlmostEqual(result.mouth_intersection_area_m2, 0.4 * 1.6, places=12)
        self.assertFalse(result.legacy_sampling_band_used)

    def test_oblique_orientation_and_pitch_couple_to_flux(self) -> None:
        height = np.full(self.grid.shape, 0.25)
        yaw = np.deg2rad(35.0)
        oblique = _apply(
            self.descriptor,
            self.grid,
            self.integrator,
            height,
            state=_state(self.descriptor, pose=_rotation_z(yaw)),
            dt=0.01,
        )
        expected = 0.25 * 1.6 * np.cos(yaw) * 0.01
        self.assertAlmostEqual(oblique.bucket_inflow_volume_m3, expected, places=12)
        pitched = _apply(
            self.descriptor,
            self.grid,
            self.integrator,
            height,
            state=_state(self.descriptor, pose=_rotation_x(np.deg2rad(60.0))),
            dt=0.01,
        )
        self.assertLess(pitched.bucket_inflow_volume_m3, oblique.bucket_inflow_volume_m3)
        away = _apply(
            self.descriptor,
            self.grid,
            self.integrator,
            height,
            state=_state(self.descriptor, pose=_rotation_x(np.pi / 2.0)),
            dt=0.01,
        )
        self.assertAlmostEqual(away.bucket_inflow_volume_m3, 0.0, places=12)

    def test_partial_immersion_reverse_empty_and_no_overlap(self) -> None:
        height = np.full(self.grid.shape, 0.3)
        partial_pose = np.eye(4)
        partial_pose[2, 3] = 0.2
        partial = _apply(
            self.descriptor,
            self.grid,
            self.integrator,
            height,
            state=_state(self.descriptor, pose=partial_pose),
            dt=0.01,
        )
        self.assertAlmostEqual(partial.bucket_inflow_volume_m3, 0.1 * 1.6 * 0.01, places=12)
        full = _apply(self.descriptor, self.grid, self.integrator, height, dt=0.01)
        self.assertAlmostEqual(full.bucket_inflow_volume_m3, 0.3 * 1.6 * 0.01, places=12)
        reverse = _apply(
            self.descriptor,
            self.grid,
            self.integrator,
            height,
            state=_state(self.descriptor, velocity=(0.0, -1.0, 0.0)),
            dt=0.01,
        )
        self.assertEqual(reverse.bucket_inflow_volume_m3, 0.0)
        empty = _apply(self.descriptor, self.grid, self.integrator, np.zeros(self.grid.shape))
        self.assertEqual(empty.bucket_inflow_volume_m3, 0.0)
        high_pose = np.eye(4)
        high_pose[2, 3] = 2.0
        no_overlap = _apply(
            self.descriptor,
            self.grid,
            self.integrator,
            height,
            state=_state(self.descriptor, pose=high_pose),
        )
        self.assertEqual(no_overlap.bucket_inflow_volume_m3, 0.0)

    def test_capacity_saturation_and_mobile_payload_conservation(self) -> None:
        height = np.full(self.grid.shape, 0.4)
        payload = _payload(
            self.descriptor,
            self.descriptor.bucket_geometry.effective_capacity_m3 - 0.001,  # type: ignore[union-attr]
        )
        before_mobile = self.integrator.integrate(height)
        result = _apply(
            self.descriptor,
            self.grid,
            self.integrator,
            height,
            payload=payload,
            dt=0.01,
        )
        removed = before_mobile - self.integrator.integrate(result.mobile_height_m)
        gained = result.payload.volume_m3 - payload.volume_m3
        self.assertTrue(result.capacity_limited)
        self.assertAlmostEqual(gained, 0.001, places=12)
        self.assertLessEqual(result.payload.volume_m3, result.payload.capacity_m3)
        self.assertAlmostEqual(removed, gained, places=12)
        self.assertAlmostEqual(np.sum(result.admitted_volume_field_m3), gained, places=12)

    def test_nonuniform_spatial_flux_and_first_moment_converge(self) -> None:
        length = 1.73
        descriptor = _descriptor(width=length)
        h0, linear, quadratic = 0.21, 0.08, 0.11
        exact_volume_factor = h0 * length + quadratic * length**3 / 12.0
        exact_centroid_x = (linear * length**3 / 12.0) / exact_volume_factor
        volume_errors = []
        centroid_errors = []
        overlap_errors = []
        for spacing in (0.2, 0.1, 0.05):
            grid, integrator = _grid(spacing)
            _, columns = np.indices(grid.shape)
            x = grid.origin_x + columns * spacing
            height = h0 + linear * x + quadratic * x**2
            result = _apply(descriptor, grid, integrator, height, dt=0.01)
            exact = exact_volume_factor * 0.01
            volume_errors.append(abs(result.bucket_inflow_volume_m3 - exact))
            centroid_errors.append(abs(result.intake_centroid_terrain_m[0] - exact_centroid_x))
            overlap_errors.append(abs(result.mouth_overlap_length_m - length))
        self.assertLess(volume_errors[-1], 0.4 * volume_errors[0])
        self.assertLess(centroid_errors[-1], 0.4 * centroid_errors[0])
        self.assertLess(max(overlap_errors), 1.0e-12)

    def test_prescribed_time_varying_uniform_flux_converges(self) -> None:
        descriptor = _descriptor(width=1.6)
        grid, integrator = _grid(0.5, extent=4.0)
        height = np.full(grid.shape, 0.1)
        exact = 0.1 * 1.6 * 1.5
        errors = []
        for steps in (5, 10, 20, 40):
            payload = _payload(descriptor)
            total = 0.0
            dt = 1.0 / steps
            for index in range(steps):
                velocity = 1.0 + (index + 1) * dt
                result = _apply(
                    descriptor,
                    grid,
                    integrator,
                    height,
                    state=_state(descriptor, velocity=(0.0, velocity, 0.0)),
                    payload=payload,
                    dt=dt,
                )
                payload = result.payload
                total += result.bucket_inflow_volume_m3
            errors.append(abs(total - exact))
        self.assertTrue(all(later < earlier for earlier, later in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 0.2 * errors[0])


if __name__ == "__main__":
    unittest.main()
