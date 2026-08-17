from __future__ import annotations

import unittest

import numpy as np

from isaac_bulk_pipeline.planning import (
    ArticulatedLatticeConfig,
    ArticulatedState,
    AttackCandidateGenerator,
    ClassicalAttackEvaluator,
    GoalObservation,
    NonlinearPathTracker,
    PlannerTerrainView,
    RecedingLoadingPlanner,
    StateLatticePlanner,
    TaskGoal,
    TaskGoalKind,
    TerrainCostConfig,
    VehicleSafetyMonitor,
    VehicleStabilityObservation,
    smooth_path,
)
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.vehicle import VehicleCommand


def scene():
    grid = TerrainGrid(81, 61, 0.5, 0.5, -10.0, -10.0, "/World/Terrain")
    x = grid.origin_x + np.arange(grid.nx) * grid.dx
    y = grid.origin_y + np.arange(grid.ny) * grid.dy
    xx, yy = np.meshgrid(x, y)
    pile = 2.2 * np.exp(-((xx - 15.0) ** 2 / 25.0 + yy**2 / 16.0))
    view = PlannerTerrainView.derive(
        grid,
        pile,
        TerrainCostConfig(maximum_slope_deg=45.0, maximum_roughness_m=0.5),
    )
    return grid, pile, view


class TestPhaseJPlanning(unittest.TestCase):
    def test_all_goal_types_are_value_driven(self) -> None:
        obs = GoalObservation(5.0, 2000.0, 100.0, 90.0, np.ones((3, 3)))
        self.assertTrue(TaskGoal(TaskGoalKind.TARGET_MASS, 9000.0).evaluate(obs).complete)
        self.assertTrue(TaskGoal(TaskGoalKind.TARGET_VOLUME, 4.0).evaluate(obs).complete)
        self.assertTrue(TaskGoal(TaskGoalKind.TARGET_REMAINING_RATIO, 0.91).evaluate(obs).complete)
        target = np.full((3, 3), 0.98)
        self.assertTrue(TaskGoal(TaskGoalKind.TARGET_HEIGHTMAP, target_heightmap_m=target, height_tolerance_m=0.03).evaluate(obs).complete)

    def test_planner_view_is_derived_from_heightmap(self) -> None:
        _, height, view = scene()
        self.assertEqual(view.slope_rad.shape, height.shape)
        self.assertGreater(float(view.slope_rad.max()), 0.0)
        self.assertFalse(view.cost.flags.writeable)
        self.assertFalse(view.slope_rad.flags.writeable)

    def test_candidates_follow_current_pile_front(self) -> None:
        _, height, view = scene()
        generator = AttackCandidateGenerator(spacing_m=1.5, pre_dig_distance_m=3.0)
        candidates = generator.generate(view, height)
        self.assertGreater(len(candidates), 3)
        for candidate in candidates:
            vector = candidate.attack_pose_xy_yaw[:2] - candidate.pre_dig_pose_xy_yaw[:2]
            vector /= np.linalg.norm(vector)
            np.testing.assert_allclose(vector, candidate.penetration_direction_xy, atol=1e-10)
            self.assertGreater(candidate.estimated_accessibility, 0.0)

    def test_evaluator_logs_pareto_and_capacity(self) -> None:
        _, height, view = scene()
        candidate = AttackCandidateGenerator(spacing_m=2.0, pre_dig_distance_m=3.0).generate(view, height)[0]
        evaluator = ClassicalAttackEvaluator(bucket_width_m=2.7, bucket_capacity_m3=3.0, assumed_density_kg_m3=2100.0)
        result = evaluator.evaluate(candidate, np.array([0.0, 0.0, 0.0]))
        self.assertLessEqual(result.estimated_payload_volume_m3, 3.0)
        self.assertEqual(set(result.pareto_values), {"delivered_mass_kg", "time_s", "energy_j", "travel_m", "steering_rad", "slope_risk", "stability_risk"})

    def test_lattice_uses_articulation_and_footprint(self) -> None:
        grid = TerrainGrid(81, 61, 0.5, 0.5, -10.0, -10.0, "/World/Terrain")
        height = np.zeros(grid.shape)
        view = PlannerTerrainView.derive(grid, height)
        planner = StateLatticePlanner(ArticulatedLatticeConfig(vehicle_length_m=2.0, vehicle_width_m=1.5, primitive_length_m=1.0, max_expansions=6000))
        path = planner.plan(view, ArticulatedState(0.0, 0.0, 0.0, 0.0), np.array([6.0, 2.0, 0.0]))
        self.assertTrue(path.reached_goal)
        self.assertGreater(len(path.states), 2)
        self.assertTrue(any(abs(state.articulation_rad) > 1e-6 for state in path.states))

    def test_smoothing_and_nonlinear_tracker_output_vehicle_command(self) -> None:
        states = (
            ArticulatedState(0.0, 0.0, 0.0, 0.0),
            ArticulatedState(1.0, 0.2, 0.2, 0.1),
            ArticulatedState(2.0, 0.8, 0.5, 0.2),
        )
        path = smooth_path(states)
        self.assertEqual(path.shape[1], 4)
        command = NonlinearPathTracker().command(states[0], 0.2, path)
        self.assertIsInstance(command, VehicleCommand)
        self.assertTrue(-1.0 <= command.steering <= 1.0)

    def test_receding_planner_stops_on_target_not_scoop_count(self) -> None:
        _, height, view = scene()
        goal = TaskGoal(TaskGoalKind.TARGET_VOLUME, 1.0)
        planner = RecedingLoadingPlanner(
            goal,
            AttackCandidateGenerator(spacing_m=2.0, pre_dig_distance_m=3.0),
            ClassicalAttackEvaluator(bucket_width_m=2.7, bucket_capacity_m3=3.0, assumed_density_kg_m3=2100.0),
            StateLatticePlanner(ArticulatedLatticeConfig(vehicle_length_m=2.0, vehicle_width_m=1.5, max_expansions=6000)),
        )
        obs = GoalObservation(1.1, 2100.0, 100.0, 98.9, height)
        decision = planner.replan(obs, view, ArticulatedState(0.0, 0.0, 0.0, 0.0))
        self.assertTrue(decision.stop)
        self.assertTrue(decision.goal_status.complete)
        self.assertIsNone(decision.selected_attack)

    def test_safety_monitor_uses_contacts_attitude_and_com_support(self) -> None:
        monitor = VehicleSafetyMonitor()
        polygon = np.array([[-1.5, -1.0], [1.5, -1.0], [1.5, 1.0], [-1.5, 1.0]])
        safe = monitor.evaluate(VehicleStabilityObservation(0.05, 0.1, (True, True, True, True), np.array([0.0, 0.0]), polygon))
        unsafe = monitor.evaluate(VehicleStabilityObservation(0.05, 0.1, (True, False, False, True), np.array([2.0, 0.0]), polygon))
        self.assertTrue(safe.safe)
        self.assertFalse(unsafe.safe)
        self.assertTrue(unsafe.wheel_contact_loss)
        self.assertTrue(unsafe.com_outside_support)


if __name__ == "__main__":
    unittest.main()
