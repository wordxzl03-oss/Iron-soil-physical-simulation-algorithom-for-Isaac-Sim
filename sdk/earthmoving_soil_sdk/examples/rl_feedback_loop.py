"""Physics-rate soil feedback aggregated for a slower RL policy."""

from earthmoving_soil import RLFeedbackAccumulator


def one_policy_interval(soil, update_external_vehicle, policy_dt_s=0.1):
    physics_dt_s = soil.config.physics_dt_s
    accumulator = RLFeedbackAccumulator()
    latest = None
    elapsed = 0.0
    while elapsed + 0.5 * physics_dt_s < policy_dt_s:
        # The RL environment/controller owns and applies the action.
        update_external_vehicle(physics_dt_s)
        result = soil.step(physics_dt_s, phase="coordinated_cut")
        tool = next(iter(soil._tool_state.values()))
        accumulator.add(
            result.rl_feedback, physics_dt_s,
            tool_linear_velocity_world=tool.linear_velocity,
            tool_angular_velocity_world=tool.angular_velocity,
        )
        latest = result.rl_feedback
        elapsed += physics_dt_s
    return accumulator.emit(latest)

