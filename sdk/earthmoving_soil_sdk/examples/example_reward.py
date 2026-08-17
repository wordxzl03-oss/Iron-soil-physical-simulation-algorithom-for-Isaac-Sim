"""EXAMPLE_ONLY — reward policy belongs to the external RL environment."""


def example_only_reward(feedback, cycle_time_s):
    force = feedback.bucket_force_peak_since_last_rl_step_n or 0.0
    work = feedback.soil_work_delta_j or 0.0
    spill = feedback.spill_mass_delta_kg or 0.0
    return (
        + feedback.captured_mass_delta_kg
        - 1.0e-5 * force
        - 1.0e-5 * abs(work)
        - 0.5 * spill
        - 0.01 * cycle_time_s
    )

