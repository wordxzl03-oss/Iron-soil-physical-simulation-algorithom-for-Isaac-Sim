# Quickstart

1. Install with `python -m pip install -e .`.
2. Load `MaterialConfig.from_yaml("configs/materials/iron_ore_reference.yaml")`.
3. Construct a metre-based `TerrainGrid`, then `SoilConfig` and `SoilPhysics`.
4. Register one bucket; optionally register `left_track` and `right_track`.
5. Supply world poses and velocities every physics step.
6. Call `soil.step(dt)` and apply the returned bucket wrench in your vehicle SDK.
7. Read `result.rl_feedback` at the physics rate or use
   `RLFeedbackAccumulator` at a slower policy rate.

Frames use column-vector homogeneous transforms. `pose` is
`T_world_from_body`; distance, time, force, torque, mass and angular velocity
units are m, s, N, N·m, kg and rad/s.

Isaac users construct `IsaacSoilAdapter` with their existing Stage and bound
RigidPrim-like bodies. The adapter is called from, but never owns, the user's
simulation loop.

