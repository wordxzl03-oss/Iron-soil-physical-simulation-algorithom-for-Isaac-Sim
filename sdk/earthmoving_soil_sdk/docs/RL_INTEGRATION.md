# RL integration

The feedback layer reads physics outputs and never changes soil state. Physics
can run at 60 Hz while a policy consumes one accumulated observation at a lower
rate.

`RLFeedbackAccumulator.add(feedback, dt, tool_linear_velocity_world=...,
tool_angular_velocity_world=...)` integrates impulse as `J += F dt` and work as
`dW = (F·v + tau·omega) dt`. `emit(latest)` returns mean/peak wrench, impulse,
work and mass deltas, then resets the interval.

Mass uses the authoritative uniform scenario density:
`mass = assumed_bulk_density * represented_material_volume`. It is never
inferred from a visual payload mesh.

Field status is included in `feedback.availability`. Direct: bucket wrench,
failure/yield geometry, payload volume, Mobile state and ledger error.
Derived without new physics: mass, deltas, impulse/work and local terrain
patches. Not available: track wrench and density/compaction evolution.

Reward belongs in the external RL environment. See `examples/example_reward.py`
for a deliberately non-production sketch.

