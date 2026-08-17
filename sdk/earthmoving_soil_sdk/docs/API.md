# Public API

Normal integrations import only `SoilPhysics`, `SoilConfig`, `MaterialConfig`,
`ToolGeometry`, `ToolState`, `TrackGeometry`, `TrackState`, `SoilStepResult`,
`RLSoilFeedback`, `RLFeedbackAccumulator`, and `PhysicsDiagnostics`.

`register_tool` accepts one bucket geometry. `set_tool_state` accepts its
world-frame pose, linear/angular velocities and timestamp. Penetration, attack
angle, failure depth, FailureSurface and contact rasterization remain internal.

`register_track` accepts exactly `left_track` or `right_track` and dimensions.
`set_track_state` carries a forward-compatible schema. `belt_speed` is consumed
through the existing effective track-surface velocity input. Sprocket angular
speed and explicit longitudinal/lateral slip are stored as observations but are
not current TrackSoil constitutive inputs.

`SoilStepResult.tool_wrench` contains world force, total torque about the tool
origin, physically defined application point and residual couple. Track wrench
entries are `None`, not zero, because production TrackSoil does not compute
reaction forces.

