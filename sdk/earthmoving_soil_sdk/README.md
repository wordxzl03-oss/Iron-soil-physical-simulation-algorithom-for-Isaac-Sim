# Earthmoving Soil SDK

`earthmoving-soil 0.3.0-alpha.1` is an RL-ready reduced-order earthmoving
soil-physics alpha with bidirectional mechanical and material feedback. It
supports one bucket plus left/right tracks. Your application owns the vehicle,
articulation, controller, RL policy and Isaac loop.

This package does **not** claim DEM/MPM or full 3-D granular accuracy, validated
real iron ore, or field-calibrated force prediction. The reference material is
`NOT_YET_PHYSICALLY_CALIBRATED`; flow/arrest final closure is
`FLOW_ARREST_CLOSURE_NOT_YET_DEMONSTRATED`.

## Install

```bash
python -m pip install -e .
```

Run Isaac integration with Isaac Sim's `python.sh`, so Warp, USD and PhysX
bindings are available.

## Minimal use

```python
from earthmoving_soil import SoilPhysics, ToolGeometry, ToolState

soil = SoilPhysics(config, initial_heightmap_m)
soil.register_tool("bucket", ToolGeometry.parameterized_bucket(
    width_m=2.0, mouth_depth_m=1.0, rear_height_m=0.8, capacity_m3=1.2,
))
soil.set_tool_state("bucket", ToolState(
    pose=T_world_from_bucket_link,
    linear_velocity=v_world_m_s,
    angular_velocity=omega_world_rad_s,
    timestamp=sim_time_s,
))
result = soil.step(dt_s, phase="coordinated_cut")
force = result.tool_wrench["bucket"].force_world
payload_mass = result.rl_feedback.payload_mass_kg
```

Register tracks as `left_track` and `right_track` with `TrackGeometry`, then
provide `TrackState` every physics step. The current production TrackSoil model
consumes footprint, rigid-body velocity and an effective surface velocity; the
facade resolves optional `belt_speed` along each track's registered forward
axis. It does not currently calculate track reaction wrenches.

## Terrain and Isaac

The physics surface is always:

```text
H_free = H_resting + h_mobile
```

`IsaacSoilAdapter(sync_visual_mesh=True)` reuses the production fixed-topology,
dirty-chunk mesh path. In `GPU_RUNTIME`, only dirty `H_free` tiles are copied for
normal viewport publication. Optional contact chunks use the same samples but
are caller-scheduled because PhysX triangle-mesh recooking is not a per-frame
zero-lag contact representation.

See [Quickstart](docs/QUICKSTART.md), [API](docs/API.md),
[RL integration](docs/RL_INTEGRATION.md), and
[capabilities](docs/CAPABILITY_MATRIX.md).

