# EARTHMOVING_SOIL_SDK_ALPHA_RELEASE_REPORT

## Identity

- PACKAGE_PATH: `/home/eric/Desktop/mesh/sdk/earthmoving_soil_sdk`
- VERSION: `0.3.0-alpha.1`
- RECOMMENDED_TAG: `v0.3.0-alpha.1`
- PRIVATE_GITHUB_READY: YES — source-ready, not published or pushed
- GIT_STATUS: workspace has no usable Git executable/metadata; no commit or push performed

## Public API

`SoilPhysics`, `SoilConfig`, `MaterialConfig`, `ToolGeometry`, `ToolState`,
`TrackGeometry`, `TrackState`, `ReactionWrench`, `SoilStepResult`,
`RLSoilFeedback`, `RLFeedbackAccumulator`, `PhysicsDiagnostics`.

## Package tree

```text
earthmoving_soil_sdk/
  src/
    earthmoving_soil/{core,terrain,interaction,running_gear,gpu,isaac,diagnostics,rl}
    isaac_bulk_pipeline/       # frozen production implementation snapshot
  configs/{materials,solver,runtime}
  examples/
  tests/
  docs/
  pyproject.toml README.md VERSION CHANGELOG.md LICENSE .gitignore
```

## Source modules reused

Production core, Failure Surface V3, intersection, FEE/SoilForce, Mobile,
LargeAvalanche, residual MiniSlope, payload/intake/retention, airborne,
deposition, TrackSoil, mass ledger, DeviceBulkState, GPU bridges/Warp kernels,
dirty visual mesh and dirty PhysX contact chunks are byte-identical frozen
copies of the current source modules. Only the extracted internal
`runtime/__init__.py` export list is narrowed to remove vehicle/presentation
orchestration imports. No physics equation or parameter changed.

## Files excluded

390F USD/config/controller/launcher, all other vehicle/robot/operation code,
wheel/tire code, GUI/presentation logic, research planning/dataset/telemetry,
acceptance forensic output, logs, checkpoints, videos, screenshots, large NPZ,
credentials and unrelated RL code.

## Capabilities

- BUCKET: excavation, world reaction force/torque/application point, payload,
  Mobile, terrain change, deposition and mass ledger are exposed.
- TRACKS: left/right footprint, effective belt-surface velocity, conservative
  sinkage/rutting are exposed. Track reaction wrench, normal support, traction
  and lateral slip are not currently computed.
- ISAAC_VISUAL_TERRAIN_SYNC: PASS
- VISUAL_SOURCE: H_FREE
- PHYSICS_AND_VISUAL_TERRAIN_SHARED_AUTHORITY: YES
- CONTACT_SYNC: PARTIAL — exact-resolution dirty static-mesh recook at
  caller-selected cadence, with explicit possible lag

## RL feedback classification

- AVAILABLE_DIRECTLY: bucket wrench, penetration/contact geometry when active,
  yielded area, failure volume, payload volume, displaced/spill/deposited
  deltas when the relevant operator runs, Mobile volume/motion, terrain state,
  mass ledger error and versioned production diagnostics.
- DERIVABLE_WITHOUT_NEW_PHYSICS: uniform-density mass/deltas, force norm,
  mean/peak wrench, impulse, soil power/work, local `H_free` patch and its
  temporal delta.
- NOT_AVAILABLE: track reaction wrench, separate sprocket constitutive input,
  explicit longitudinal/lateral slip closures, compaction/density evolution.
  These are `None`/explicitly unavailable, never fake zeros.

## Install

```bash
python -m pip install -e /home/eric/Desktop/mesh/sdk/earthmoving_soil_sdk
```

For the archive: `python -m pip install earthmoving_soil_sdk-0.3.0-alpha.1.zip`.
Isaac/GPU execution must use Isaac Sim's Python/Kit extension environment.

## Minimal integration

```python
soil = SoilPhysics(config, H0)
soil.register_tool("bucket", bucket_geometry)
soil.set_tool_state("bucket", ToolState(pose, v, omega, sim_time))
step = soil.step(dt, phase="coordinated_cut")
vehicle.apply(step.tool_wrench["bucket"])
observation = step.rl_feedback
```

## Test results

- Lean packaging/integration suite: `7 passed`.
- Wheel construction: PASS (`earthmoving_soil-0.3.0a1-py3-none-any.whl`).
- Isaac 4.5 package import after Kit startup: PASS.
- Warp/CUDA capability in real Isaac environment: PASS.
- GPU `DEVICE`, 701×701 at 0.05 m, production step → dirty USD visual mesh:
  PASS; 9 dirty visual tiles updated, valid USD chunk authored.
- Secret/proprietary binary scan: PASS; no vehicle/model binaries included.

## Capability gaps and physics limitations

Track wrench/traction and density evolution remain gaps. Reference material is
`NOT_YET_PHYSICALLY_CALIBRATED`. The package preserves
`FLOW_ARREST_CLOSURE_NOT_YET_DEMONSTRATED`; it does not reuse structural V3
acceptance as proof of final arrest closure.

