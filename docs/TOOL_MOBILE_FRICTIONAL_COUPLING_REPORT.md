# P0-2D Conservative Tool–Mobile frictional impulse coupling

## Status

The production coupling is implemented. Synthetic tests and the real
production Warp kernel pass on Warp's CPU device. Formal 390F CUDA PRE_DUMP
acceptance is **not demonstrated** because the execution sandbox cannot
enumerate the NVIDIA/Vulkan device; Isaac reported driver version 0 and failed
before Stage/physics initialization. No physics result was produced by that
failed attempt.

This is therefore:

```text
IMPLEMENTED_SYNTHETIC_DEVICE_PASS_PRODUCTION_REPLAY_ENVIRONMENT_BLOCKED
```

It is not labelled a production acceptance pass.

## Implemented physical law

For every geometry-confirmed Mobile dual-control prism, the control mass is

```text
m_i = rho_bulk * A_i * h_i
```

where `A_i` is the authoritative Triangle-A-C vertex area. No uniform `dx*dy`
substitute is used at boundaries.

The contact geometry uses the existing CAD-derived 390F L1 closed triangle
descriptor. Candidate Mobile prisms must pass an exact triangle/AABB
intersection (or closed-cavity containment) test. The operator then computes
the closest surface point, a tool-to-material normal, and

```text
v_T(x) = v_linear + omega × (x - x_tool_origin).
```

`material_mask` remains a compact broad-phase hand-off; it is not accepted as
contact proof by itself.

With `v_rel = u_i - v_T` and a normal pointing from tool to Mobile, closing is
`v_n = dot(v_rel,n) < 0`. The parameter-free normal impulse is the minimum
inelastic, infinite-kinematic-wall impulse that removes only this closing
component:

```text
J_n = -m_i * min(v_n, 0) * n.
```

It is non-attractive and introduces no stiffness, relaxation time, penalty,
capture coefficient or force multiplier.

Tangential slip is resolved after the normal impulse using the
maximum-dissipation Coulomb projection:

```text
|J_t| = min(m_i*|v_t|, mu_tool_mobile*|J_n|)
```

with direction opposite slip. The coefficient is the existing
`MaterialScenario.tool_friction_coefficient`, whose established semantic is
the soil–tool wall interface. Internal friction and Mobile basal friction are
not reused. No new uncalibrated parameter was added.

Because Mobile V2 is depth-averaged and stores only horizontal momentum, a
surface whose normal has no horizontal projection is explicitly excluded;
inventing vertical Mobile momentum would require a different model.

## Operator order and conservation

The production DIG order is now:

```text
FailureSurface R→M mass-only activation (q unchanged)
  -> Mobile V2 shared-face transport / pressure
  -> geometry-confirmed Tool–Mobile impulse per CFL substep
  -> general external acceleration source (currently zero)
  -> Mobile basal friction
  -> bucket intake
  -> existing deposition / later operators
```

The accepted tool impulse is written directly to `q` as the equivalent
`delta_q = J/(rho*A)`. The production solver accumulates all CFL-substep
impulses once into `WarpMobileStep.tool_impulse_on_mobile_terrain_ns`.
`MobileMomentumBudget` uses that same value to issue machine reaction
`-sum(J_k)/dt_frame`; it is not recomputed from FEE. The corresponding angular
impulse `sum(r×J)` is carried through the budget, and the Isaac rigid-body call
now applies the existing residual couple instead of dropping it.

The legacy soil-force cap now applies only to the quasi-static FailureSurface
estimate. It cannot scale a measured Mobile reaction independently of the
`+J` already committed to Mobile.

Quasi-static failure force, dynamic Mobile reaction and payload/deadload remain
separate channels. R→M activation still injects zero horizontal momentum.

## Energy and telemetry

Each CFL substep records contact count/area/volume, requested and accepted
normal/tangential impulse, Mobile contact-only kinetic-energy change,
tool-to-Mobile work, machine reaction work, frictional and total contact
dissipation, and the exact opposite machine reaction.

The CUT_AND_FILL audit now retains those substeps and reports frame/cycle
totals plus action–reaction and unexplained-energy residuals.

## Synthetic acceptance

Pure-law tests cover:

1. moving wall into stationary Mobile;
2. separating wall with zero attractive impulse;
3. tangential slip and Coulomb bound;
4. static relative contact with no energy injection;
5. multi-cell action–reaction closure;
6. multi-substep, one-machine-frame accumulation without dt double scaling;
7. unequal Triangle-A-C areas using physical `rho*A*h` mass.

The same source path then ran through the actual
`WarpProductionMobileV2Solver` kernel on Warp CPU:

| Metric | Result |
|---|---:|
| Mobile volume before / after | 0.01600000000000001 / 0.016000000000000004 m³ |
| mass residual | -6.938893904e-18 m³ |
| normal impulse | 1.808152471 N·s |
| tangential impulse | 0.7232609885 N·s |
| Mobile impulse | (1.808152471, 0.7232609885, 0) N·s |
| machine reaction | exact negative |
| action–reaction residual | 0 N·s |
| tool work | 2.169782965 J |
| frictional dissipation | 0.1851916531 J |
| total contact dissipation | 0.8906008001 J |
| unexplained contact energy | 2.498001805e-16 J |
| active CFL substeps | 5 |

P0-2B canonical dual-control-volume tests were rerun on Warp CPU after source
integration and remain PASS; shared-face transport equations were not changed.

Focused local regression: 25 passed, 1 skipped under the ordinary venv (the
skip is Warp unavailable there), with the known unrelated Phase-H
width-monotonicity test explicitly deselected. The Isaac/Warp CPU device test
was then run separately and passed. The missing external `slope_model` fixture
also remains outside P0-2D.

## Formal replay blocker

The unchanged production command was attempted once. Isaac failed before
loading the Stage with `No device could be created`, driver version 0, and a
PhysX CUDA-context error. Therefore these production-only values remain
unmeasured: total cycle impulse, peak dynamic force, work/dissipation, payload,
penetration and mass error. `PRE_DUMP_REACHED=NO` means the attempted run never
entered physics, not that the trajectory failed.

The one remaining blocker is:

```text
CUDA_VULKAN_DEVICE_NOT_EXPOSED_TO_EXECUTION_SANDBOX_FOR_390F_REPLAY
```

## Files

- `src/isaac_bulk_pipeline/bulk_interaction/tool_mobile_contact.py`
- `src/isaac_bulk_pipeline/bulk_interaction/warp_mobile_v2.py`
- `src/isaac_bulk_pipeline/experimental/mobile_v2_warp.py`
- `src/isaac_bulk_pipeline/runtime/gpu_failure_bridge.py`
- `src/isaac_bulk_pipeline/runtime/gpu_bulk_operator_chain.py`
- `src/isaac_bulk_pipeline/runtime/v2_physics_core.py`
- `src/isaac_bulk_pipeline/soil_force/model.py`
- `isaac_loader/run_390f_v2.py`
- `tests/test_tool_mobile_frictional_coupling.py`
- `outputs/mobile_v2_production/tool_mobile_p0_2d_synthetic_device.json`
- `outputs/mobile_v2_production/tool_mobile_p0_2d_report.json`
