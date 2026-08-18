# P0-2C Tool–Mobile momentum coupling causal report

## Decision

`ROOT_CAUSE_CLASS = COUPLING_NOT_IMPLEMENTED`

The production `GPU_RUNTIME / DEVICE` path detects a CAD bucket/terrain
intersection, transfers Resting mass to Mobile without injecting horizontal
momentum, and writes the selected contact neighborhood to `material_mask`.
The production Mobile V2 adapter, however, receives neither the tool state nor
an accepted contact impulse. It clears `v2_external_x/y` to zero at the start
of every machine frame and never consumes `material_mask`. The returned tool
impulse and tool work are therefore explicitly zero. The downstream budget,
soil-force reaction, and Isaac force application paths are present, but have
only a zero dynamic input.

`FIX_APPLIED = NO`. Porting the HOST/V1 velocity-relaxation forcing would be a
new production constitutive choice and would violate this task's prohibition
on hidden velocity forcing/injection toward the bucket. No arbitrary contact
law, force multiplier, larger contact radius, FEE reuse, or post-hoc reaction
bookkeeping was added.

## Production call graph and execution status

| Stage | Production implementation | Status |
|---|---|---|
| tool kinematics | `run_390f_v2.py` builds the real CAD `ToolState`; `ContinuousSweepBuilder.build` consumes previous/current states | `PRESENT_AND_EXECUTED` |
| intersection / proximity | `DeviceFailureZoneBridge.execute` → `ToolTerrainIntersectionModel.compute` | `PRESENT_AND_EXECUTED` |
| contact support | `intersection.affected_mask & (mobile_after > 0)` plus 4-neighbor stencil → device `material_mask` | `PRESENT_AND_EXECUTED` |
| R→M activation | `DeviceBulkState.entrain_host_indices`; q is deliberately unchanged | `PRESENT_AND_EXECUTED` |
| requested dynamic impulse | no production Mobile V2 contact law/operator | `ABSENT` |
| accepted dynamic impulse | no acceptance operator; activation impulse is explicitly zero | `PRESENT_ZERO_INPUT` |
| source assembly | persistent `v2_external_x/y` arrays exist but are cleared to zero; `material_mask` is not read | `PRESENT_ZERO_INPUT` |
| Mobile update | `GpuBulkOperatorChain.step_mobile` → `WarpProductionMobileV2Solver.step_resident` → `apply_update_and_sources` | `PRESENT_AND_EXECUTED` |
| Mobile tool ΔP diagnostic | returned `tool_impulse_on_mobile_terrain_ns=np.zeros(3)` | `TELEMETRY_ONLY` |
| action–reaction budget | `EarthmovingPhysicsCore._step_gpu` → `MobileMomentumBudget` → `SoilForceModel.compute` | `PRESENT_AND_EXECUTED`, zero dynamic input |
| Isaac reaction | `RigidPrim.apply_forces_and_torques_at_pos` uses `core_result.applied_force_terrain_n` when nonzero | `PRESENT_AND_EXECUTED`, zero dynamic contribution |
| frozen trajectory telemetry | dynamic Mobile force recorded per retained frame | `PRESENT_AND_EXECUTED`, always zero |

The exact break is between contact-support construction and requested Mobile
momentum transfer:

```text
real 390F ToolState / CAD sweep
  -> DeviceFailureZoneBridge.execute
  -> contact_neighborhood -> DEVICE material_mask
  -> [ABSENT: contact kinematics + constitutive law -> requested/accepted J]
  -> GpuBulkOperatorChain.step_mobile(dt only)
  -> WarpProductionMobileV2Solver clears external_x/y
  -> apply_update_and_sources(external = 0)
  -> WarpMobileStep(tool impulse = 0, tool work = 0)
  -> MobileMomentumBudget(machine reaction = -0)
  -> SoilForceModel(dynamic force = 0/dt)
  -> Isaac rigid-body force path receives no dynamic component
```

The contact mask is cleared only after Mobile V2 returns, so this is not a
stale-buffer/early-clear defect. The mask is live during the Mobile call but
the selected solver never reads it.

## Mobile V2 source and units audit

The Warp kernel applies transport first. For the post-transport depth `h` and
momentum state `q=h*u`, it then performs

```text
u1 = u_transport + external_acceleration * dt_sub
q1 = h * u1
```

before basal friction. `external_x/y` therefore have units m/s²; they are not
force or impulse arrays. Kernel diagnostics 2/3 accumulate
`sum(A_i * h_i * delta_u_i)` using the authoritative triangle A–C vertex
dual-control areas. Density is not applied inside the kernel. A future adapter
must convert that source-only integral to physical impulse exactly once:

```text
h                         [m]
q = h*u                   [m²/s]
A*q                       [m4/s]
rho*A*q                   [kg*m/s] = [N*s]
J_tool_to_mobile
  = rho * sum_i(A_i*h_i*a_i*dt_sub)
```

The source is after flux transport and before friction. Therefore source-only
momentum must be measured at that boundary, not reconstructed from total
before/after Mobile momentum. Spatial/contact weights, if introduced by the
future contact model, must be applied once when forming the per-cell accepted
source; the existing `weights` array is geometric dual area and must not be
reused as a contact multiplier.

## First expected event

The accepted frozen coordinated curl-scoop retained frame-level diagnostics,
not Mobile CFL substeps or the compact `material_mask`. The earliest retained
conservative proxy satisfying CUT_AND_FILL + mouth-prism Mobile volume above
`1e-12 m³` + positive relative-normal speed is:

| Quantity | Value |
|---|---:|
| simulation time | 5.733333632 s |
| contact Mobile volume proxy | 0.0002749934962 m³ |
| mouth intersection area proxy | 0.06068863394 m² |
| relative normal velocity | 0.2221676790 m/s |
| cutting-edge velocity | (-0.16014815, -0.25620433, -1.99563926) m/s |
| mean mouth-prism Mobile velocity | (0, 0) m/s |
| requested impulse | absent |
| accepted / Mobile source / machine reaction | all zero |

This establishes a physically expected interaction and locates the first
causal break at the missing request law. It is not represented as exact
substep contact telemetry. The JSON preserves `null` rather than fabricating
physics-step, CFL-substep, tangential-relative-velocity, or exact mask-volume
values.

Across the frozen frame series, 17 retained frames meet that same proxy. Peak
mouth-prism Mobile volume is `0.001329627878 m³`; peak dynamic Mobile momentum
force remains `0 N`.

## Why the zero ledger is not a pass

Production currently gives

```text
accepted J = 0
measured Mobile tool delta-P = 0
machine reaction = 0
action/reaction residual = 0
```

The residual is mathematically zero but physically vacuous. The action/
reaction artifact labels it `NOT_VALIDATED_TRIVIAL_ZERO_BECAUSE_COUPLING_ABSENT`.
It must not be cited as production coupling acceptance.

The separately exported `ToolMobileSubstepContract` and
`ToolMobileFrameContractLedger` only specify/test conservation once a reviewed
contact operator supplies an accepted impulse. They are not imported by the
production physics path and cannot generate a force.

Four focused contracts pass:

1. external acceleration gives `rho*sum(A*h*a*dt)`;
2. an accepted synthetic `J` yields Mobile `+J` and reaction `-J`;
3. CFL impulses sum once per machine frame without dt double-scaling;
4. zero contact cannot leak a dynamic impulse.

## Required interface for the next design review

The smallest defensible future operator should consume:

- exact CAD tool surface/swept geometry, contact normals/tangents and support;
- cutting-edge plus local rigid-body point velocity (`v + omega × r`);
- post-transport Mobile `h`, `q`, velocity and authoritative dual areas;
- candidate contact weights with an explicit geometric meaning;
- the Mobile CFL substep `dt_sub` and machine-frame identity.

It should produce, per substep:

- requested and admissibility-limited per-cell tool→Mobile impulses;
- accepted `J_k = rho*sum(A*delta_q_tool)` measured from the actual source write;
- the exact machine command `-J_k`, summed once to `-sum_k J_k` per 60 Hz frame;
- source-boundary momentum ledger and a work/dissipation ledger with a stated
  contact work velocity;
- no impulse under separation/no-contact and no R→M activation impulse.

The review must choose and justify the contact constitutive/admissibility law,
including non-penetration, no-suction, tangential dissipation and maximum work
behavior. P0-2C intentionally does not choose those semantics.

## Replay and scope

No new production CUDA replay was run because the current execution sandbox
does not expose the NVIDIA device (`nvidia-smi` cannot communicate with the
driver). P0-2B's formal CUDA replay is likewise still marked
`NOT_RUN_CUDA_DEVICE_NOT_EXPOSED`; its dual-control-volume implementation was
not modified here.

The causal conclusion uses the accepted real production replay that already
reached PRE_DUMP with:

- net payload gain: `0.03802625208 m³`;
- gross capture ratio: `0.03745731071`;
- maximum penetration: `0.6560070643 m`;
- peak quasi-static soil force: `216159.1912 N`;
- CUT/CURL maximum mass error: `2.137312549e-11 m³`;
- inherited full PRE_DUMP maximum error before formal P0-2B replay:
  `0.0002190679920 m³`.

Morphology remains explicitly out of scope and unchanged:
`VISIBLE_NEEDLE_FOREST=YES`, `VISIBLE_TRIANGULAR_FINS=YES`, and
`GRID_SCALE_SPIKE_PROLIFERATION=YES`.

## Artifacts

- `outputs/mobile_v2_production/tool_mobile_momentum_causal_report.json`
- `outputs/mobile_v2_production/tool_mobile_momentum_timeseries.json`
- `outputs/mobile_v2_production/tool_mobile_first_expected_coupling.json`
- `outputs/mobile_v2_production/tool_mobile_action_reaction_ledger.json`
- `tests/test_tool_mobile_momentum_contract.py`

No before/after files exist because no production fix was applied.
