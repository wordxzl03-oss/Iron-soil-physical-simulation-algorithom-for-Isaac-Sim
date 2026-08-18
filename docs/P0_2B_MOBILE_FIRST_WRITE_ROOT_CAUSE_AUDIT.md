# P0-2B Mobile first-write root-cause audit

## Decision

The exact first different write is the geometry-confirmed
`TOOL_MOBILE_IMPULSE` in production physics step 169 at
`t = 5.6333336271345615 s`, while the task state is `PENETRATE`.

This audit made no physics, material, trajectory, grid, timestep, gate or
ownership change.  The replay used the production 390F Stage,
`GPU_RUNTIME / DEVICE`, the 701×701 grid and 0.05 m spacing.  The full machine
record is
`outputs/mobile_v2_production/p0_2b_mobile_first_write_root_cause_audit.json`.

## Exact first creation and first momentum write

The first `Resting -> Mobile` transaction activated seven cells and
`0.0005967213875513226 m3`.  Its mode was
`GEOMETRIC_SWEEP_ONLY_OUTSIDE_FEE_FORCE_DOMAIN`.  Immediately after the
authoritative mass commit:

- integrated Mobile momentum was exactly `[0, 0] kg m/s`;
- the count of cells with non-zero horizontal momentum was exactly zero;
- the implementation rule was the documented mass-only `entrain_host_indices`
  transaction, which does not write `momentum_x` or `momentum_y`.

Therefore `R2M_HORIZONTAL_MOMENTUM_CONTRACT = PASS`.

The first Mobile CFL substep then used `dt_sub = 0.015938609232869022 s`.
Its exact integrated source decomposition was:

| Fused-kernel contribution | impulse x (kg m/s) | impulse y (kg m/s) |
|---|---:|---:|
| Shared-face + hydrostatic/topography correction | -0.0128141644468 | -0.0037525261580 |
| Tool-Mobile | -0.0901355138237 | -0.2450743866514 |
| External source | 0 | 0 |
| Basal friction | +0.0153914559349 | +0.0279915925890 |
| Actual state delta | -0.0875582223356 | -0.2208353202204 |

The components close exactly to the actual state delta at stored precision.
The Tool-Mobile component is the first and dominant new momentum source; it
exists before bucket intake, deposition or the next task-state transition.

The largest per-cell Tool-Mobile impulse occurs at grid cell `[y=392, x=108]`:

- control area: `0.0025 m2`;
- Mobile depth after the face update: `0.0776150209656 m`;
- pre-contact velocity: `[-0.011594710214, -0.005776345925] m/s`;
- tool surface velocity: `[-0.303714871529, -0.522199041253] m/s`;
- tool-to-Mobile contact normal: `[0.422211447322, -0.906497376582]`;
- closing speed: `-0.344799342421 m/s`;
- accepted Tool-Mobile impulse:
  `[-0.0181924045630, -0.1095861412080] kg m/s`;
- contact point in terrain coordinates:
  `[5.38738295353, 19.6270890796, 0.345739392745] m`.

The negative closing speed satisfies the implemented non-attractive contact
condition.  The per-cell reconstruction sums to the Warp kernel's integrated
Tool-Mobile diagnostic to floating-point precision.  This identifies the
operator and cell; it does not establish that the already accepted contact law
should be changed.

## True production ordering

The observed DIG ordering is:

```text
BEFORE_FAILURE_SURFACE
AFTER_FAILURE_SURFACE_GEOMETRY
AFTER_R2M_ACTIVATION
BEFORE_MOBILE_V2
AFTER_MOBILE_FUSED_SUBSTEP
AFTER_MOBILE_FUSED_FACE_TOPOGRAPHY_TOOL_FRICTION
AFTER_BUCKET_INTAKE
AFTER_LARGE_AVALANCHE
END_BULK_CORE
```

Production does not have separately publishable terrain states named
`AFTER_MOBILE_FACE_TRANSPORT`, `AFTER_MOBILE_TOPOGRAPHY_SOURCE`,
`AFTER_TOOL_MOBILE_IMPULSE` and `AFTER_BASAL_FRICTION`.  Those operations are
fused in `apply_update_and_sources`.  The audit therefore uses the kernel's
existing conservative diagnostic vector and an exact read-only replay of its
per-cell algebra; it does not invent intermediate state writes.

At the known CUT-entry timestamp `t = 5.700000297278166 s`, the audit's
`AFTER_R2M_ACTIVATION` state reproduces the frozen current checkpoint exactly:
Mobile volume `0.01640141674169811 m3` and momentum
`[-0.7655750134105322, -1.3701451864801428] kg m/s`.  The prior frozen baseline
has only `[-0.0333347374626, -0.0129907055621] kg m/s`, so the newly localized
PENETRATE write closes the earlier `-0.2 < tau <= 0` bracket.

## Dual-control-volume and thin-layer checks

For the first event:

- actual Mobile volume change across each transport substep is zero;
- positivity-clipped volume is zero;
- weighted shared-face mass residual is zero;
- weighted shared-face momentum residual is zero;
- the independently reconstructed topography-pressure impulse matches the
  kernel diagnostic component;
- the highest-ranked affected face uses equal interior areas
  `A_i = A_j = 0.0025 m2`.

Thus `DUAL_CV_IMPLEMENTATION_ERROR = NO`.  The first different write is not an
unequal-area dual-CV effect, so
`PHYSICALLY_EXPECTED_DUAL_CV_DIVERGENCE = NO` for this event.

The first-creation sample and the following ten production physics samples
were retained.  Although positive depths reach floating-point-small advected
tails, extreme velocity is not concentrated there.  Across the retained
samples, maximum speed in `h < 1e-4 m` reaches about `0.798274 m/s`, while the
`h >= 1e-2 m` population reaches about `0.763129 m/s`; momentum in the thin
tail remains small and no isolated speed blow-up appears.  Therefore
`THIN_LAYER_VELOCITY_PATHOLOGY = NO`.

## Optional Isaac diagnostic visualization

`--mobile-first-write-visual-diagnostic` adds non-physical USD overlays sourced
from the same DEVICE readbacks: Mobile thickness points, authoritative `q/h`
velocity arrows, new R2M cells, exact Tool-Mobile contact cells, bucket mouth
and positive direction, max-speed marker and a compact HUD.  It neither writes
soil state nor replaces the production visual/collision terrain.  The formal
numerical replay was headless, so `VISUAL_DIAGNOSTIC = NOT_RUN`; video/frame
generation was correctly not made a blocker.

## Required result

```text
FIRST_NONZERO_MOBILE_TIME_S:
5.6333336271345615

FIRST_NONZERO_MOBILE_PHASE:
PENETRATE

FIRST_NONZERO_MOBILE_VOLUME_M3:
0.0005967213875513226

R2M_HORIZONTAL_MOMENTUM_CONTRACT:
PASS

MOMENTUM_IMMEDIATELY_AFTER_R2M_KG_M_S:
[0.0, 0.0]

FIRST_LARGE_MOMENTUM_CHANGE_TIME_S:
5.6333336271345615

FIRST_LARGE_MOMENTUM_CHANGE_OPERATOR:
TOOL_MOBILE_IMPULSE

MOMENTUM_BEFORE_KG_M_S:
[0.0, 0.0]

MOMENTUM_AFTER_KG_M_S:
[-0.0875582223356095, -0.22083532022039812]

DELTA_MOMENTUM_KG_M_S:
[-0.0875582223356095, -0.22083532022039812]

WEIGHTED_MASS_CONSERVATION_AT_EVENT:
PASS

WEIGHTED_SHARED_FACE_MOMENTUM_CONSERVATION:
PASS

THIN_LAYER_VELOCITY_PATHOLOGY:
NO

DUAL_CV_IMPLEMENTATION_ERROR:
NO

PHYSICALLY_EXPECTED_DUAL_CV_DIVERGENCE:
NO

TOOL_MOBILE_PRECEDES_FIRST_DIVERGENCE:
YES

ROOT_CAUSE_CLASS:
TOOL_MOBILE_IMPULSE

ROOT_CAUSE_OPERATOR:
TOOL_MOBILE_IMPULSE

ROOT_CAUSE_PHYSICS_STEP:
169

ROOT_CAUSE_CELL_OR_FACE:
[y=392, x=108], exact cell record retained in JSON

VISUAL_DIAGNOSTIC:
NOT_RUN

VISUAL_OBSERVATION:
Optional authoritative DEVICE overlay implemented; headless numerical replay
did not render it.

FIX_APPLIED:
NO

PRIMARY_NEXT_FIX:
NONE — audit only
```

## Verification

- Production CUDA replay: completed without crash, output run
  `run_1786960497`.
- Focused regressions: `19 passed, 1 skipped`.
- Python syntax compilation: passed for all touched runtime modules.
- Physics equations and parameters changed: none.

