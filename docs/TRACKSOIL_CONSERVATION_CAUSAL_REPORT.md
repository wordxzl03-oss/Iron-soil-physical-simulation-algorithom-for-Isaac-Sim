# TrackSoil conservation causal report

## Outcome

The first bad production boundary is **Mobile V2 transport, not TrackSoil**.
The frozen 390F / 701×701 / 0.05 m / GPU_RUNTIME / DEVICE replay reached
PRE_DUMP without changing any physics parameter. No conservation fix was
applied because the causal repair is a Mobile V2 control-volume change, and
Mobile V2 is explicitly frozen in P0-2A.

## First bad ledger boundary

- physics step: `639`
- simulation time: `21.300001110882 s`
- phase: `REVERSE_TRAVEL`
- boundary: `AFTER_MOBILE_TRANSPORT`
- last-good mass error: `3.41060513165e-12 m³`
- first-bad mass error: `-2.29117631534e-08 m³`
- first-bad delta: `-2.29151737585e-08 m³`
- locator threshold: `2.31921148952e-09 m³`

At the immediately preceding TrackSoil boundary, the accepted R→M transfer was
`0.000378 m³`: Resting lost
`0.000378000002911 m³`, Mobile gained
`0.000378000000001 m³`, and the local residual was only
`-2.91044965905e-12 m³`.

At `AFTER_MOBILE_TRANSPORT`, the observed authoritative delta was:

```text
ΔV_R = 1.20508047985e-11 m³
ΔV_M = -2.29272938412e-08 m³
ΔV_P = 0
ΔV_A = 0
ΔV_O = 0
ΔV_total = -2.29152430364e-08 m³
```

The Mobile step itself reports the same unmatched transport residual. Across
all 398 audited post-cut steps, cumulative Mobile transport residual is
`-0.000219067996727 m³`, while final ledger
error is `-0.000219067989065 m³`; their difference is only
`-7.66209318215e-12 m³`.

## Cause

`experimental/mobile_v2_warp.py::_kernels.faces` applies equal-and-opposite
**height** increments to adjacent vertex samples. That conserves an unweighted
sum of heights. The authoritative terrain/material measure instead uses exact
Triangle-A-C vertex control areas from `DeviceBulkState._vertex_weights`.
Interior weights are uniform, but edge/corner weights are smaller. TrackSoil
creates Mobile on the real heightmap boundary (first active footprint: 383
vertices, exact weighted area `0.9375 m²`), so a
Mobile face transfer across unequal weights changes authoritative material.

This is not stale host data, a second terrain mirror, double application, or
reduction noise: `resting` aliases `b_eff` to the same Warp allocation;
TrackSoil reads `z_base`, writes `b_eff` once and `mobile` once; every audited
boundary synchronizes before reduction. Reduction scatter is O(1e-11 m³),
whereas the coherent Mobile residual reaches O(1e-4 m³).

The focused CUDA probe independently reproduces the measure mismatch:
TrackSoil boundary R→M residual is
`-3.33066907388e-16 m³`; Mobile V2
preserves the uniform measure to
`-5.42101086243e-20 m³` but
changes the authoritative Triangle-A-C measure by
`5.97661014753e-05 m³`.
That value equals the solver-reported transport residual exactly in the probe.

## Normalized severity

- absolute final residual: `0.000219067989065 m³`
- ε_initial: `1.62012465546e-07`
- disturbed TrackSoil volume: `0.0252312192304 m³`
- ε_disturbed: `0.00868241788336` (0.868242%)
- payload: `0.0382227840486 m³`
- ε_payload: `0.00573134570171` (0.573135%)

## Fix decision

`FIX_APPLIED = NO`. Fixing the ledger, globally renormalizing terrain, hiding
the residual, or redepositing TrackSoil material directly would be invalid.
The smallest physical fix is a weight-aware/dual-control-volume Mobile V2 face
transfer. That necessarily changes Mobile V2 and therefore is outside the
frozen P0-2A scope. No after-fix replay was run.

## Mandatory conclusions

```text
FIRST_BAD_STEP_FOUND: YES
FIRST_BAD_PHASE: REVERSE_TRAVEL
FIRST_BAD_TIME_S: 21.300001110881567
LAST_GOOD_MASS_ERROR_M3: 3.410605131648481e-12
FIRST_BAD_MASS_ERROR_M3: -2.2911763153388165e-08
FIRST_BAD_DELTA_MASS_ERROR_M3: -2.2915173758519813e-08
TRACKSOIL_ACTIVE_AT_FIRST_BAD_STEP: YES
ROOT_CAUSE_CLASS: PHYSICS_STATE_NONCONSERVATION
ROOT_CAUSE: Production Mobile V2 applies equal-and-opposite height increments at each face, which conserves an unweighted vertex sum but not the authoritative Triangle-A-C control-volume weights at the terrain boundary; TrackSoil exposes Mobile there, and the first Mobile transport step loses weighted material with no outflow.
RESPONSIBLE_OPERATOR: WarpProductionMobileV2Solver
RESPONSIBLE_FILE_FUNCTION: src/isaac_bulk_pipeline/experimental/mobile_v2_warp.py::_kernels.faces / apply_update_and_sources, called by src/isaac_bulk_pipeline/bulk_interaction/warp_mobile_v2.py::WarpProductionMobileV2Solver.step_resident
FIX_APPLIED: NO
FIX_DESCRIPTION: None. A valid repair requires a conservative dual-control-volume/weight-aware Mobile V2 face update; Mobile V2 was explicitly frozen for P0-2A. No TrackSoil or ledger workaround was applied.
PHYSICS_PARAMETERS_CHANGED: NONE
MOBILE_V2_CHANGED: NO
FAILURESURFACE_V3_CHANGED: NO
CURL_SCOOP_TRAJECTORY_CHANGED: NO
BEFORE_MAX_MASS_ERROR_M3: 0.00021906799202042748
AFTER_MAX_MASS_ERROR_M3: NOT_RUN
RELATIVE_ERROR_VS_DISTURBED_MATERIAL_BEFORE: 0.00868241788335676
RELATIVE_ERROR_VS_DISTURBED_MATERIAL_AFTER: NOT_RUN
PRE_DUMP_REACHED_AFTER_FIX: NOT_RUN
VISIBLE_NEEDLE_FOREST: NOT_EVALUATED
PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N: UNCHANGED
PRIMARY_REMAINING_BLOCKER: MOBILE_V2_BOUNDARY_CONTROL_VOLUME_WEIGHT_MISMATCH
```
