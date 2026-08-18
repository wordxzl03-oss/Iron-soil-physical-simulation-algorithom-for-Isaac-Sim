# Mobile V2 dual-control-volume conservation fix

## Status

The P0-2B code fix is implemented. Canonical algebra, the real Warp kernel on
the CPU device, all accepted Mobile V2 reference contracts, frozen 5 s, and the
full Python regression suite have been run. Formal CUDA and the unchanged
390F/GPU_RUNTIME/DEVICE production replay are not reported as passes because
the current Codex sandbox exposes no NVIDIA device (`nvidia-smi` cannot reach
the driver and Warp reports `CUDA devices not available`).

Current formal status:

`IMPLEMENTED_AWAITING_CUDA_PRODUCTION_REPLAY`

No physical parameter, constitutive model, FailureSurface V3 behavior,
TrackSoil behavior, intake, payload, Airborne, deposition, LargeAvalanche,
trajectory, controller, grid, or timestep was changed.

## Root cause and old update

`experimental/mobile_v2_warp.py::_kernels.faces` constructs one hydrostatically
reconstructed two-state Rusanov flux. For an oriented x-face from `i` to `j`:

- `F_h` has units m²/s and is volume flux per metre of face;
- `F_qx` and `F_qy` have units m³/s² and are momentum-measure fluxes per metre
  of face;
- face length, `dt`, and division by a control area were not yet included in
  those fluxes;
- positive flux means removal from `i` and addition to `j`;
- x-face length is `dy`; y-face length is `dx`.

The old square-grid update was

```text
Δh_i = -(dt / spacing) F_h
Δh_j = +(dt / spacing) F_h
```

and identically for the shared momentum flux, with the existing one-sided
hydrostatic topography correction on the corresponding endpoint. This is
equivalent to using `A_i=A_j=dx*dy`. It conserves `sum(h)` but not the
authoritative Triangle-A-C measure `sum(A_i h_i)` across unequal-area faces.

## New shared-transfer update

For every internal face the kernel now evaluates the numerical flux once and
forms one integrated transfer:

```text
ΔV_face = F_h * L_face * dt
ΔP_face = F_q * L_face * dt

Δh_i = -ΔV_face / A_i
Δh_j = +ΔV_face / A_j

Δq_i = -ΔP_face / A_i
Δq_j = +ΔP_face / A_j
```

`A_i` and `A_j` are the already-persistent DEVICE `weights` buffer created by
`DeviceBulkState._vertex_weights`. No host round-trip, second one-sided flux,
global correction, renormalization, grid padding, boundary suppression, or
fake outflow was introduced.

The physical contracts are now explicitly:

```text
MobileVolume(h)   = sum_i A_i h_i
MobileMomentum(q) = density * sum_i A_i q_i
```

The helper in `experimental/mobile_v2_control_volume.py` is the executable CPU
form of the same contract used by canonical tests.

## Dimensional and source-term audit

| Term | Kernel quantity before endpoint update | Endpoint operation | Weight treatment |
|---|---|---|---|
| mass face divergence | `F_h [m²/s]` | `F_h L dt / A [m]` | divide by the receiving/donating endpoint area once |
| shared momentum face divergence | `F_q [m³/s²]` | `F_q L dt / A [m²/s]` | divide by endpoint area once |
| hydrostatic reconstruction/topography correction | momentum-flux correction `[m³/s²]` | same face integration | retains accepted left/right well-balanced correction; divide by endpoint area once |
| external acceleration | `h a dt [m²/s]` | local state-rate source | no division by area; diagnostic impulse multiplies by `A` once |
| Coulomb friction | velocity reduction, then `q=h u` | local state-rate source | no division by area; diagnostic impulse/work multiplies by `A` once |
| R→M / M→R | separate conservative DEVICE operators | unchanged | already uses authoritative weights; not touched |

The production adapter previously summed diagnostic `q`/energy changes without
weights and multiplied by uniform `dx*dy` afterward. Diagnostics now integrate
each local term with its resident `A_i`; the later uniform-area multiplication
was removed. This changes accounting geometry, not source physics.

## CFL and positivity

The accepted simulation `dt` and `cfl=0.18` remain unchanged. Adaptive Mobile
substeps now use the local metric rate

```text
(|u| + sqrt(K g h)) * L_face / A_i
```

and `dt_sub <= cfl / max(metric_rate)`. For equal-area interior vertices this
reduces exactly to the legacy `cfl*min(dx,dy)/wave` restriction. Smaller
Triangle-A-C boundary/corner control volumes therefore receive the physically
required smaller Mobile substep.

The apply kernel records any negative raw depth as an authoritative weighted
negative-volume diagnostic. A violation beyond a floating-point-derived
safety tolerance raises an error; clipping is not accepted as a mass-fix path.
The closed physical boundary still has no exterior face and no declared
outflow.

## Validation completed in this environment

- focused dual-CV pytest: 9 passed;
- real Warp face kernel, both directions: weighted mass residual exactly 0;
- real Warp face kernel weighted momentum maximum residual:
  `3.39e-21 m4/s`;
- real Triangle-A-C edge/corner tests: exact-zero recorded residuals;
- random closed boundary, Mobile touching all boundaries, 40 production steps
  and 113 adaptive substeps: `-2.08e-17 m³` weighted residual, positive minimum
  depth `0.0352581 m`;
- equal-area Warp versus CPU: maximum height error `1.11e-16 m`, maximum
  momentum-state error `1.67e-16 m²/s`;
- flat, equilibrium slope, valley, mound, wet/dry, bookkeeping, entrainment,
  deposition: PASS;
- frozen continuous 5 s: mass max `1.78e-15 m³`, positivity PASS, energy PASS;
- full suite: 290 passed, 16 skipped, 57 subtests; the same three unrelated
  pre-existing failures remain (two stale Phase-A manifest checks and the
  Phase-H width monotonicity assertion).

CPU Warp is reported only as kernel compilation/equation evidence. It is not
relabeled as `GPU_DEVICE_TEST=PASS`.

## Formal replay path and remaining gate

The runner now has a dedicated `--mobile-v2-dual-cv-audit PATH` output. It
reuses the accepted scalar operator-boundary observer but never overwrites the
frozen P0-2A causal artifacts. The exact pending formal command is recorded in
`mobile_v2_dual_cv_production_replay.json`.

Once CUDA is exposed, that one command produces the unchanged normal prelude,
coordinated curl-scoop, reverse, and PRE_DUMP fields plus the raw post-cut
operator ledger. `tools/finalize_mobile_v2_dual_cv_fix.py` then computes the
required after mass/residual, payload/capture/force, morphology-only record,
and old-versus-new field dynamics. Missing CUDA results remain JSON `null`;
the finalizer never substitutes theoretical values for measurements.

## Artifacts

- `outputs/mobile_v2_production/mobile_v2_dual_cv_fix_report.json`
- `outputs/mobile_v2_production/mobile_v2_dual_cv_canonical_tests.json`
- `outputs/mobile_v2_production/mobile_v2_dual_cv_production_replay.json`
- `outputs/mobile_v2_production/mobile_v2_dual_cv_before_after_metrics.json`

Primary remaining blocker for formal P0-2B closure:

`CUDA_DEVICE_NOT_EXPOSED_TO_CODEX_SANDBOX_FOR_FORMAL_390F_REPLAY`
