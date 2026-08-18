# P0-2D post-integration payload regression causal audit

## Decision

This was a read-only audit of retained production artifacts. No physics,
configuration, threshold, trajectory, CUDA path, or completion condition was
changed.

The first retained divergence is already present at the
`CUT_AND_FILL_START_BEFORE_FIRST_CUT_CORE_STEP` checkpoint. The two runs enter
CUT with effectively identical commanded and measured machine configuration,
but with different local Mobile state and, most importantly, very different
integrated Mobile momentum. The primary supported class is therefore
`C. MOBILE_TRANSPORT_DIVERGENCE`, with causal onset bounded to the preceding
PENETRATE phase. The `tau=-0.2000000104 s` runtime sample has zero Mobile in
both runs, while the before-first-CUT checkpoint at `tau=0` is different;
therefore the onset is bracketed to `-0.2000000104 < tau <= 0`. Retained
telemetry does not identify the exact physics step/operator inside that
interval.

Tool-Mobile contact is correlated with the changed transport and its net
impulse points away from the mouth and laterally. That is not enough to prove
that Tool-Mobile caused the payload regression: no controlled production
ablation isolates it from the concurrent dual-control-volume Mobile change.
The GPU geometry migration itself is not implicated by the available
aggregate evidence, but full support-set equivalence cannot be certified
because the required per-contact geometry was not recorded.

## Frozen sources

- accepted realistic curl-scoop baseline:
  `outputs/390f_v2/interactive_runs/run_1786937073`
- post-integration CUDA run:
  `outputs/390f_v2/interactive_runs/run_1786956921`
- intermediate CPU-exact Tool-Mobile production run:
  `outputs/390f_v2/interactive_runs/run_1786950984`
- baseline CUT-entry checkpoint:
  `outputs/mobile_v2_production/cut_fill_realistic_trajectory_candidate2_audit.cut_start.npz`
- current CUT-entry checkpoint:
  `outputs/mobile_v2_production/p0_2b_p0_2d_production_timeseries.cut_start.npz`
- complete 360-frame aligned comparison and contact records:
  `outputs/mobile_v2_production/p0_2d_post_integration_payload_regression_causal_audit.json`

Both CUT traces start at `5.700000297278166 s`, use the same approximately
30 Hz sampling period, and have a common aligned horizon of
`tau = 0 ... 11.966667290776968 s`. The baseline has 420 retained frames and
the current timed-out CUT has 360; the machine-readable artifact compares all
360 common samples. At `tau=2.633333470672369 s`, the baseline has already met
the payload gate and transitioned to `CURL_AND_BREAKOUT`, while the current run
remains in CUT. Values after that boundary are retained as downstream outcome
comparisons, not as evidence of the first divergence.

## First divergence

At the checkpoint immediately before the first CUT core step:

| Quantity | Baseline | Current |
|---|---:|---:|
| Joint-position norm difference | - | `1.32231e-6 rad` |
| Joint-velocity norm difference | - | `1.32936e-5 rad/s` |
| Requested-target norm difference | - | `1.32038e-6 rad` |
| Mouth-position difference | - | `4.60382e-6 m` |
| Mobile volume | `0.004632544601 m3` | `0.004676274107 m3` |
| Integrated Mobile momentum x | `-0.03333473746 kg m/s` | `-0.7655750134 kg m/s` |
| Integrated Mobile momentum y | `-0.01299070556 kg m/s` | `-1.370145186 kg m/s` |

The authoritative `H_free = b_eff + mobile` is also already locally different:

- affected support: 63 vertices, index box `y=391..401, x=102..112`, or
  terrain-local `x=5.10..5.60 m, y=19.55..20.05 m`;
- maximum absolute difference: `0.01691696549 m`;
- net volume difference: `+2.918400237e-5 m3`;
- weighted absolute difference: `2.324257640e-4 m3`.

That local free-surface difference accounts for the first-frame penetration
difference despite nearly identical tool pose: baseline `0.2318950741 m`,
current `0.2488129274 m`. The first CUT step then produces:

| Quantity at `tau=0` | Baseline | Current |
|---|---:|---:|
| FailureZone R2M step | `0.01176455182 m3` | `0.01172514263 m3` |
| Mobile total | `0.01612622659 m3` | `0.01625246425 m3` |
| candidate/admitted flux | `0.0002708698370 m3` | `0.0001489524894 m3` |
| payload after step | `0.0004674018032 m3` | `0.0003163004532 m3` |
| Tool-Mobile accepted cells | `0` | `55` |
| dynamic reaction-force norm | `0 N` | `249.320204 N` |

The candidate flux is already 45.0% lower on the first CUT frame. Intake
accepts every valid candidate in both runs, so this is upstream of intake.

Command targets remain effectively identical. Over the first 17 CUT frames,
the maximum requested-target difference is `1.32038e-6 rad`, while actual
joint position first exceeds a `1e-3 rad` between-run difference at
`tau=2.033333439 s`. This later actual-motion divergence is a downstream
response to different soil/Mobile dynamics, not an independently changed
trajectory command.

## Downstream aligned evidence

| `tau` | FailureZone R2M cumulative baseline/current | Mobile baseline/current | payload baseline/current |
|---:|---:|---:|---:|
| `0 s` | `0.011765 / 0.011725 m3` | `0.016126 / 0.016252 m3` | `0.000467 / 0.000316 m3` |
| `2 s` | `0.633112 / 0.428735 m3` | `0.630275 / 0.429726 m3` | `0.007666 / 0.003853 m3` |
| `2.633 s` | not used for first-cause attribution | `0.930331 / 0.709851 m3` | `0.021585 / 0.009987 m3` |
| `4 s` | `1.001389 / 0.766617 m3` | `0.981795 / 0.766818 m3` | `0.038223 / 0.016843 m3` |

The current FailureZone activation outcome is lower by 23.45% over the run,
even though current maximum penetration is higher (`0.7796001523 m` versus
`0.6560070643 m`). Thus a simple loss of geometric penetration does not
explain the regression. Two distinct diagnostics must not be conflated:

- the higher-rate CUT audit's `failure_zone_r2m_step_m3` is conservative R2M
  activation transfer (`1.0013890609 m3` baseline versus
  `0.7666167798 m3` current);
- runtime `material_funnel.failure_volume_m3` is the retained cumulative
  FailureSurface/candidate-volume diagnostic (`0.7246961809 m3` baseline
  versus `0.4653015955 m3` current).

Both change downstream. Neither is the first retained divergence because the
Mobile/momentum checkpoint is already different before the first CUT step.

Final candidate flux, accepted flux, and payload gain are the same in each
run:

- baseline: `0.03802625208 m3`;
- current: `0.01667549367 m3`;
- capacity rejected: `0 m3` in both.

Therefore `INTAKE_REJECTED_VALID_MOUTH_FLUX = NO`.

## Tool-Mobile audit

The current run's first geometry-confirmed contact is the first CUT frame at
`tau=0`: 55 candidate cells and `0.01247635258 m3` geometry-confirmed Mobile.
Contact is active over three intervals:

1. `0.000000000 ... 1.133333392 s` (35 frames)
2. `1.500000078 ... 3.200000167 s` (52 frames)
3. `3.266666837 ... 3.366666842 s` (4 frames)

Recorded CUT totals are:

- 91 contact-active frames, 454 active substeps, 7,823 active
  cell-substeps;
- normal impulse `625.4351978 N s`;
- tangential impulse `371.5789402 N s`;
- Mobile impulse `[-91.57903674, -479.6981900, 0] N s`;
- equal-and-opposite machine reaction
  `[91.57903674, 479.6981900, 0] N s`;
- action-reaction residual exactly zero;
- tool-to-Mobile work `439.1600499 J`;
- frictional dissipation `88.00447411 J`;
- total contact dissipation `132.8516602 J`;
- unexplained contact-energy residual `1.81879e-15 J`.

Projecting every frame's impulse onto its recorded horizontal mouth normal
(positive toward mouth) gives signed normal impulse `-252.3126294 N s` and
signed lateral impulse `-433.9922404 N s`. The current coupling therefore has
a net away-from-mouth and lateral redirection. This is a physically relevant
hypothesis for reduced mouth flux, but the retained runs do not isolate it
causally from the Mobile transport/control-volume integration change.
Accordingly `TOOL_MOBILE_CAUSALLY_RESPONSIBLE = NOT_DEMONSTRATED`.

## GPU geometry migration equivalence

The intermediate real production run `run_1786950984` used the CPU-exact
contact geometry and shares the current post-integration state. Its first 17
CUT frames match the current GPU run as follows:

- accepted candidate-count difference: exactly `0` for every frame;
- maximum geometry-confirmed Mobile-volume difference: `5.20417e-17 m3`;
- maximum normal-impulse difference: `1.86517e-13 N s`;
- maximum tangential-impulse difference: `1.13687e-13 N s`;
- maximum Mobile-impulse-vector difference: `1.95642e-13 N s`;
- maximum dynamic-force difference: `1.16854e-11 N`.

This strongly rejects the GPU migration as the source of the aggregate
production regression. It is not the requested full equivalence proof:
accepted flat indices, closest points, normals, surface velocities, and signed
distances were not retained by either run, and the per-frame full device state
needed to reconstruct them offline was not checkpointed. The formal result is
therefore `GPU_GEOMETRY_PRODUCTION_EQUIVALENCE = NOT_ENOUGH_EVIDENCE`, not
`PASS`.

## Completion gate

The production state machine requires both cut distance and payload gain. The
current run measured:

- cut distance `6.608409154 m >= 0.25 m`: **passed**;
- payload gain `0.01667549367 m3 < 0.02 m3`: **failed**, short by
  `0.003324506334 m3`.

The timeout was caused specifically by the unchanged minimum-payload-gain
clause. No other CUT completion clause remained unmet.

## Required contract

```text
BASELINE_RUN:
run_1786937073

CURRENT_RUN:
run_1786956921

BASELINE_PAYLOAD_M3:
0.03802625208251054

CURRENT_PAYLOAD_M3:
0.016842841629717515

CUT_COMPLETION_FAILED_CLAUSE:
minimum_payload_gain_m3 (0.01667549366595567 < 0.02); cut distance passed

FIRST_DIVERGENCE_TAU_S:
0.0 first differential checkpoint; onset bracket is -0.2000000104 < tau <= 0

FIRST_DIVERGENCE_VARIABLE:
DeviceBulkState Mobile momentum_x/momentum_y, with local Mobile/H_free differences

BASELINE_VALUE:
integrated Mobile momentum = [-0.03333473746, -0.01299070556] kg m/s

CURRENT_VALUE:
integrated Mobile momentum = [-0.7655750134, -1.370145186] kg m/s

ROOT_CAUSE_CLASS:
C. MOBILE_TRANSPORT_DIVERGENCE

TOOL_MOBILE_CAUSALLY_RESPONSIBLE:
NOT_DEMONSTRATED

GPU_GEOMETRY_PRODUCTION_EQUIVALENCE:
NOT_ENOUGH_EVIDENCE

TRAJECTORY_CHANGED:
NO (commanded trajectory; later actual response divergence is downstream)

FAILURESURFACE_CHANGED:
YES (activation result; geometric FailureSurface volume was not retained)

MOBILE_TRANSPORT_CHANGED:
YES

INTAKE_REJECTED_VALID_MOUTH_FLUX:
NO

FIX_APPLIED:
NO
```

Audit stops here. No fix or parameter change was applied.
