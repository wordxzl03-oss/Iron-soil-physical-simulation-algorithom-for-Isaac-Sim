# Physics Core V3 Closure Audit

Scope: `PHYSICS_CORE_V3_CLOSURE_AUDIT` only. This audit does not reuse the
broad structural-validation pass as evidence of production morphology, does
not tune material parameters, and does not change production physics.

## Frozen replay stage semantics

- T0: pre-contact `H_resting`, zero `h_mobile`, and zero mobile momentum.
- T1: intersection diagnostic. Terrain reservoirs are still T0; the saved
  additions are intersection support and penetration depth.
- T2: immediately after conservative FailureZone Resting-to-Mobile activation
  and before Mobile transport. `H_free = H_resting + h_mobile` is unchanged by
  this state-label transfer.
- T3: after one Mobile physical step of `1/60 s`.

## Metric definitions and attribution

All five values in `old_vs_v3_morphology_summary.json` are T2 diagnostics.
They are not interchangeable with published free-surface morphology.

| Metric | Source field | Mathematical definition | ROI | Quantity measured |
|---|---|---|---|---|
| `boundary_serration_mean_step_m` | `FailureZone.active_mask`, derived from positive T2 failure active thickness | For every active lateral row, `front[y] = origin_x + max(active_column[y])*dx`; report `mean(abs(diff(front, n=2)))` over the finite row sequence | All lateral rows containing at least one active cell | Failure support boundary, not `H_free` |
| `shallow_ring_fraction` | T2 `failure active thickness` | `mean(t_i < 0.2*p95(t))` for every `t_i > 0` | Positive failure-thickness support only | Resting-to-Mobile state-label thickness, not a geometric ring test and not `H_free` |
| `local_protrusion_max_over_p95` | T2 `failure active thickness` | `max(t)/p95(t)` for every `t > 0` | Positive failure-thickness support only | Peak state-label activation relative to its thickness distribution, not a free-surface protrusion |
| `terminal_cat_ear_ratio` | T2 `failure active thickness` | Mean of the complete first/last active rows divided by the mean of the complete middle active row | First, last, and middle rows of lateral failure support, including zero cells in each complete row | Terminal failure activation, not `H_free` |
| `forward_momentum_outside_contact_fraction` | T2 positive x component of `mobile_momentum_m2_s` | Area-weighted integral of `max(momentum_x,0)` outside T1 contact divided by the same integral over the whole grid | Full grid, partitioned by T1 intersection support | Forward Mobile momentum outside contact |

The reported changes therefore remain exactly:

- boundary serration: `0.0326530612 -> 0.0204081633` (37.5% lower);
- terminal cat-ear ratio: `0.103329118 -> 0.000117913`;
- forward momentum outside contact: `0.676916865 -> 0`;
- shallow-ring fraction: `0.245119306 -> 0.266509434` (worse);
- local protrusion max/p95: `1.128881336 -> 1.154859340` (slightly worse).

The last two are not declared resolved by their original metrics. In both OLD
and V3, T2 `H_free - T0_H_free` has maximum absolute magnitude
`1.1102230246251565e-16 m` and zero cells above `1e-12 m`. V3 T3, after real
Mobile transport, has a bounded `H_free` change of `[-0.0058698131,
+0.0048947252] m`; it is no longer the T2 activation metric. The shallow-ring
result is classified:

`STATE_LABEL_ARTIFACT_NOT_PHYSICAL_FREE_SURFACE_DEFECT`.

No physics was changed to optimize either label-space ratio.

## Maximum V3 active-thickness attribution

The maximum is at grid `(y=83, x=81)`, terrain coordinate `(x=0.05 m,
y=0.15 m)`, with T2 active thickness `0.2562227477 m`. It is the conservative
sum of two adjacent 0.05 m quadrature slices at their shared vertex:

| Slice | Lateral bounds (m) | d (m) | beta0 (deg) | beta (deg) | L (m) | transition | coverage at max | activated contribution (m3) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 27 | `[0.10, 0.15]` | 0.2848389430 | 30.6623787 | 30.6616019 | 0.4804566382 | 1.0 | 0.5 | 0.0003189955245 |
| 28 | `[0.15, 0.20]` | 0.2868942409 | 30.6646966 | 30.6638417 | 0.4838803222 | 1.0 | 0.5 | 0.0003215613447 |

The surrounding final penetration samples are smooth:

| Slice | d (m) | beta0 (deg) | beta (deg) | L (m) | transition |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.2699984579 | 30.6452704 | 30.6449986 | 0.4557252092 | 1.0 |
| 26 | 0.2788014174 | 30.6555470 | 30.6549611 | 0.4703970235 | 1.0 |
| 27 | 0.2848389430 | 30.6623787 | 30.6616019 | 0.4804566382 | 1.0 |
| 28 | 0.2868942409 | 30.6646966 | 30.6638417 | 0.4838803222 | 1.0 |
| 29 | 0.2845118482 | 30.6620285 | 30.6612300 | 0.4799120066 | 1.0 |
| 30 | 0.2780760108 | 30.6547107 | 30.6540898 | 0.4691893759 | 1.0 |

Cause classification:

`FAILURE_ACTIVE_THICKNESS_QUADRATURE_METRIC_ARTIFACT_NOT_PHYSICAL_FREE_SURFACE_DEFECT`.

The ratio increases because a smooth peak and conservative shared-vertex
quadrature change the max/p95 normalization. It is not a discontinuity in
`d(s)`, `beta(s)`, or `L(s)`, and it creates no T2 `H_free` displacement.
Consequently FailureSurface was not modified and no smoothing was added.

## Activation-volume change

| Statistic | V2 legacy strips | V3 continuous 2.5D |
|---|---:|---:|
| Active volume (m3) | 0.1260511265 | 0.1101229187 |
| Active cells | 461 | 424 |
| Penetration mean / median / max (m) | 0.2339399444 / 0.2454084242 / 0.2893505434 | 0.2159813442 / 0.2439257758 / 0.2868942409 |
| Longitudinal extent mean / median / max (m) | 0.3954306257 / 0.4147244843 / 0.4879537672 | 0.3651962861 / 0.4122363848 / 0.4838803222 |
| Failure support x range (m) | 0.00 to 0.50 | 0.00 to 0.50 |
| Failure support y range (m) | -1.25 to 1.25 | -1.25 to 1.25 |
| Total FEE resistance (N) | 8286.864826 | 7321.654437 |

Active volume is 12.6363% lower and FEE resistance is 11.6475% lower. The
median penetration and maximum spatial extents are nearly unchanged, while the
mean depth/extent and terminal support are reduced. This is consistent with
removal of legacy per-strip maximum-depth over-activation. A single synthetic
frozen replay cannot establish systematic under-activation, and V2 is not a
ground truth; V3 was not tuned to recover V2 volume.

## One production Isaac excavation

Exactly one run was made with the production launcher and production Core:

`run_390f_v2.py --config configs/390f_v2_interactive.yaml --headless
--acceptance-cycles 1 --runtime-backend GPU_RUNTIME --v3-closure-audit`

Run directory: `outputs/390f_v2/interactive_runs/run_1786528807`.

Configuration was 390F, 701x701, `dx=0.05 m`, GPU_RUNTIME, DEVICE authority,
the frozen reference material, and the existing production trajectory. No
presentation behavior or physics override was enabled.

| Observation | sim time (s) | payload (m3) | soil force norm (N) | Mobile / moving Mobile (m3) | terrain state / reason | dynamic flow time (s) | residual iterations | mass error (m3) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| pre-dig | 2.3333 | 0 | 0 | 0 / 0 | SETTLED / none | 0 | 0 | 5.00e-12 |
| post-cut | 13.1000 | 0.0200868 | 515.23 | 3.30200 / 3.30200 | diagnostic label `QUASI_STATIC_RESIDUAL_PROJECTION` / DEVICE_DYNAMIC_ACTIVITY | 3.7333 | 0 | 8.87e-12 |
| post-breakout | 19.6667 | 0.335270 | 0 | 6.18832 / 6.18832 | same label / DEVICE_DYNAMIC_ACTIVITY | 7.0167 | 0 | 8.64e-12 |
| early post-dig | 20.1667 | 0.335270 | 0 | 5.23815 / 5.23815 | same label / DEVICE_DYNAMIC_ACTIVITY | 7.2833 | 0 | -3.80e-11 |
| last arrest observation | 146.5000 | approximately 0 | 0 | 0.000179839 / not captured separately | LOCAL_STATIC_INSTABILITY / DEVICE_DYNAMIC_ACTIVITY | not emitted in periodic record | 0; residual pending on 4 tiles | -5.23e-12 |

The cumulative applied soil-force peak was `111384.19 N` over 248 nonzero
steps; peak payload was `0.335270136 m3`; peak Mobile was `6.188319296 m3`;
maximum absolute ledger error was `3.43334e-11 m3`. The production `H_free`
volume loss at post-cut and post-breakout equals the payload volume (within
roundoff), as expected.

Production morphology checks use `H_free`, not the T2 label field. At post-cut,
the one-cell annulus around the `delta H_free < -0.02 m` excavation core has
only 6 of 178 cells in the shallow-negative `[-0.005,-0.0005] m` interval; they
do not form a connected ring. All `delta H_free > 0.002 m` cells form one broad
transported-material component (3812 cells), not an isolated local protrusion.
No paired terminal cat-ear depression is present. Momentum activation remains
restricted to contact; the production morphology contains transported Mobile,
not a whole-wedge assignment of tool velocity.

The process had no recorded Python/CUDA/PhysX/runtime failure, but it ended
before an explicit settled `ARREST_FINAL` terrain array and closure summary
were written. The last scalar observation still had a tiny Mobile reservoir,
2237 unstable cells (largest connected region 2162 cells / 5.405 m2), and a
pending four-tile residual projection. Therefore reasonable-time physical
arrest is **not yet demonstrated**. This is not converted into a pass and the
early post-dig terrain is not relabeled as final.

Residual time semantics are correct in the captured evidence: dynamic Mobile
advances in simulation seconds; residual projection remained at zero
iterations during the physical flow and was only pending after Mobile became
small. Residual iterations were never accumulated as fake physical seconds.
The diagnostic terrain-state label prioritizes residual-pending even while its
reason says DEVICE_DYNAMIC_ACTIVITY; that is an observability naming
inconsistency, not evidence that residual iterations advanced physical time.

## Closure answers

```text
CAT_EAR_IN_PRODUCTION: NO
SHALLOW_RING_IN_H_FREE: NO
LOCAL_PROTRUSION_IN_H_FREE: NO
ARTIFICIAL_WHOLE_WEDGE_FORWARD_LAUNCH: NO
MOBILE_ARRESTS_IN_REASONABLE_PHYSICAL_TIME: NOT_YET_DEMONSTRATED
RESIDUAL_SOLVER_TIME_SEMANTICS_CORRECT: YES
PAYLOAD_REGRESSION: NO
SOIL_FORCE_REGRESSION: NO
MASS_LEDGER_REGRESSION: NO
```

Final status:

`V3_NEEDS_FLOW_ARREST_FIX`

No targeted FailureSurface fix is justified: neither worse label-space metric
is demonstrated as a defect in authoritative `H_free`. Freeze is instead
blocked because the only production excavation did not reach and persist in a
physically settled state before its final observation. The tiny remaining
Mobile reservoir, connected static instability, and pending residual solve
must be closed through the existing flow/arrest hierarchy before V3 can be
frozen; the early post-dig state is not accepted as final.
