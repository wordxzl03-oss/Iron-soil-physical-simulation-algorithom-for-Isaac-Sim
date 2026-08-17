# V3 Flow / Arrest final causal fix

## Result

The frozen production dump checkpoint now reaches `ARREST_FINAL` at 1.35 s and remains there through the 5 s observation. The prior build produced 1,793 same-region Resting→Mobile triggers in 30 s and did not settle. The repaired build produces zero triggers after this checkpoint. No material parameter, grid spacing, time step, force-settle rule, cooldown, or smoothing was changed.

## Root cause

The previous latch could be released from Eulerian `H_free` movement. That did not establish that the activated tranche had left its control volume: local Mobile→Resting deposition and simultaneous inflow/outflow can change little—or change `H_free` numerically—while ownership remains local. The same 0.0825 m² region was consequently exposed every time step. R2M and M2R both accumulated at about 0.396 m³/s while only 0.0066 m³ remained Mobile.

## Ownership contract

Each activation records an owned tranche and its cumulative-export baseline. The Warp Mobile donor-limited flux operation now accumulates only actual conservative Mobile export crossing from the owned donor control volume into an unowned receiver. The latch becomes eligible for release when that exported volume exceeds the existing discretization dry-volume resolution (`dry_tolerance_m × exact vertex area`). `H_free` is retained only as secondary geometry captured at activation; it is not a release signal.

Local redeposition alone therefore does not clear ownership. A physically stable and quiet region may close its completed event, but a still-yielded locally redeposited tranche remains owned. There is no time cooldown and no new constitutive epsilon.

CPU reference and optimized wrappers expose the same donor-limited conservative export field. Device checkpoint state now includes owned surface, owned-export baseline, and cumulative Mobile export; legacy checkpoints are restored compatibly.

## Required contracts

- A: activation → local redeposition → no identical immediate retrigger: PASS.
- B: activation → actual donor-limited transport export → newly exposed yielded Resting can retrigger: PASS.
- Focused solver/runtime regression: 21 passed.
- Y_start/Y_stop remain distinct; no threshold changed.
- Physical LargeAvalanche owns Y_start before residual MiniSlope; residual iterations in the final replay are zero.

## Frozen replay

| Metric | Before | After |
|---|---:|---:|
| Observation | 30 s | 5 s |
| Retrigger count | 1,793 | 0 |
| Cumulative R2M | 13.0308579272 m³ | 1.1970579272 m³ |
| Cumulative M2R | 14.4164196759 m³ | 2.5892196759 m³ |
| Final Mobile | 0.0066 m³ | 0 m³ |
| Final moving Mobile | 0.0066 m³ | 0 m³ |
| ARREST_FINAL | not reached | 1.35 s |
| Maximum mass error | ~2.41e-11 m³ | 2.09e-11 m³ |

The after totals include pre-checkpoint cumulative history. R2M does not increase after restore; the remaining 0.1058299256 m³ M2R increment is the already airborne dump landing. Raw final Y_start contains 49 cells in seven components, but eligible mobilizable volume is exactly zero because these cells refer to the owned tranche, not new terrain. This allows settled-state closure without ignoring genuinely eligible Y_start.

`H_free` gains 0.1058299256 m³ from checkpoint to final, localized to three deposition scatter cells. Its 14.1107 m point maximum is an existing reduced-order parcel scatter morphology limitation, not an ownership-loop symptom and was not tuned in this fix.

Machine-readable evidence: `outputs/v3_flow_arrest_final/summary.json`.
