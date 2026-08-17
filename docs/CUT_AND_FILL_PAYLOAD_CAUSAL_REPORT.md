# CUT_AND_FILL_PAYLOAD_CAUSAL_REPORT

Status: `CAUSAL_ROOT_CAUSE_ESTABLISHED_NO_PHYSICS_FIX_APPLIED`

The first broken link is `MOBILE_NOT_REACHING_MOUTH`. Failure activation is
large and conservative, but only 0.2991% of the activated Mobile crosses the
authoritative mouth control surface. Intake accepts 100% of that candidate
flux and the Payload retains all of it. Capacity, spill, SoilForce saturation,
and mass accounting are not limiting this run.

## Closed transition budget

| Quantity | CUT_AND_FILL total |
|---|---:|
| R2M gross | 4.7818756317 m3 |
| FailureZone R2M | 4.6147201645 m3 |
| LargeAvalanche R2M | 0.1671554672 m3 |
| M2P gross | 0.0143007154 m3 |
| Payload spill | 0 m3 |
| Payload to Airborne | 0 m3 |
| M2R | 0 m3 |
| Mobile to Outflow | 0 m3 |
| Net Payload gain | 0.0143007154 m3 |
| Payload budget residual | -1.735e-18 m3 |
| Final whole-system mass error | -6.821e-12 m3 |

The old 0.167155 m3 R2M value was only the LargeAvalanche cumulative counter.
The corrected total is reconstructed from the conservative Mobile control
volume:

`R2M = DeltaMobile + M2P + M2R + Outflow - PayloadSpill`.

## Causal evidence

- Mouth-prism Mobile: mean 0.002566 m3, p95 0.005285 m3, maximum 0.174148 m3.
- Candidate mouth flux = accepted mouth flux = 0.0143007154 m3.
- Mouth capture ratio = 1.0; capacity-rejected flux and spill are zero.
- Mean positive relative-normal speed on active steps is 0.018782 m/s.
- Mouth intersection area is not missing: mean 1.0982 m2, maximum 3.2030 m2.
- Requested joints are reached; final maximum joint error is 0.000316 rad.
- Tool progress is 1.0171 and maximum penetration is 1.9030 m.
- Peak SoilForce is 288473.46 N, entirely the quasi-static/FEE component in
  this interval. Dynamic Mobile reaction is 0 N, effort saturation occurs on
  2.22% of steps, and shared-power limiting occurs on 0% of steps.
- Payload capacity is 5.6334 m3, leaving 5.6189 m3 unused. There is no
  intake-then-immediate-spill behavior.

Most Mobile is activated in a broad region outside the thin mouth control
volume. Once the commanded CUT motion has effectively completed, continuing
activation/flow has weak positive normal transport through the stationary
mouth. This explains the large global Mobile reservoir and small Payload
without requiring an intake multiplier or retention change.

## Decision

- `PRIMARY_BOTTLENECK = MOBILE_NOT_REACHING_MOUTH`
- Secondary: broad activation outside the mouth control volume; weak positive
  normal transport after command completion.
- `MINIMAL_FIX = NOT_APPLIED`. The evidence does not show a local arithmetic,
  sign, geometry, capacity, retention, or force-coupling implementation defect.
  Changing trajectory or Mobile transport is outside this audit and would not
  be a semantics-preserving local fix.
- `CUT_AND_FILL_AFTER_FIX = NOT_APPLIED`
- `PRE_DUMP_REACHED = NO`
- Morphology classifications = `NOT_EVALUATED`

Machine-readable evidence is in
`outputs/mobile_v2_production/CUT_AND_FILL_PAYLOAD_CAUSAL_REPORT.json`. The raw
360-step audit and CUT start checkpoint remain preserved alongside it.
