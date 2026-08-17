# Iron-ore-fines parameter provenance

## Claim boundary

These values define a literature-derived uncertainty envelope. They are not a
probability distribution, confidence interval, single “typical iron ore” or
calibration of the user's site.

The main source is Mohajeri et al., *Bulk properties variability and
interdependency determination for cohesive iron ore*, Powder Technology 367
(2020) 539--557, DOI `10.1016/j.powtec.2020.04.018`. It reports ring-shear,
wall-friction and ledge tests on three Brazilian pellet/sinter feeds across
moisture and 2/8/20 kPa pre-consolidation. The high friction endpoint also uses
Wang et al., *Geotechnical properties of a type of iron ore fines*, DOI
`10.3208/jgssp.JPN-079`.

Mohajeri's samples are Carajas pellet feed I1 (`d50` about 0.053 mm,
13.3% as-received dry-basis moisture), Carajas sinter feed I2 (`d50` 0.880
mm, 8.7%), and finer-than-0.053 mm Minas-Rio pellet feed I3 (6.8%). Wall
friction was measured against blasted hot-rolled stainless steel. Wang's
separate cargo passed a 9.5 mm sieve and was tested saturated at about 12%
water content with 50--200 kPa effective confinement; this condition is not
treated as interchangeable with Mohajeri's ring-shear states.

## Values passed to force calculations

| Parameter | Low | Reference | High | Exact provenance and limitation |
|---|---:|---:|---:|---|
| Bulk density (kg/m³) | 1370 | 1995 | 2799 | Mohajeri text around Figs. 17/19: loose/as-received Minas-Rio I3 is 1370; loose/as-received Carajas I2 is 1995 and rises to 2799 after shearing at 20 kPa. The endpoints mix material and consolidation state deliberately. |
| Internal friction (deg) | 29.8 | 37.8 | 45.6 | Mohajeri I3 linear yield-locus averages are 29.8, 40.5 and 37.8 degrees at 2, 8 and 20 kPa, max SD 2 degrees. The 45.6-degree endpoint is Wang's undrained Mohr--Coulomb result on a different cargo/test method, so it is a broad sensitivity bound. |
| Cohesion (Pa) | 700 | 800 | 3000 | Mohajeri reports I3 `0.8 +/- 0.1 kPa` at 2 kPa. For I2, moisture changes cohesion by 1.3 kPa, corresponding to 77%; interpreting that stated span gives an upper order of about 3 kPa. The high value is an explicit conservative endpoint, not a measured universal constant. |
| Tool/wall friction angle (deg) | 20.7 | 31.5 | 34.4 | Mohajeri Fig. 20/text: I2 gives 20.7--23.3 degrees at 17.1 kPa; I1 includes 31.5 degrees at high normal stress; I3 averages 34.4 degrees at 1.1 kPa. Wall coupon, stress and moisture dependence are retained as uncertainty. |

The SoilForce input uses `mu_tool=tan(delta)`. All four parameters are varied
together in an unweighted full-factorial `3^4=81` grid. Correlations observed
in the paper are not modeled, so this is a bounding design, not a joint
probability model.

## Flow-angle data not used by SoilForce

Mohajeri's ledge tests mostly give angle of repose from 55 to 70 degrees;
Carajas I2 is about 63 degrees as received and about 58 degrees dry. The
near-liquid non-flowing outlier is excluded. These values are stored only as an
`ANGLE_OF_REPOSE_PROXY_ONLY` for `theta_start`. A distinct dynamic stop angle
was not published for these materials, so `theta_stop` is null. Neither field
is consumed by the current FailureZone/SoilForce acceptance fixture.

Wang reports specific gravity `Gs=4.444`; this is particle specific gravity,
not bulk density, and is therefore not substituted into the model's bulk-
density field.

## Propagated force envelope

The standardized case uses the real 390F descriptor, 0.50 m penetration,
60-degree rake, flat ground, 0.05 m terrain resolution and no dynamic impulse.
The 81-case grid produces:

| Output | Force (N) |
|---|---:|
| Literature-grid minimum | 12,861.32 |
| All-reference case | 34,220.44 |
| Unweighted grid median | 34,532.36 |
| Literature-grid maximum | 93,164.54 |

No case hit the 800 kN safety cap. The min/max are envelope extrema and the
median is only the median of the deterministic grid; none is a percentile or
confidence bound.

Machine-readable source conditions live in
`configs/literature/iron_ore_fines_envelope.json`; propagated results live in
`outputs/soil_force_literature_acceptance.json`.

## Condition-coherent runtime scenario

The integrated 390F runtime does not use the mixed-sample reference tuple from
the broad envelope.  Its soil-force tuple uses I3 throughout: as-received loose
bulk density 1,370 kg/m³, the 2 kPa yield-locus values 29.8 degrees and 0.8 kPa,
and the I3 1.1 kPa wall-friction angle 34.4 degrees.  This is labelled
`SAME_SAMPLE_SAME_MOISTURE_NEAR_MATCHED_LOW_STRESS`, because wall and ring-shear
normal stresses are close but not identical.

The 38/30 degree mobile-layer start/stop angles remain engineering inputs: the
paper's ledge angle is not a measured dynamic start/stop pair for this reduced-
order model.  Therefore the full material scenario is still uncalibrated even
though the SoilForce tuple no longer mixes I2 and I3.  A second I3 20 kPa entry
is retained as `INCOMPLETE_FOR_SIMULATION`; missing same-condition values are
not borrowed from other samples.  Both records are machine readable in
`configs/literature/iron_ore_condition_scenarios.json`.
