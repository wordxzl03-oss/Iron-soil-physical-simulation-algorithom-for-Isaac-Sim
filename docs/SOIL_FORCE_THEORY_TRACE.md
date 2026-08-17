# SoilForce theory trace

## Claim and scope

This is a literature-constrained, conservative reduced-order soil--bucket force
model. It is neither field-calibrated nor a complete FEE/DEM/continuum solve.
It consumes the already accepted `FailureStripGeometry`; it does not solve a
second failure wedge or modify its rasterization.

Primary sources are Luengo, Singh & Cannon (1998), DOI
`10.1109/IROS.1998.724873`; McKyes & Ali (1977), DOI
`10.1016/0022-4898(77)90001-5`; Zhang & Kushwaha (1995), DOI
`10.1016/0167-1987(95)00466-6`; and Servin, Berglund & Nystedt (2021), DOI
`10.1186/s40323-021-00196-3`.

## Quasi-static backbone

For each activated strip, the implemented magnitude is

```text
F = rho g d^2 w Nw + c d w Nc + q d w Nq
```

with the sloped-ground factors from Luengo equations (5)--(6):

```text
D  = cos(rho_rake + delta) + sin(rho_rake + delta) cot(beta + phi)
Nw = (cot(beta) - tan(alpha))
     (cos(alpha) + sin(alpha) cot(beta + phi)) / (2 D)
Nc = (1 + cot(beta) cot(beta + phi)) / D
Nq = (cos(alpha) + sin(alpha) cot(beta + phi)) / D
```

The exact `Nw/Nc/Nq`, weight, cohesion and surcharge values are calculated in
`FailureZoneModel._solve_strip()` and stored in `FailureStripGeometry`.
`SoilForceModel._validate_shared_geometry()` rejects parameter drift; it then
scales those same terms only by the conservatively available activation ratio.

| Term | Classification | Direct source | Implementation deviation |
|---|---|---|---|
| `Nw`, `Nc`, `Nq` | `DIRECT_LITERATURE_MODEL` | Luengo (1998), eqs. (5)--(6) | Singular/non-physical angle domains are rejected. |
| Weight `rho g d²wNw` | `DIRECT_LITERATURE_MODEL` | Luengo/Reece FEE | Uses constant scenario bulk density; no compaction/dilation law. |
| Cohesion `cdwNc` | `DIRECT_LITERATURE_MODEL` | Luengo/Reece FEE | The supplied cohesion is an uncalibrated scenario value. Adhesion is omitted. |
| Pressure surcharge `qdwNq` | `LITERATURE_BASED_REDUCED_ORDER` | Classical pressure form | Luengo's excavator reformulation uses swept volume `Vs`; current state has no validated swept surcharge history, so it is not claimed as the complete Luengo reformulation. Benchmark uses `q=0`. |
| Critical `beta` | `LITERATURE_BASED_REDUCED_ORDER` | Trial-wedge minimum resistance in McKyes & Ali; total-resistance minimization in Zhang & Kushwaha | Luengo identifies `beta` from force data. Here it is closed online by bounded minimization of the above resistance and boundary hits remain diagnostic. |
| Strip availability scaling | `CONSERVATION_BASED_ENGINEERING_MODEL` | finite available mass | Avoids applying force for material clipped by terrain/domain availability. |

Luengo explicitly treats inertia as negligible for low-acceleration digs. The
old repository expression `0.5 * inertial_coefficient * rho * w * d * v²` and
its `inertial_coefficient=0.65` are removed. No replacement gain is hidden in
the force path.

## Force direction and contact resultant

Luengo equation (2) resolves the blade reaction using rake plus soil--tool
friction. For a strip, the implementation uses

```text
F_t = F abs(sin(rho_rake + delta))
F_n = F cos(rho_rake + delta)
vector = -F_t cutting_tangent + F_n bottom_plate_normal
```

This resolution is `LITERATURE_BASED_REDUCED_ORDER`. Its local orthonormal
basis comes from the real cutting-edge direction, CAD bottom-plate normal and
the measured tool velocity (with the failure-zone approach direction used only
at near-zero speed). It is no longer based solely on terrain tangent/normal.

Each strip is applied at its effective cutting-edge contact point, not the soil
wedge centroid. Forces and moments are summed as

```text
F_total = sum(F_i)
tau_origin = sum((r_i - r_origin) cross F_i)
```

The equivalent point is force-magnitude weighted and the adapter also applies
the residual couple needed to reproduce `tau_origin` exactly. This pressure-
resultant closure is an `ENGINEERING_APPROXIMATION`; no cited paper provides a
validated pressure distribution for this 390F bucket.

## Active-soil momentum reaction

Servin et al. represent active/agitated soil with mass, momentum and
frictional-cohesive interaction with equipment. The present mobile layer is not
their multiscale solver, but it now exposes the actual SI impulse chain:

```text
P = rho sum_i (h_i A_i u_i)
Delta P_mobile = J_gravity/pressure + J_basal + J_tool + J_numerical
J_soil_on_tool = -J_tool_on_mobile
F_active_on_tool = J_soil_on_tool / Delta t
```

`J_tool` includes the momentum used to activate resting material at cutting-
edge velocity and the subsequent explicit tool-forcing velocity update.
Gravity/pressure, basal friction and numerical/advection/capping residuals are
reported separately. `MobileMomentumBudget` provides both the balance residual
and the exact action--reaction residual. This term is classified
`CONSERVATION_BASED_ENGINEERING_MODEL`, not a direct implementation of Servin.

The active momentum reaction is applied at the real cutting-edge centre. Its
force and torque are combined with the static strip resultants. No claim is
made that all mobile-soil momentum originates at that single point.

## Runtime safety cap

`maximum_resultant_force_n` is an explicit numerical safety cap. It is not a
physical correction or calibration parameter. If it triggers, force, torque,
static and dynamic components are scaled together and `force_was_limited=true`.
Any capped external benchmark is ineligible for a physical pass. The current
published benchmark and all 81 iron-ore envelope cases have zero cap hits.

## Known omissions

The model omits adhesion, swept-volume surcharge/remolding history, cavity
expansion, rate-dependent strength, pore pressure, compaction/dilation,
three-dimensional side shear, tooth-by-tooth contact and a validated bucket
pressure distribution. It is suitable as a transparent real-time interaction
model and test-data generator, not as a field force predictor.
