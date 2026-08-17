# Mobile / LargeAvalanche causal report

Status: `CAUSAL_AUDIT_COMPLETE__NO_BROAD_FIX_APPLIED`.

This audit restored the frozen production `T_DUMP_END` DEVICE checkpoint and
advanced the production `EarthmovingPhysicsCore` for 5.0 terrain-physics
seconds at `dt=1/60 s`, with a stationary tool, idle phase, and no external
forcing. It did not run a new excavation, a 60 s replay, or Isaac GUI.

## Required result

```text
MOBILE_LARGE_AVALANCHE_CAUSAL_REPORT

PRE_DUMP_FIRST_SPIKE_TIME_S:
earliest captured = 6.5333333333333155 device s (POST_CUT)
exact observation bracket = (1.1500000000000008, 6.5333333333333155] device s

PRE_DUMP_FIRST_SPIKE_OPERATOR:
MOBILE_TRANSPORT
(causal inference; exact pre-dump first-frame boundary was not persisted)

MOBILE_GROWTH_SOURCE:
R2M_LONG_TERM_WITH_FINITE_INITIAL_A2M

R2M_0_TO_5S_M3:
1.3253100527867243

M2R_0_TO_5S_M3:
1.5781131130961064

ENERGY_BUDGET_STATUS:
PASS__MECHANICAL_ENERGY_DECREASED

NUMERICAL_ENERGY_CREATION:
NO

ARREST_STATE_A_VOLUME_M3:
1.3867294032273212

ARREST_STATE_B_VOLUME_M3:
1.6204958856252651

ARREST_STATE_C_VOLUME_M3:
0.19080574784545223

ARREST_STATE_D_VOLUME_M3:
0.0

SETTLE_SUBCELL_TAIL_ROLE:
IRRELEVANT__NOT_TRIGGERED

HEADLESS_GUI_5S_EQUIVALENCE:
FAIL_NOT_DEMONSTRATED__GUI_RUN_BLOCKED_BY_FAST_ITERATION_GATE

CPU_DEVICE_YIELD_EQUIVALENCE:
PASS

ROOT_CAUSE:
OVERACTIVATION_FROM_HEIGHTFIELD_CLOSURE

MINIMAL_PHYSICS_FIX:
NONE_APPLIED__NO_UNAMBIGUOUS_LOCAL_FIX
```

## First-bad-write evidence

The saved production stages show the first captured sharp morphology between
PRE_DIG and POST_CUT:

| stage | device s | J_H (m) | J_R (m) | J_M (m) | max h_mobile (m) |
|---|---:|---:|---:|---:|---:|
| PRE_DIG | 1.150000 | 0.036814 | 0.036814 | 0.000000 | 0.000000 |
| POST_CUT | 6.533333 | 0.748975 | 2.050816 | 1.425807 | 1.540654 |
| POST_BREAKOUT | 9.816667 | 0.882145 | 1.670415 | 1.194904 | 1.496939 |
| EARLY_POST_DIG | 10.083333 | 0.776024 | 1.670415 | 1.146090 | 1.491939 |

The production artifacts do not contain every pre-dump operator boundary, so
the exact first frame cannot be recovered without rerunning the excavation.
That rerun was explicitly avoided in fast-iteration mode. The operator
attribution is nevertheless constrained: Failure and LargeAvalanche R2M/M2R
are pointwise relabels and cannot change `H_free`; in the exact frozen replay,
the largest per-operator `J_H` write occurred at `AFTER_MOBILE_TRANSPORT` at
3.75 s, changing `J_H` from 1.3058510503 m to 1.4973098356 m. Intake is a real
free-surface removal, but its POST_CUT payload was only 0.0200868 m3 and an
exact pre-dump intake boundary was not saved. Therefore `MOBILE_TRANSPORT` is a
causal attribution, not a falsely claimed recovered frame record.

## Reservoir and activation cause

The exact five-second reservoir identity is:

```text
Delta Mobile
= R2M + A2M - M2R - Outflow + residual
= 1.3253100527867243
 + 0.4304643700385642
 - 1.5781131130961064
 - 0
 + 2.6536932373755207e-15
= 0.1776613097291851 m3
```

Airborne input ends at 1.35 s and is finite. It cannot explain the historical
growth from 3.02 to 7.02 m3. Continued reservoir supply is R2M.

There were 74 R2M activation events. Source attribution was:

- `MOBILE_REDEPOSITION`: 1.3113338268811878 m3 (98.945% of R2M)
- `PREVIOUSLY_OWNED_RESTING`: 0.0123762259055390 m3
- `NEWLY_EXPOSED_RESTING`: 0.0016000000000000 m3
- `AIRBORNE_DEPOSIT`: 0 m3

Ownership release attribution was 1.0995100527867274 m3 by actual conservative
export, 0.0628 m3 by Y_stop-stable release, and 0.163 m3 on first activation.
Consequently, this is not the former same-cell, no-export ownership retrigger
bug. Mobile transport alters the Eulerian free surface and exports the owned
tranche; ordinary deposition then relabels locally quiet Mobile to Resting;
the heightfield remains steep, so the fixed-depth transition repeatedly
mobilizes mainly redeposited material. The present closure has no finite
failure-tranche exhaustion or three-dimensional runout provenance capable of
deciding when that redeposited material is the same exhausted failure mass.

## Energy, transport, and arrest

Over five seconds:

- Kinetic energy: 1931.8770954 -> 2088.7396067 J.
- Heightfield potential energy: 37069945.5594282 -> 37064405.1209573 J.
- Total mechanical change: -5383.5759596 J.
- Mobile transport numerical energy residual: -260244.4910706 J.
- Tool work: approximately zero.
- Donor export = receiver import = 166.6075583518 m3 of internal edge
  crossings; the accumulated end-to-end Mobile volume residual was
  1.2967404928e-13 m3.

The large internal crossing number is throughput, not reservoir creation.
Transport is conservative and strongly dissipative, so this audit rejects
`NUMERICAL_ENERGY_CREATION`.

At 5 s, A/B/C/D volumes were 1.3867294 / 1.6204959 / 0.1908057 / 0 m3.
Only 0.1908 m3 was in quiet, below-Y_stop state C; normal deposition did run
and transferred 1.5781131 m3 cumulatively. Most material remained in A or B,
so `ARREST_IMPLEMENTATION_BUG` is not supported. The system keeps regenerating
yielded/moving state instead.

The actual global subcell-tail threshold is 0.000125 m3. Mobile stayed near
3 m3, so it triggered on zero of 300 steps and transferred zero volume. The
318397 cell updates and 1.5781131 m3 previously visible in `deposition_work`
were ordinary deposition, not the subcell-tail path.

## CPU/DEVICE cohesive onset

For the requested 45 degree, 0.05 m layer with rho=2200 kg/m3,
cohesion=1500 Pa and phi_start=38 degrees:

```text
tau_drive = 763.0389275784 Pa
tau_resist_start = 2096.1513465821 Pa
Y_start margin = -1333.1124190037 Pa
```

CPU and DEVICE both returned zero unstable cells and `NO_LARGE_EVENT`.
The old angle-only expectation was a stale test fixture; the mechanics tests
now explicitly use cohesionless material, while the cohesive analytical test
retains 1500 Pa. Focused regression: 20 passed.

## Fix disposition and execution-path gate

No production physics fix was retained. A targeted experiment replacing the
authoritative `H_free` gradient with `H_resting` was rejected and reverted: it
worsened five-second Mobile volume to 4.5100390732 m3, maximum speed to
6.8078467303 m/s, and `J_H` to 2.1918340676 m because R2M/M2R ownership
interfaces make `H_resting` an invalid physical bed surface.

The remaining candidate solutions (a finite failed-mass/tranche closure or a
more complete depth-resolved/runout model) change model semantics and are not
an unambiguous minimal edit. Per the hard gate, no GUI or long closure run was
started. `HEADLESS_GUI_5S_EQUIVALENCE` is therefore not demonstrated and is
reported as fail, not silently promoted to pass.

Two same-checkpoint headless repeats ended at 3.19874943 and 3.19803104 m3
Mobile, but their maximum speeds were 3.95750 and 4.63105 m/s. This records
GPU atomic-order sensitivity amplified by thresholded R2M activation. It is a
repeatability warning, not evidence of a separate GUI execution-path semantic
difference; an exact GUI comparison remains behind the repair gate.

Full machine-readable evidence is in
`outputs/mobile_large_avalanche_causal_audit/short_replay_5s.json`; the exact
strongest operator fields are in `strongest_operator_boundary.npz`.
