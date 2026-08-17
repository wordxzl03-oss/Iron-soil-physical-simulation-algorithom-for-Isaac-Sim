# Mobile single-step flux report

This is a read-only reconstruction of
`outputs/mobile_large_avalanche_causal_audit/strongest_operator_boundary.npz`.
No production physics, material parameter, grid, LargeAvalanche behavior, or
checkpoint was modified. No GUI, excavation, five-second replay, or long replay
was run.

## Required result

```text
MOBILE_SINGLE_STEP_FLUX_REPORT

MASS_UPDATE_RECONSTRUCTION:
PASS

NEW_LOCAL_MAX_COUNT:
862

NEW_LOCAL_MIN_COUNT:
622

NEW_EXTREMUM_VOLUME_M3:
0.9122419144217069

STRONGEST_JH_CAUSE:
CENTERED_GRAVITY_PRESSURE_STENCIL_DRIVEN_DIFFERENTIAL_DIVERGENCE_AT_A_PREEXISTING_GRID_SCALE_FREE_SURFACE_VALLEY

MAX_VOLUME_CFL:
0.26341251192220755

MAX_WAVE_CFL:
0.3499999999999999

MASS_MOMENTUM_FLUX_CONSISTENCY:
PASS

COMPRESSION_FRONT_PHYSICAL_CAUSE:
NOT_A_COMPRESSION_FRONT__LOW_SIDE_DIVERGENCE; EXISTING MOMENTUM PLUS A CENTERED SOURCE GRADIENT EVACUATES THROUGH AN IMMEDIATELY_UPHILL_FACE

GRID_STENCIL_ARTIFACT:
YES

EFFECTIVE_MEAN_TRANSPORT_DISTANCE_5S_M:
2.654733903388448

ROOT_CAUSE_REFINED:
NON_WELL_BALANCED_CELL_CENTERED_GRAVITY_PRESSURE_SOURCE_AT_A_GRID_SCALE_FREE_SURFACE_VALLEY

SAFE_MINIMAL_NUMERICAL_FIX:
NONE
```

## Exact update reconstruction

The production update split `1/60 s` into six CFL substeps:

```text
0.0032090833
0.0031783168
0.0031657196
0.0031701679
0.0031987319
0.0007446472 s
```

Replaying the production equations in an independent NumPy finite-volume
ledger reproduced the frozen POST arrays with:

- maximum `h_mobile` error: `4.4408921e-16 m`;
- maximum x-momentum-density error: `8.8817842e-16 m2/s`;
- maximum y-momentum-density error: `6.6613381e-16 m2/s`;
- maximum per-cell volume equation residual: `1.6593375e-18 m3`;
- global sum of per-cell residuals: `1.5740169e-17 m3`.

For every cell, including the strongest-gradient pair, the reconstruction
satisfies

```text
A_i (h_new - h_old) = sum(Delta V_in) - sum(Delta V_out)
```

with zero explicit Mobile mass source. Momentum source terms were reconstructed
separately using the production equation

```text
du/dt = -g [grad(H_free) + 0.45 grad(h_mobile)]
```

followed by the existing Coulomb speed decrement. Tool forcing was zero.

## Strongest pair

The POST maximum is the Y face between cells `(393,136)` and `(394,136)`.

| quantity | high `(393,136)` | low `(394,136)` |
|---|---:|---:|
| old h_mobile (m) | 0.0002766242 | 0.3944415194 |
| new h_mobile (m) | 0.0000577808 | 0.2027638907 |
| delta h_mobile (m) | -0.0002188434 | -0.1916776287 |
| old H_free (m) | 1.8202925696 | 0.5144415194 |
| new H_free (m) | 1.8200737262 | 0.3227638907 |
| incoming volume (m3) | 0 | 0.0000767655 |
| outgoing volume (m3) | 0.0000005471 | 0.0005559595 |
| net convergence (m3) | -0.0000005471 | -0.0004791941 |

Thus the `J_H` increase from `1.3058510503` to `1.4973098356 m` is
not a compression maximum. Both cells diverge; the low cell is evacuated
about 876 times more strongly in volume than the high cell. Its six substep
`-div(q)` equivalents remain negative, from `-14.2107` to `-6.9700 m/s`.

The decisive stencil geometry is:

- high-cell Resting height: `1.8200159454 m`;
- low-cell old `H_free`: `0.5144415194 m`;
- forward/downstream-cell old `H_free`: `0.8266754619 m`;
- the immediate forward face is therefore uphill by `0.3122339426 m`;
- nevertheless, the centered source stencil adds `+0.9725067614 m/s`
  cumulative y velocity at the low cell;
- `0.0005559595 m3` crosses from the low cell through that uphill face.

The initial low/downstream y velocities were `+3.9002021` and `-0.1160910 m/s`.
Their pre-source face average was `+1.8920556 m/s`; after the first source and
friction update it was `+1.9799134 m/s`. Existing momentum supplies most of
the transport, while the centered gravity/pressure source reinforces it even
though the immediate face is uphill. This is an interior, uniform-control-area,
active–active face, so neither domain boundary treatment nor vertex weights
explain the event.

Every contributing face, donor, receiver, candidate and limited volume,
transported momentum, neighbor state, and substep balance is stored in the
machine-readable report.

## Monotonicity

For cells with no explicit mass source, the audit compared POST `h_mobile`
against the envelope of the full-step PRE value at the cell itself plus every
donor that actually supplied it. Results:

- 862 values exceed the old contributor maximum;
- 622 values fall below the old contributor minimum;
- their combined POST Mobile volume is `0.9122419144 m3`;
- volume above the contributor maximum envelope is `0.0368108888 m3`;
- volume below the contributor minimum envelope is `0.0223945861 m3`.

The scheme is conservative and positivity preserving, but variable-velocity
convergence/divergence means it is not a monotone height-averaging map. All
1,484 cell records are included in the JSON rather than being hidden behind
aggregate counts.

## CFL and mass–momentum consistency

Donor-limited volume CFL statistics, evaluated for every active donor in every
actual substep, were:

```text
max = 0.2634125119
p95 = 0.0989048256
p99 = 0.1413329040
```

Candidate and actual values were identical: this event never invoked the
donor limiter, and every donor was below one.

Production uses

```text
c = sqrt(pressure_coefficient * g * h_mobile)
C_wave = (|u| + c) dt_sub / dx
```

giving max/p95/p99 `0.35 / 0.1874128851 / 0.2236274181`. This is the exact
implemented shallow-layer CFL proxy. Because pressure and gravity are applied
as explicit cell-centered momentum sources rather than a conservative pressure
flux, the whole implementation is not a complete conservative hyperbolic
eigensystem; no alternative wave speed was invented for this report.

For each transported face, `Delta p/(rho Delta V)` matches the donor velocity
to `4.58e-16 m/s`. There are zero momentum-without-mass and zero
mass-without-finite-momentum faces. The face-average normal velocity determines
the crossing rate, while donor velocity determines momentum per transported
mass; their maximum normal-component difference is `3.4688231 m/s`, which is
the explicit first-order upwind scheme definition, not an accounting mismatch.
No checkerboard face-velocity pattern was detected at the strongest pair.

## Five-second path diagnostic and disposition

The existing five-second audit contains `166.6075583518 m3` of internal face
crossings. With 0.05 m faces:

```text
transport_path_volume = 8.3303779176 m4
representative Mobile volume = 3.1379332998 m3
effective mean transport distance = 2.6547339034 m
```

The representative volume is the arithmetic time mean of the 300 saved Mobile
reservoir samples. This is diagnostic throughput only.

No fix was applied. The evidence points toward a future face-consistent,
well-balanced gravity/pressure reconstruction, but that is a numerical-model
change requiring its own conservation, steady-state, CPU/DEVICE and morphology
acceptance. It is not safe to present as a one-line minimal fix.
