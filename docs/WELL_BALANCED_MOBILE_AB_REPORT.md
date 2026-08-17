# Experimental well-balanced Mobile source A/B

Status: `EXPERIMENTAL_ONLY_NOT_PRODUCTION`

This audit replays exactly one frozen production operator boundary at
`t = 3.75 s` with `dt = 1/60 s`. It does not run GUI, excavation, a 5 s
continuation, or a 60 s continuation. It does not alter production defaults,
material values, clipping, damping, CFL, or the production Mobile operator.

## Intended continuous reduced-order model

The conserved Mobile variables are thickness `h = h_mobile` and depth
momentum `m = h u`. The authoritative published free surface is

```text
H_free = H_resting + h_mobile.
```

The intended balance-law structure is

```text
partial_t h + div(h u) = 0

partial_t(h u) + div(h u tensor u)
    = -g h [face gradient of H_free + K face gradient of h]
      + cohesive start/stop gating
      + Coulomb basal resistance.
```

Roles and classification:

| Quantity/equation | Role | Classification |
|---|---|---|
| `h_mobile` | Conserved moving-layer thickness | `PAPER_DIRECT` balance-law architecture |
| `h_mobile * u` | Conserved depth momentum transported with donor mass | `PAPER_DIRECT` balance-law architecture |
| `H_free` | Authoritative physical free surface used for gravity/yield geometry | `LITERATURE_INFORMED_REDUCED_ORDER` |
| `H_resting` | Resting reservoir label; **not** silently treated as physical bed | `ENGINEERING_CLOSURE` state split |
| `K = 0.45` | Existing earth-pressure coefficient multiplying the Mobile pressure contribution | `ENGINEERING_CLOSURE`, uncalibrated |
| gravity | Face-local downslope drive | `LITERATURE_INFORMED_REDUCED_ORDER` |
| start/stop cohesion gate | Granular static/yield classification with existing hysteresis | `LITERATURE_INFORMED_REDUCED_ORDER`, uncalibrated |
| Coulomb Mobile friction | Existing basal speed decrement | `LITERATURE_INFORMED_REDUCED_ORDER`, uncalibrated |
| experimental face selection | Select the immediate transport face and use it for both `H_free` and `h` contributions | `ENGINEERING_CLOSURE` |

An independent effective basal/support surface is required for a
constitutively complete granular hydrostatic reconstruction. It is not present
in the current authoritative state. `H_resting` cannot safely substitute for
it: conservative Resting/Mobile relabeling can make `H_resting` discontinuous
while leaving `H_free` exactly unchanged. Consequently this experiment tests
face consistency and equilibrium contracts; it does not claim to establish a
unique granular hydrostatic law.

Hydrostatic-reconstruction work is used only as numerical-architecture
guidance: match flux and source geometry at the same face and preserve relevant
steady states and positivity. The equilibrium here is granular start/stop
equilibrium, not merely the shallow-water lake-at-rest state.

Primary numerical/constitutive guidance used for this classification:

- Audusse et al., hydrostatic reconstruction, well balancing, and positivity:
  <https://arxiv.org/abs/1409.3825>
- Bouchut and Morales de Luna, hydrostatic reconstruction on irregular
  topography: <https://epubs.siam.org/doi/10.1137/090758416>
- Granular depth-averaged/Coulomb and earth-pressure context:
  <https://doi.org/10.1061/%28ASCE%29HY.1943-7900.0000772>

## Experimental discrete source

For each coordinate direction, moving material selects the immediate face in
its current direction of travel. Quiet material selects an immediately
downhill face only; a local valley with uphill neighbours on both sides gets no
force merely because one opposite cell is very high. Once selected from
`H_free`, the exact same face is used for the `h_mobile` pressure gradient.
The existing start/stop gate and Coulomb friction remain unchanged.

The source is then applied before the unchanged conservative donor-mass and
donor-momentum face transfer. This is a face-consistent engineering candidate,
not a production promotion and not a direct copy of a shallow-water formula.

## Static and device contracts

| Contract | Result |
|---|---|
| Flat static Mobile layer | PASS |
| Stable inclined layer with `Y_start <= 0` | PASS |
| Static after `Y_stop` arrest | PASS |
| Resting/Mobile relabel at identical `H_free` | PASS |
| Wet/dry nonnegativity | PASS (`min h = 0`) |
| CPU/Warp DEVICE classification and field equivalence | PASS |

Frozen full-field CPU/DEVICE errors are `4.44e-16 m` in `h_mobile` and
`1.11e-15 m^2/s` in momentum. Both used six CFL substeps.

## Strongest-pair A/B

The strongest pair is `(393,136)` / `(394,136)`; the known immediately
uphill downstream cell is `(395,136)`.

| State | Pair `H_free` (m) | Pair `h_mobile` (m) |
|---|---|---|
| PRE | `[1.8202925696, 0.5144415194]` | `[0.0002766242, 0.3944415194]` |
| CURRENT_CENTERED | `[1.8200737262, 0.3227638907]` | `[0.0000577808, 0.2027638907]` |
| EXPERIMENTAL | `[1.8208886889, 0.5839215260]` | `[0.0008727435, 0.4639215260]` |

At low `(394,136)` toward uphill downstream `(395,136)`:

| Measure | CURRENT_CENTERED | EXPERIMENTAL |
|---|---:|---:|
| cumulative gravity/pressure `delta v_y` (m/s) | `+0.9725067614` | `-1.8400098974` |
| classification | reinforces uphill motion | decelerates uphill motion |
| transported volume (m3) | `5.5595954609e-4` | `2.4467463376e-4` |
| transported momentum `[x,y]` (kg m/s) | `[1.1558100, 3.2288074]` | `[0.2499589, 1.0198712]` |

For the experimental low cell, cumulative face-associated gravity/pressure
impulses are `[-0.6493408, 0] kg m/s` on selected x faces and
`[0, -2.4182513] kg m/s` on selected y faces. Basal friction contributes
`[-0.0108253, -0.0739928] kg m/s`. The current centered source cannot be
truthfully decomposed as a same-face source; its cell source and all current
fluxes are retained in the JSON evidence.

All per-substep face mass fluxes, carried momentum fluxes, selected face
offsets, same-face gradients, x/y source impulses, and friction impulses for
both pair cells are recorded in
`outputs/mobile_large_avalanche_causal_audit/well_balanced_mobile_ab.json`.

## Morphology and conservation

| Metric | CURRENT_CENTERED | EXPERIMENTAL |
|---|---:|---:|
| `delta J_H` (m) | `+0.1914587853` | `-0.0425111477` |
| new local maxima | `862` | `647` |
| new local minima | `622` | `499` |
| extremum Mobile volume (m3) | `0.9122419144` | `0.5402332912` |
| above contributor envelope (m3) | `0.0368108888` | `0.0207512338` |
| below contributor envelope (m3) | `0.0223945861` | `0.0298159738` |

The experimental result improves the maximum jump and reduces counts and
above-envelope volume, but below-envelope volume is worse. Therefore this is
not a blanket morphology PASS. For auditability, “large” means the top one
percent of nonzero outside-envelope distances; this threshold is reporting
only and never enters physics. All 12 large experimental extrema reconstruct
as conservative face convergence (maxima) or divergence (minima), and each
record includes its local cumulative source-impulse direction. The 15 current
large extrema reconstruct in mass, but their centered source is explicitly
classified as not face-attributable.

Conservation for the experimental frozen step:

```text
initial volume                 3.0804917641483325 m3
final volume                   3.0804917641483330 m3
mass residual                  4.440892098500626e-16 m3
integrated source impulse      [-13.3644522, 25.1722393] kg m/s
momentum balance residual      [-5.0583e-08, -1.7274e-07] kg m/s
minimum h_mobile               0 m
maximum volume CFL             0.2231445772
maximum wave CFL               0.35
```

The momentum residual is about `6.3e-9` relative to the integrated source
impulse norm and is floating accumulation from the full-field sums. The
unchanged velocity cap fired zero times and dry cleanup removed zero momentum;
no external momentum or mass was added.

## Interpretation and remaining ambiguity

The A/B causally supports the diagnosis: using the same immediate face for
force and transport removes the demonstrated opposite-cell reinforcement of
uphill motion and sharply reduces the one-step grid-scale jump. It also shows
that the candidate is not yet sufficient for production promotion: it does not
define an independent support geometry or calibrated granular pressure law,
and it increases below-envelope deficit volume in this one step.

Final status: `EXPERIMENTAL_ONLY_NOT_PRODUCTION`.
