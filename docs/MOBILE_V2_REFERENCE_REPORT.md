# Mobile V2 reference report

Status: standalone numerical reference complete; **not a production
candidate**.

## Model and state

The physical state is independent of the legacy Resting/Mobile bookkeeping:

```text
b_eff   effective physical support/static-flowing interface
h       Mobile thickness
q       h*u, Mobile depth momentum
H_free  b_eff+h where Mobile occupies the surface
```

The reduced-order equations are

```text
dh/dt + div(h*u) = E-D

d(h*u)/dt + div(h*u tensor u + P(h) I)
  = -g*h*grad(b_eff)
    + Coulomb friction
    + external/tool forcing
    + entrainment/deposition momentum transfer,

P(h) = 0.5*K*g*h^2.
```

`K=0.45` is `ENGINEERING_CLOSURE_UNCALIBRATED`; this is not claimed as a
complete granular constitutive law.

The discrete mechanical energy per bulk density is

```text
sum A [0.5*h*|u|^2 + g*h*b_eff + 0.5*K*g*h^2].
```

Every step records kinetic, gravitational and pressure/internal energy,
external work, friction dissipation, exchange-energy transfer and numerical
residual.

## Numerical architecture

Mass and both momentum components use one shared state per internal face.
Pressure is inside the conservative face momentum flux. A generalized
hydrostatic reconstruction pairs that flux with `-g*h*grad(b_eff)`; the
first-order Rusanov face flux supplies positivity-compatible numerical
dissipation. There is no velocity-direction-dependent face selection,
smoothing, morphology clipping or added damping.

Pure bookkeeping relabel leaves `b_eff/h/q/H_free` bitwise unchanged. Physical
entrainment moves `b_eff` down and `h` up; deposition performs the reverse.
Both expose their momentum and energy exchange rather than disguising it as a
label operation.

## Contract results

| Contract | Result |
|---|---|
| Mass conservation/accounting | PASS |
| Positivity and finite CFL | PASS |
| Energy consistency | PASS |
| Flat static layer | PASS |
| Stable inclined layer | PASS |
| Historical grid-scale valley | PASS |
| Grid-scale mound | PASS |
| Smooth downslope packet | PASS |
| Opposing packets | PASS |
| Wet/dry front | PASS |
| Bookkeeping relabel | PASS |
| Physical entrainment | PASS |
| Physical deposition | PASS |

In the reconstructed historical valley (`rear=1.82 m`, `low=0.5144415194 m`,
`forward=0.8266754619 m`), the shared low/forward face does not accelerate the
low cell uphill. This follows from the two-state face flux/reconstruction; no
valley special case exists in the solver.

All zero-external-work dynamic contracts have non-positive energy residual.
Flat and stable-slope states remain exactly motionless. The remaining dynamic
cases have finite CFL, no negative layer, no energy-growing oscillation and
machine-precision mass/momentum accounting.

## DEVICE and frozen-field validation

The identical equations were ported to standalone Warp DEVICE only after CPU
contracts passed. Across flat, slope, valley, mound, wet/dry and smooth-packet
cases, maximum CPU/DEVICE field error is `1.39e-16`.

One uninterrupted frozen full-field DEVICE run captured 0.5, 2 and 5 seconds:

```text
maximum mass error          1.776e-15 m3
minimum h                   0 m
energy, initial -> 5 s      14.80327 -> 11.07223 J per density
max H_free jump             1.30585 -> 1.29415 m
p99 H_free jump             0.058042 -> 0.053793 m
```

The legacy checkpoint has no independently evolved `b_eff`. Its adapter is
therefore explicitly classified
`UNCALIBRATED_SNAPSHOT_COMPATIBILITY_ADAPTER`: algebraically
`b_eff := frozen H_free - frozen h`, without claiming legacy `H_resting` is a
validated physical support surface.

## Decision

`PRODUCTION_CANDIDATE: NO`.

The numerical V2 reference has passed its requested standalone, DEVICE and
frozen-field contracts. Production promotion would first require an
authoritative `b_eff` lifecycle coupled consistently to production physical
entrainment and deposition. Those changes were not made here. The production
Mobile operator remains frozen.

