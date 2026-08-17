# Earthmoving Physics Core V3 redesign

Status: implemented structural baseline; reference iron-ore-fines parameters are
`NOT_YET_PHYSICALLY_CALIBRATED`.

## Scope and invariants

V3 changes the failure morphology, momentum closure, yield/arrest rule and
solver time semantics. It does not change the formal 701×701, 0.05 m production
terrain resolution or weaken the conservative material ledger. UI, rendering,
trajectory, RL, DEM, MPM, world-model and product work are outside this change.

The only physical height-field surface is now

```text
H_free = H_resting + h_mobile
```

Tool intersection, Failure Surface geometry, Mobile Layer sources, deposition,
LargeAvalanche and stability diagnosis consume this surface. Resting remains the
capacity available for Resting→Mobile transfer. Because activation only changes
the Resting/Mobile label, T2 activation preserves `H_free` exactly.

## Failure Surface V3

The old model formed independent 0.20 m strips and let the maximum penetration
vertex in each strip define a constant triangular prism. That directly assigned
periodic boundary steps, local extrema and hard terminal edges.

V3 samples the real cutting-edge lateral coordinate at no coarser than 0.05 m
and constructs continuous fields

```text
d(s), alpha(s), beta0(s), beta(s), L(s)
```

`beta0(s)` is still selected by minimum admissible FEE resistance. `d` and
`alpha` use a compact lateral filter; `beta` blends the pointwise optimum with
a compact continuity filter. A cubic smoothstep supplies finite left/right
transition width. Each quadrature slice is conservatively integrated into the
same height-field control volumes. The resulting `FailureStripGeometry` tuple
is a compatibility name for these quadrature slices and remains the single
geometry consumed by SoilForce; it is not an independent strip model.

Classifications:

- FEE coefficients and force terms: `PAPER_DIRECT`.
- Critical-plane minimum-resistance selection: `LITERATURE_INFORMED_REDUCED_ORDER`.
- Robust lateral depth sampling, smoothing and endpoint transition:
  `ENGINEERING_CLOSURE` / `REDUCED_ORDER_ENGINEERING_CLOSURE`.
- Coefficients and transition widths for this iron ore:
  `NOT_YET_PHYSICALLY_CALIBRATED`.

The legacy implementation remains selectable only as
`surface_version=LEGACY_STRIPS` for the frozen comparison.

## Momentum closure

Failure activation transfers mass only:

```text
Resting -> Mobile
delta(momentum)_activation = 0
```

The cutting-edge velocity is no longer copied over the failure support. Mobile
tool forcing is restricted to the current intersection plus one conservative
stencil-neighbour contact layer. `MobileMomentumBudget` measures the resulting
soil impulse, and SoilForce applies the paired reaction:

```text
J_soil_on_tool = -J_tool_on_soil
```

This is a `CONSERVATION_BASED_ENGINEERING_CLOSURE`. It removes the former
whole-wedge forward launch while retaining the existing explicit impulse ledger.

## Cohesive yield, flow and arrest

CPU and resident Warp paths evaluate the same height-field Mohr-Coulomb balance:

```text
tau_drive  = rho g h sin(theta)
tau_resist = c + rho g h cos(theta) tan(phi)
Y          = tau_drive - tau_resist
```

Configured start and stop angles act as effective start/stop friction angles.
Static material starts when `Y_start > 0`; moving material continues while
`Y_stop > 0`. This permits a steep but thin cohesive face to remain static.
Deposition occurs only after speed and stop-yield conditions are both satisfied.
LargeAvalanche uses the same stress criterion on its configured mobilizable
Resting depth instead of a fixed slope-angle trigger.

This is `LITERATURE_INFORMED_REDUCED_ORDER`. Mapping 3-D cohesive granular
strength to one height-field failure depth and using the scenario start/stop
angles as effective friction parameters are `ENGINEERING_CLOSURE`; all current
iron-ore parameters remain `NOT_YET_PHYSICALLY_CALIBRATED`.

## Time semantics

- `physical_simulation_time_s`: advanced only by production physics `dt`.
- `dynamic_flow_time_s`: accumulated only while resolved Mobile material moves.
- `residual_solver_iterations`: numerical work counter; never converted to
  seconds.
- Mobile and LargeAvalanche evolve in physical time.
- MiniSlope is `QUASI_STATIC_RESIDUAL_PROJECTION`. Once physical Mobile and
  airborne reservoirs are quiet and the physical instability detector admits a
  local residual, the compact frontier performs a bounded multi-round solve in
  that same physics call. Large connected failures remain on the Mobile/yield
  path. Exceeding the independent numerical safety bound is an explicit
  `NUMERICAL_NONCONVERGENCE`, never a force-settled state.

No velocity/deposition multiplier, relaxed tolerance, timeout-settled rule or
resolution reduction was introduced.

## PhysicsDiagnostics

`runtime.physics_diagnostics.PhysicsDiagnostics` is a frozen observational
schema exposed by `EarthmovingPhysicsCore.physics_diagnostics` and on each step
result. It reports material/backend identity, all four active reservoirs, soil
force and torque, yielded/failure/moving measures, velocity p95 (a conservative
maximum-speed scalar proxy on DEVICE), terrain state/reason, mass error,
physical/dynamic time, residual iteration count and RTF. These values are
computed after the physics decision and never feed back into it.

## One-time forensic replay

The offline tool `tools/run_physics_core_v3_validation.py` is the only added
forensic instrumentation. It saves T0–T3 Resting, Mobile, free-surface,
intersection/failure support, activation thickness and momentum fields to
`outputs/390f_v3/frozen_excavation_forensics.npz`. It is not imported by the
production package and therefore adds no hot-path diagnostics.

Root-cause assignment from that replay:

- cat-ear/terminal depression: independent terminal strips and max depth;
- shallow surrounding Resting ring: partial conservative cells were displayed
  as the surface instead of the unchanged authoritative `H_free` at activation;
- serration: piecewise-constant 0.20 m `d/beta/L`;
- local protrusions: one deepest vertex controlled an entire strip;
- forward launch: whole failure support inherited tool velocity and forcing.

The four and only four V3 gates are flat static terrain, cohesive steep cut,
cohesionless over-steep start/arrest, and this frozen old-vs-V3 excavation.
