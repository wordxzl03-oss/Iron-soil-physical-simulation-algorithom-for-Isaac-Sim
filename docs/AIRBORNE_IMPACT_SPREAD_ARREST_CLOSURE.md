# Airborne Impact / Spread / Arrest Closure

## Result

The structural top-hat and zero-Mobile-lifetime defects are closed in the production 701×701, 0.05 m, GPU_RUNTIME/DEVICE path. The same production path conserves the landing, preserves its horizontal momentum, transports it as Mobile, and passes the new `AIRBORNE_LANDING_CAN_MOBILIZE` ownership contract.

`ARREST_FINAL` is **not** demonstrated. The exact full-cycle dump state remains dynamically active after a causally bounded 60 s soil-physics observation. Nothing was force-settled, smoothed, clipped, or parameter-tuned. The status is therefore:

`IMPACT_AND_SPREAD_CLOSED__ARREST_NOT_REACHED_AT_CAUSAL_60S_HORIZON`

The machine-readable result is [summary.json](../outputs/airborne_impact_spread_arrest/summary.json).

## Causal findings

### Uniform plateau

The old plateau was structural, not noise. Uniform footprint weights imposed

```text
delta_h = parcel volume / footprint area
        = 0.03527664186 / 0.03430534635
        = 1.02831324 m
```

The accepted conservative polygon/control-area rasterizer was retained. Only its internal mass profile changed. Because the released aggregate parcels do not retain a resolved bucket-internal fill field, the source now uses the simplest compact, nonnegative, normalized profile available without new calibration parameters: an elliptic cone whose volume, bucket-mouth aspect, and boundary slope are fixed by the existing parcel volume, bucket geometry, and frozen `Y_stop` angle. This is a `CONSERVATION_BASED_ENGINEERING_CLOSURE`, not calibrated granular dispersion.

There is no Gaussian/median filter, blur, height clipping, manual slope clamp, visualization-only deformation, or larger footprint selected by appearance.

### Zero Mobile lifetime

The physics ordering was already capable of transporting newly landed Mobile. The zero lifetime came from split device authority after checkpoint restore:

1. `restore_device_checkpoint()` replaced the canonical Warp `mobile` allocation.
2. `WarpMobileLayerSolver` retained a historical `height` alias to the old allocation.
3. Airborne wrote height and horizontal momentum into the restored canonical arrays.
4. Mobile read zero height through the stale alias but read nonzero momentum through the canonical momentum arrays.
5. Its dry-state guard removed that momentum, after which same-frame subcell-tail deposition consumed the landing.

The solver now reads the canonical `mobile` array at every kernel boundary. No cooldown or hysteresis epsilon was added.

At the corrected impact frame:

| Quantity | Value |
|---|---:|
| Mobile before transport | 0.113715508959 m³ |
| Mobile after transport | 0.113715508959 m³ |
| Moving Mobile after transport | 0.112747617545 m³ |
| Maximum Mobile speed | 1.252868106 m/s |
| Same-frame Mobile→Resting | 0.000268196132 m³ |
| Impact Mobile substeps | 2 |

Thus `IMPACT_MOBILE_LIFETIME_ORDERING_BUG = false`; the defect was an authoritative-array alias bug.

## Production operator order

The observed step order is:

1. Airborne collision and terrain query.
2. Airborne→Mobile height write.
3. Horizontal Mobile momentum write.
4. Mobile conservative transport.
5. Mobile→Resting deposition.
6. LargeAvalanche Resting→Mobile transition.
7. Residual projection only after Airborne/Mobile/physical yield are quiet.

The persisted boundary capture contains `H_resting`, `h_mobile`, `H_free`, momentum, velocity, volume, Y_start/Y_stop, deposition eligibility, ownership, cumulative conservative export, and reservoir metadata around the exact impact. See `impact_operator_boundaries.npz` and its JSON metadata.

## Momentum semantics

Horizontal impact momentum is written to Mobile and is then evolved by the existing conservative Mobile solver. The terrain-normal collision is represented as perfectly inelastic:

- normal momentum is recorded as an impulse on terrain;
- normal kinetic energy is recorded as unresolved dissipated granular impact energy;
- no arbitrary conversion of vertical energy into lateral velocity is applied.

For the three simultaneous parcels, each has mass 48.329 kg and approximately −10.299 m/s vertical impact speed. Each books approximately −497.748 N·s terrain-normal impulse and 2563.19 J unresolved impact dissipation. This is an explicit `CONSERVATION_BASED_ENGINEERING_CLOSURE`. A vertical agitation/spreading-energy state remains absent rather than silently implied.

## Morphology

Each 0.03527664 m³ parcel occupies about 0.473–0.477 m² of nonzero compact support over 198 cells. The per-parcel initial thickness maxima are 0.22068, 0.21191, and 0.20648 m.

| Metric | Old uniform top-hat | New compact source |
|---|---:|---:|
| max delta H | 1.028313240 m | 0.220684375 m |
| max 4-neighbor jump | 0.907103965 m | 0.039460899 m |
| p95 4-neighbor jump | — | 0.030930916 m |
| p99 4-neighbor jump | — | 0.036803004 m |

The new post-write field has one two-cell anomalous-gradient component under the acceptance diagnostic. The improvement follows directly from replacing constant areal density with the normalized finite source profile; subsequent shape change is generated by Mobile transport and Y_stop deposition.

The new Mobile remains moving for the complete 13.65 s isolated observation, versus 0 s before the alias fix. Conservative flux crossings out of the landing support total 0.327380979 m³; this is cumulative boundary crossing and may exceed source volume through repeated crossings. The aggregate Eulerian Mobile centroid moves 4.038 m over the isolated horizon. The three parcels merge into one authoritative Eulerian field on landing, so parcel-specific distance and parcel-specific arrest time are intentionally not claimed.

## Ownership and raw Y_start

`AIRBORNE_LANDING_CAN_MOBILIZE = PASS`:

- newly arrived Mobile does not inherit an unrelated Resting avalanche latch;
- 0.112747618 m³ is moving after the first Mobile transport;
- authoritative conservative transport crosses cells;
- no global ownership clear was introduced.

At the isolated horizon the raw Y_start attribution is:

| Category | Cells |
|---|---:|
| total Y_start | 129 |
| previously owned terrain | 16 |
| new landing support | 0 |
| footprint boundary | 0 |
| incorrectly ownership-shielded new landing | 0 |

Eligible mobilizable volume is 0.0066 m³. The previously reported 49→114/116 comparison came from the stale-alias replay and is not a like-for-like final state after the authoritative-array repair.

## Full-cycle attribution

The real production cycle checkpoints are materially different from the older isolated checkpoint.

| State | T_DUMP_END | T_DUMP+9 s |
|---|---:|---:|
| device time | 20.766667 s | 29.766667 s |
| Mobile | 3.020369727 m³ | 3.376385166 m³ |
| moving Mobile | 2.951438507 m³ | 3.326270815 m³ |
| max speed | 3.575060190 m/s | 4.195539625 m/s |
| Airborne | 0.430464370 m³ | 0 m³ |
| parcels | 11 | 0 |
| Y_start cells | 1936 | 2434 |

The difference is classified as `MORE_MOBILE_INITIAL_CONDITION`, `DIFFERENT_MOMENTUM_STATE`, `CONTINUING_AIRBORNE_INPUT`, and `GUI_STATE_MACHINE_SIDE_EFFECT`. Here the last label means that ongoing production vehicle/task evolution changes the authoritative state relative to a stationary-tool replay; it is not a viewport-renderer numerical effect.

A stationary-tool headless replay from the exact T_DUMP_END checkpoint remains in `LARGE_AVALANCHE_MOBILE_PATH` after 60 s: Mobile and moving Mobile are both 7.020193207 m³, maximum speed is 5.039443511 m/s, Y_start has 4930 cells, and eligible mobilizable volume is 0.0132 m³. `ARREST_FINAL` is false and the reason is `DEVICE_DYNAMIC_MOBILE_FLOW`.

This establishes a remaining flow/arrest limitation. It does not justify a force-settle, arbitrary cooldown, parameter change, or truncation masquerading as arrest.

## Conservation

All 0.105829925584 m³ of isolated Airborne material lands. The isolated net H_free volume error is 2.2737×10⁻¹² m³ and the maximum ledger error is 2.0918×10⁻¹¹ m³. The 60 s headless replay maximum mass error is 3.9563×10⁻¹¹ m³. The final GUI RESET error is recorded in the machine-readable report.

`AIRBORNE_VOLUME_CONSERVED = YES` and `MASS_LEDGER = PASS`.

## Non-headless GUI regression

The production Isaac GUI ran the real 390F GPU_RUNTIME/DEVICE cycle for 60.0167 s of post-dump terrain physics and then paused without claiming settled state. At pause, Mobile was 4.496000865 m³ and `root_pose_write_count = 0`. The measured mean frame time was 74.753 ms; the core's reported mean RTF was 6.9569 (operator RTF, not rendered wall-clock RTF).

Lifecycle evidence:

- READY GUI hold: 30.0040 s, process alive; a real UI event entered RUNNING at the boundary, and the idempotent hook did not issue a duplicate start;
- post-dump PAUSE hold: 30.0070 s, process alive;
- RESET→READY hold: 30.0112 s, process alive;
- RESET: Payload 0, Mobile 0, base-pose error 0, terrain volume error 1.5461×10⁻¹¹ m³;
- exit occurred only through the explicit test-exit hook.

`GUI_REGRESSION = PASS_WITH_ACTIVE_FLOW_AT_OBSERVATION_HORIZON`.

## Regression scope

- Phase G Airborne/Spill/Dump: 10 passed.
- Phase G contains the source-profile and Airborne ownership contracts. A separate focused local-redeposit/real-export ownership, differential-track, and 390F state-machine run passed 21 tests.
- Modified Python modules compile successfully.

An additional legacy CPU LargeAvalanche compatibility sweep produced six failures because those fixtures expect angle-only activation while their material specifies 1500 Pa cohesion and a 0.05 m failure layer. Under the frozen cohesive V3 criterion, the driving stress does not exceed cohesion in those cases. Those fixtures require a separate semantic migration; this closure did not alter frozen physics thresholds to make them pass.

## Remaining limitations

1. `ARREST_FINAL` is not reached at the 60 s causal horizon; the full-cycle event grows into the existing LargeAvalanche Mobile path.
2. There is no vertical granular agitation reservoir. Normal impact energy is explicitly accounted as unresolved dissipated energy, not converted with a visual/tuned multiplier.
3. Eulerian Mobile has no parcel lineage, so per-parcel distance and resting time are unavailable after simultaneous sources merge. Aggregate conservative flux and motion are reported instead.
