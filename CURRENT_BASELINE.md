# Current Authoritative Baseline

Version boundary: `v0.1.0-mobile-v2-baseline`

This document is the authoritative acceptance boundary for the first Git
baseline. It describes the state before realistic 390F scoop-trajectory
correction, final tool-Mobile momentum validation, and PRE_DUMP morphology
acceptance.

## ACCEPTED

- FailureSurface V3 morphology and FailureZone activation semantics.
- Mobile V2 conservative finite-volume numerical core.
- Generalized well-balanced reconstruction and two-state shared-face flux.
- First-order Rusanov reference flux and positivity-compatible CFL handling.
- Authoritative state: `z_base`, `b_eff`, `h_mobile`, and Mobile momentum.
- State migration and `b_eff` checkpoint persistence.
- Conservative entrainment and deposition contracts.
- Airborne-to-V2 deposition path and ownership contract.
- FailureZone-to-V2 and LargeAvalanche-to-V2 transfers.
- MiniSlope as a local/residual static-relaxation operator.
- Production integration Gate A and frozen production Gate B.
- 701 x 701 grid at `dx=dy=0.05 m` for the production terrain configuration.

## REJECTED

- Mobile V1 centered gravity/pressure source.
- Velocity-selected face heuristic.
- Treating iteration-count thresholds as physical avalanche transition rules.
- Any claim that the current full 390F excavation cycle is production-accepted.

## CLOSED HISTORICAL BUGS

- Resting-to-Mobile / Mobile-to-Resting same-tranche retrigger loop.
- Airborne single-cell Dirac landing spike.
- Warp stale `resting` alias after authoritative checkpoint restore.
- Blocking quasi-static MiniSlope use for sustained large avalanches.

## CURRENT BLOCKERS

- Existing CUT_AND_FILL motion behaves primarily as push/penetrate/drag rather
  than coordinated stick retraction plus bucket curl.
- Gross R2M does not translate into sufficient mouth-crossing Mobile flux.
- Final tool-to-Mobile momentum coupling has not passed production validation.
- The unchanged 0.02 m3 CUT_AND_FILL Payload-gain gate is not reached within
  the unchanged 12 s phase timeout.
- PRE_DUMP morphology and the complete production cycle remain unobserved.

## VALIDATION STATUS

- `FAILURESURFACE_V3: ACCEPTED`
- `MOBILE_V2_CORE: ACCEPTED`
- `PRODUCTION_GATE_A: PASS`
- `PRODUCTION_GATE_B: PASS`
- Frozen 5 s energy/conservation: `PASS`.
- Terrain physics RTF in the frozen Gate B evidence: approximately `1.05`.
- Mass balance: machine-precision scale in accepted frozen checks.
- `PRE_DUMP_REACHED: NO`
- `VISIBLE_NEEDLE_FOREST: NOT_EVALUATED`
- `VISIBLE_TRIANGULAR_FINS: NOT_EVALUATED`
- `GRID_SCALE_SPIKE_PROLIFERATION: NOT_EVALUATED`

Latest CUT_AND_FILL causal evidence:

- `R2M_GROSS_M3 = 4.7818756317`
- `M2P_GROSS_M3 = 0.0143007154`
- `PAYLOAD_SPILL_GROSS_M3 = 0`
- `MOUTH_CAPTURE_RATIO = 1.0`
- `GROSS_CAPTURE_RATIO = 0.002990608`
- `PRIMARY_BOTTLENECK = MOBILE_NOT_REACHING_MOUTH`

## KNOWN LIMITATIONS

- Iron-ore constitutive values remain literature-informed reduced-order values,
  not a completed site-specific calibration.
- The height-field/Mobile formulation is not a replacement for DEM, MPM, or
  full three-dimensional granular constitutive simulation.
- External 390F vehicle USD/controller assets are deployment dependencies and
  are not distributed in this Git baseline.
- Large raw checkpoints and runtime telemetry are excluded from ordinary Git;
  compact reports preserve the acceptance decisions.
- Isaac GUI/full-cycle behavior is not established by the pure-Python test
  suite alone.

## NEXT VALIDATION TARGET

Implement and validate one realistic 390F curl-scoop trajectory using
coordinated stick retraction and bucket curl without changing Mobile V2,
material parameters, the 0.02 m3 gate, or the 12 s timeout. Then rerun through
PRE_DUMP and classify the physical free-surface morphology before any dump or
Airborne acceptance.
