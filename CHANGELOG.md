# Changelog

All notable baseline boundaries are recorded here. This project uses explicit
validation labels rather than treating an implementation milestone as proof of
full physical acceptance.

## [v0.1.0-mobile-v2-baseline] - 2026-08-17

### Added

- First formal, rollback-capable Git baseline for the 390F iron-ore-fines
  earthmoving simulator.
- Mobile V2 conservative finite-volume reference and production integration.
- Authoritative `z_base` / `b_eff` / `h_mobile` / momentum state hierarchy.
- Conservative entrainment, deposition, ownership, FailureZone,
  LargeAvalanche, Airborne, and residual MiniSlope contracts.
- Compact Gate A/B, frozen 5 s, and CUT_AND_FILL causal evidence.
- Repository-specific exclusions for runtime outputs, checkpoints, datasets,
  caches, recordings, and machine-local files.

### Accepted at this boundary

- FailureSurface V3.
- Mobile V2 numerical core.
- Production Gate A and frozen production Gate B.
- Machine-precision-scale conservation in the accepted frozen validations.

### Rejected historical approaches

- Mobile V1 centered gravity/pressure source.
- Velocity-selected face heuristic.

### Open after this boundary

- Realistic coordinated 390F scoop trajectory.
- Final tool-Mobile momentum coupling validation.
- PRE_DUMP morphology and visible needle/fin/grid-spike classification.
- Complete production digging cycle and GUI full-cycle validation.

This version must not be described as a fully calibrated or fully validated
production excavation system.
