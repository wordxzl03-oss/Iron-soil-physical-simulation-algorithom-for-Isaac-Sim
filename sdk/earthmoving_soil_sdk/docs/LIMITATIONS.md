# Limitations

- `FLOW_ARREST_ROOT_CAUSE = MULTIPLE_WITH_BREAKDOWN`; ownership fixes exist,
  but a post-fix production `ARREST_FINAL` was externally interrupted. Status is
  `FLOW_ARREST_CLOSURE_NOT_YET_DEMONSTRATED`.
- Reference iron-ore parameters are literature-informed and
  `NOT_YET_PHYSICALLY_CALIBRATED`.
- TrackSoil is a conservative reduced-order rut model, not Bekker/Wong
  validation, and produces no track reaction wrench.
- No wheels, multi-tool physics, compaction/density evolution, DEM/MPM or full
  3-D granular claim.
- Isaac contact chunks are recooked at caller-selected events/rates and can lag
  the physics/visual surface. This lag is explicit rather than hidden.

