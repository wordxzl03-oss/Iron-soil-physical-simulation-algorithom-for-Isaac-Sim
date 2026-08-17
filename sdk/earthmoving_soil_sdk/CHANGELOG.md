# Changelog

## 0.3.0-alpha.1

- Packages the production Failure Surface V3 implementation without equation changes.
- Preserves localized momentum coupling and cohesive-yield ownership corrections.
- Defines the authoritative free surface as `H_free = H_resting + h_mobile`.
- Exposes bucket force/torque, payload mass/volume, terrain and Mobile feedback for RL.
- Exposes the existing conservative TrackSoil rut/sinkage path and its current gaps.
- Preserves the GPU `DEVICE` authority and compact dirty-tile publication path.
- Adds a passive Isaac adapter and production visual terrain synchronization.
- Known limitation: `FLOW_ARREST_CLOSURE_NOT_YET_DEMONSTRATED`.
- Material profile remains `NOT_YET_PHYSICALLY_CALIBRATED`.

