# Isaac integration

`IsaacSoilAdapter` receives an existing Stage and existing RigidPrim-like body
bindings. It observes poses/velocities, calls `SoilPhysics.step`, optionally
applies the returned bucket wrench, and publishes terrain. It never creates or
closes `SimulationApp`, steps World, owns an articulation/controller, commands
joints/tracks, or writes a vehicle root pose.

Three surfaces are distinct:

- **Physics terrain:** authoritative `H_free = H_resting + h_mobile`, updated at
  production resolution and physics cadence.
- **Visual mesh:** fixed-topology USD chunks whose vertices come only from the
  same `H_free`. GPU mode downloads dirty tiles, not the full field per frame.
- **Collision/contact:** optional hidden PhysX triangle-mesh chunks derived from
  the same dirty samples. They recook only when the caller invokes sync, so
  contact may have explicit scheduling lag; zero-lag continuous synchronization
  is not claimed.

External vehicle USDs, controllers and articulation logic are intentionally
absent from this package.

