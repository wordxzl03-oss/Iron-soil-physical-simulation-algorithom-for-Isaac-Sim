# Phase C ContactView capability decision

## Decision

Isaac Sim 4.5 uses `TriangleMeshContactBackend` for the production Phase C
ContactView. `PhysXHeightFieldContactBackend` remains a guarded interface and
cannot be constructed until both public USD authoring and stable runtime update
probes pass.

The local Isaac Sim 4.5 runtime probe found no HeightField authoring symbol in
the public `pxr.UsdPhysics` or `pxr.PhysxSchema` modules. Symbol discovery would
not be sufficient by itself: a future backend also has to demonstrate a public,
repeatable update/recook path. No private PhysX interface is used as a fallback.

Runtime evidence is saved in:

- `outputs/phase_c_contact_runtime_00deg.json`
- `outputs/phase_c_contact_runtime_10deg.json`
- `outputs/phase_c_contact_runtime_20deg.json`

Each probe records `state: unavailable_public_api` and selects the triangle-mesh
backend.

## Public API path used

The Isaac adapter uses only schemas exposed after `SimulationApp` startup:

- `UsdGeom.Mesh.Define` for the hidden support surface;
- `UsdPhysics.CollisionAPI.Apply`;
- `UsdPhysics.MeshCollisionAPI.Apply` with approximation `none`;
- `UsdPhysics.CollisionGroup.Define`;
- `Usd.CollectionAPI.Apply(group_prim, "colliders")` for group membership;
- `CollisionGroup.CreateFilteredGroupsRel` in both directions for the
  `TerrainSupport` / `BucketInteraction` pair.

The contact mesh has no `RigidBodyAPI`, is authored invisible, and is a different
Prim from `TerrainVisual`. The visual Prim has no collision API and is excluded
from every collision group.

## Update policy

`update_from_heightmap` only stages NumPy buffers. It does not modify committed
USD or request a PhysX cook. Publication is explicit:

- default: once at `action_end`, after material interaction and MiniSlope;
- optional: `low_frequency`, rate-limited by simulation time and disabled unless
  selected in configuration;
- never: every physics frame.

The 0/10/20 degree probes validate deterministic slope construction, independent
0.05 m source / 0.10 m contact resolution, exact static-mesh schema authoring,
actual WheelLoader collider membership, and the symmetric bucket/terrain filter.
Vehicle drive, contact-load telemetry, and comparative speed/energy measurements
are intentionally exposed as a Phase B/D runtime handoff rather than fabricated
inside the contact authoring probe.
