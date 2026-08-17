# Real 390F bucket integration

## Audited asset

The authoritative source is
`/home/eric/桌面/bulldozer_sim/bulldozer_main.usd`, SHA-256
`db2bb54fd81b43c9ce57f8ab9ad7d7b07dbe81b6dbf4dbd9dbb321eabe747b7c`.
`tools/audit_real_390f_bucket.py` refuses to reuse the semantic vertex
annotations if this hash changes.

The read-only USD audit records the following facts in
`outputs/real_390f_bucket_audit.json`:

| Item | Audited value |
|---|---|
| Vehicle | tracked hydraulic excavator, not a wheel loader |
| Bucket rigid body | `.../tn__390F_DETAIL_DELIVERY_Bucket1_zf0` |
| Visual mesh | bucket child `Mesh`; 9,566 vertices; 10,086 triangles |
| Authored bucket mass | 6,000 kg |
| Bucket joint | parent Stick, child Bucket, revolute X, limits -120 to +60 degrees |
| Existing collision approximation | `convexHull` |
| Collision classification | `CURRENT_COLLISION_ONLY_NOT_SOIL_GEOMETRY` |
| Stage convention | Z-up, stage metres-per-unit 1.0; the CAD hierarchy carries a 0.001 ancestor scale |

Both the raw local and accumulated world transforms are retained in the audit.
The raw local translation is in the CAD hierarchy's authored coordinate scale;
the accumulated matrix is the metre-scale transform used during extraction.

## Three deliberately separate representations

```text
real 390F CAD
  +-- visual mesh: rendering only
  +-- PhysX collider: current rigid collision approximation
  `-- BucketGeometryDescriptor: custom bulk interaction geometry
```

The current convex hull closes the bucket concavity. It is never read to infer
the mouth, cutting edge, interior profile, retention volume, capacity or
failure geometry.

For later rigid-obstacle contact, a validated convex decomposition is the
preferred first candidate. An SDF can represent concavity more closely, but it
must first pass Isaac 4.5 scale, update-cost and contact-stability tests. Neither
candidate becomes the soil geometry.

The contact policy is:

```text
tracks/chassis <-> rigid ground: PhysX
bucket <-> heightmap bulk: custom interaction and SoilForce
```

Bucket collision against a second rigid representation of the same bulk must
be disabled while custom SoilForce is active. Otherwise PhysX reaction and
custom reaction are double counted. The current integration is interface-ready;
that collision-filter policy has not yet been closed-loop validated in Isaac.

## Semantic CAD extraction

CAD establishes positions and scale; audited semantic indices establish
meaning. The contract identifies left/centre/right cutting edge, left/right top
edge, left/right rear bottom, and an ordered concave side-interior profile. The
centre cutting marker is the mean of the two vertices beside the central tooth
gap. The side profile is not a convex-hull section.

The extractor defines Tool +X left-to-right along the edge, Tool +Y from rear
bottom toward the cutting tip, and Tool +Z by their right-handed cross product.
It produces:

| Quantity | Value |
|---|---:|
| Cutting-edge width | 2.742099 m |
| Interior width | 2.560000 m |
| Mouth area | 5.478292 m² |
| Concave profile area | 2.200547 m² |
| Reduced-order geometric capacity | 5.633400 m³ |

The capacity is the concave side-profile area times interior width. It is a CAD-
constrained reduced-order geometric volume, not a manufacturer-rated/heaped
capacity.

The generated descriptor records `geometry_source=USD_MESH_MARKERS`, has a
watertight consistently wound extruded interior, and preserves the exact
CAD-to-bucket-link transform. Runtime loaders use
`configs/excavator_390f_real_bucket.yaml`; the full 0.05 m project wiring is
`configs/project_25m_390f.yaml`.

## Reproduction

```bash
USD_LIB=/home/eric/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
PYTHONPATH="$PWD/src:$USD_LIB" \
LD_LIBRARY_PATH="$USD_LIB/bin" \
/home/eric/isaacsim/python.sh tools/audit_real_390f_bucket.py
```

The offline JSON can then be loaded with ordinary Python; Isaac and `pxr` are
only required to repeat the USD audit/extraction.

## Full articulation and runtime audit

The referenced articulation exposes exactly four revolute DOFs in the order
`swing_joint`, `boom_joint`, `stick_joint`, `bucket_joint`.  Its seven PhysX
bodies have authored masses 12, 12, 12, 40, 8, 4 and 6 tonnes (94 tonnes
total).  The source USD authors non-finite/zero COM and inertia placeholders;
the public Isaac `RigidPrim` runtime resolves finite mesh-derived COM and
inertia.  Those runtime values, not the invalid authored placeholders, are
recorded in `outputs/real_390f_runtime_probe.json`.

The 94 tonne source mass is above the manufacturer's published 390F L operating
mass range and is retained as an audited asset fact, not relabelled as a
validated mass configuration.  No cylinder pin coordinates were recovered
from a manufacturer source or unambiguous CAD semantics, so no cylinder-to-
joint Jacobian or fictitious lever arm is implemented.

The source bucket drive contains `targetVelocity=800`; the integrated runner
clears this value.  It commands the real articulation without per-cycle pose
setting or teleportation.  A causal command shaper limits target velocity and
acceleration, generalized effort and shared positive power.  Caterpillar's
published 260 kN m / 6.2 rpm swing values and 391 kW net engine power are used
as manufacturer bounds.  Front-equipment torque bounds are explicitly marked
engineering upper bounds because the cylinder pin geometry is missing.

Direct effort control repeatedly drove the source articulation into the bucket
hard limit and caused an Isaac 4.5 native PhysX crash.  The accepted numerical
fallback is a force-type PhysX position servo whose target is advanced only by
the bounded velocity/acceleration command and whose `maxForce` is updated from
the bounded actuator output.  This is not a hydraulic-cylinder validation.

The track contact-force sensor view also caused a reproducible Isaac 4.5 native
crash during the long integrated run.  Track and support collision schemas stay
enabled, but force-sensor evidence is reported `DEGRADED`, never inferred from
collision-schema presence.

The completed A/B/C, three-cycle runtime evidence and its explicit acceptance
boundary are documented in
[INTEGRATED_390F_EARTHMOVING_V1.md](INTEGRATED_390F_EARTHMOVING_V1.md).
