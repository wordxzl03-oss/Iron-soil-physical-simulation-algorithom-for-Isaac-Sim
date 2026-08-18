# P0-2D GPU-resident exact Tool–Mobile contact optimization

## Result

`GPU_GEOMETRY_IMPLEMENTED = YES`.

The production `GPU_RUNTIME / DEVICE` path no longer calls the serial
`build_tool_mobile_contact_support()` Python candidate×triangle loop.  That
function remains unchanged as the deterministic CPU oracle.  The new Warp
operator preserves the accepted geometry and impulse semantics; no physics,
material, trajectory, timestep, grid, FailureSurface V3, or Mobile V2 equation
was changed.

The formal real-host CUDA replay was not run in this execution environment:
Warp 1.5.0 reports `WARP_DEVICE_UNAVAILABLE:cuda:0`.  Therefore CUDA wall time,
production RTF, and the formal `GPU_CONTACT_GEOMETRY_NO_LONGER_DOMINANT` gate
remain unclaimed rather than substituting Warp-CPU numbers.

## Architecture change

Before:

```text
DEVICE Mobile patch
  -> compact D2H
  -> np.argwhere(wet)
  -> Python wet-cell loop
  -> Python 68-triangle exact loop
  -> host compact contact arrays
  -> H2D scatter
  -> fused Mobile V2 contact/source kernel
```

After:

```text
persistent DEVICE local vertices (36) + faces (68)
  -> one rigid-vertex Warp transform per frame
  -> DEVICE wet-cell candidate test in physical FailureZone bbox
  -> exact 68-triangle SAT + closed-cavity ray parity
  -> exact closest CAD point + tool->Mobile normal
  -> v_linear + omega x (x_contact - x_tool_origin)
  -> persistent DEVICE Mobile V2 contact fields
  -> existing fused Mobile V2 contact/source kernel
```

The FailureZone bbox is only a physical broad spatial restriction.  A cell is
accepted only by the exact CAD prism-overlap or closed-cavity containment
test.  `material_mask` is an output and is not used as geometric proof.

Static local vertices and face topology are uploaded once when the production
operator chain is built.  Per-frame contact H2D is zero.  Only two small
diagnostic arrays are read to host; full contact fields remain on DEVICE.
Compaction time is zero because the already accepted fused Mobile kernel
directly consumes the resident full-grid contact mask/fields.

## Exact geometry semantics retained

- Same valid CAD face filtering and static topology.
- Same 13-axis triangle/AABB SAT with `1e-12` tolerance.
- Same non-axis-aligned odd/even ray test and `1e-9` co-planar hit de-duplication.
- Same Ericson closest-point regions and first-face tie ordering.
- Same inside/outside signed distance and face-normal fallback.
- Same projected horizontal tool-to-Mobile normal consumed by Mobile V2.
- Same rigid surface velocity definition.
- Same Triangle-A-C dual-control area for contact volume/mass.
- Same normal impulse and maximum-dissipation Coulomb law.

## Production integration

`GpuBulkOperatorChain` creates one `WarpExactToolMobileContactGeometry` through
the production `DeviceFailureZoneBridge`.  The bridge performs unchanged
FailureSurface computation, commits Resting→Mobile, invokes exact DEVICE
contact geometry, and returns `DeviceToolMobileContactSupport`.  The existing
`WarpProductionMobileV2Solver` recognizes that handle by runtime identity and
consumes the already resident arrays without zero/upload/scatter.  The old
host support branch remains available only for reference callers.

## Equivalence and conservation

Isaac's Warp 1.5 CPU device compiled and executed the same kernels.  Twenty-five
focused tests passed:

- 10 required exact geometry/impulse categories: no contact, triangle
  intersection, closed-cavity containment, edge contact, vertex-near contact,
  multiple candidates, unequal Triangle-A-C areas, moving/rotating bucket,
  separating contact, and tangential sliding;
- production FailureZone bridge returns resident support with zero contact H2D;
- prior frictional coupling and momentum-contract regressions.

Compared quantities include accept/reject, flat cell identity, closest point,
published 3-D and projected normals, signed distance, rigid surface velocity,
Mobile volume, normal impulse, and tangential impulse.  The direct DEVICE
Mobile test also verifies zero compact-contact H2D, mass residual within
`2e-12 m3`, Coulomb bound, non-positive unexplained contact energy within
numerical tolerance, and exact `Mobile +J / machine -J` bookkeeping.

## 701×701 actual 390F CAD benchmark

The benchmark uses `dx=dy=0.05 m`, the actual descriptor's 36 vertices and 68
triangles, and identical deterministic candidates for both implementations.
These are Warp-CPU architecture measurements, not CUDA claims.

| wet candidates | accepts | Python CPU oracle | compiled Warp CPU total | reduction | equivalence |
|---:|---:|---:|---:|---:|:---:|
| 100 | 55 | 1133.012 ms | 0.812 ms | 1395.3× | PASS |
| 500 | 317 | 5451.213 ms | 2.352 ms | 2317.4× | PASS |
| 1000 | 632 | 10714.800 ms | 4.279 ms | 2504.0× | PASS |

For 1000 candidates the new kernel recorded 68,000 exact triangle/AABB tests,
1000 containment tests and 57,460 closest-point queries.  Per-frame contact
H2D was 0 bytes and geometry performed no full-field D2H.  The acceptance-only
oracle download is explicitly excluded from production transfer accounting.

The prior demonstrated real-host frame remains the formal baseline:

- old activation/contact envelope: `48,596.660 ms`;
- existing CUDA Mobile envelope: `49.170 ms`;
- old production RTF: `0.000342`.

Because this sandbox has no CUDA device, `NEW_GPU_GEOMETRY_MS`, CUDA speedup,
new production `bulk_core_ms`, RTF and pre-dump status are `NOT_RUN`/null.

## Instrumentation

Production timing/counter records now expose:

- candidate build, CAD transform, GPU geometry, direct-mask compaction,
  contact H2D, Mobile transport/source, fused Tool–Mobile impulse upper bound,
  audit readback, bulk core and total frame envelopes;
- wet candidates, geometry candidates, CAD triangles, triangle/AABB tests,
  ray tests, containment tests, closest-point queries, accepted contacts, and
  Mobile substeps.

## Formal status

```text
GPU_GEOMETRY_IMPLEMENTED: YES
PHYSICS_CHANGED: NO
CONTACT_SEMANTICS_CHANGED: NO
CPU_REFERENCE_EQUIVALENCE: PASS
ACTION_REACTION: PASS
MASS_CONSERVATION: PASS
OLD_CPU_GEOMETRY_MS: 48596.660 (old production envelope)
NEW_GPU_GEOMETRY_MS: NOT_RUN
SPEEDUP: NOT_RUN_FOR_CUDA
BULK_CORE_MS_SYNTHETIC_OR_REPLAY: NOT_RUN_FOR_CUDA
GPU_CONTACT_GEOMETRY_NO_LONGER_DOMINANT: NO (not yet demonstrated on CUDA)
FORMAL_390F_CUDA_REPLAY: NOT_RUN
PRE_DUMP_REACHED: NOT_RUN
RTF: NOT_RUN
PRIMARY_REMAINING_BLOCKER: CUDA device unavailable in this execution environment
```

No BVH was added.  The exact exhaustive-68 implementation is the required
Stage 1; any Stage 2 acceleration remains conditional on a real CUDA profile.

Machine-readable evidence:
`outputs/mobile_v2_production/p0_2d_gpu_contact_geometry_benchmark.json`.
