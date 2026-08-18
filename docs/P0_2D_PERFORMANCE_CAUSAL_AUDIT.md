# P0-2D performance causal audit

## Decision

```text
PERFORMANCE_REGRESSION_CONFIRMED = YES
HOTTEST_OPERATOR = Tool-Mobile CPU geometry support construction
HOTTEST_FUNCTION = build_tool_mobile_contact_support
EXECUTION_DEVICE = MIXED
ROOT_CAUSE_CLASS = CPU_PYTHON_EXHAUSTIVE_WET_CELL_X_CAD_TRIANGLE_CONTACT_SEARCH
PHYSICS_FIX_REQUIRED = NO
PERFORMANCE_ARCHITECTURE_FIX_REQUIRED = YES
```

No contact law, Mobile V2 equation, FailureSurface equation, material value,
trajectory, resolution, timestep, payload or conservation contract was
changed. No optimization is included in this audit.

## Real-host timing containment

The authoritative source is
`outputs/390f_v2/interactive_runs/run_1786952005/runtime_telemetry.json`.
The final recorded CUT_AND_FILL frame has:

| Region | Wall time |
|---|---:|
| bulk core | 48,726.294 ms |
| activation scatter envelope | 48,596.660 ms |
| existing FailureSurface/intersection envelope | 32.266 ms |
| complete CUDA Mobile V2 envelope | 49.170 ms |
| intake | 10.833 ms |
| LargeAvalanche | 12.700 ms |
| two normal scalar ledger snapshots | 12.180 ms |
| track/drive | 5.862 ms |
| PhysX/render | 1.267 ms |
| external heightfield contact update | 5.634 ms |

The activation envelope is 99.734% of `bulk_core`. Across the six recorded
CUT frames, its Pearson correlation with Mobile volume is 0.999772. In
contrast, the entire CUDA Mobile operator—including Tool-Mobile impulse work—
stays between 33.8 and 49.2 ms. PhysX, rendering, TrackSoil and the external
terrain contact updater are excluded as the dominant cause.

The old `activation_scatter` timer starts before Resting-to-Mobile commit and
ends after Tool-Mobile geometry construction, material-mask scatter and dirty
tile collection. It did not split those operations, so the report keeps the
48,596.660 ms value as a measured containment envelope rather than falsely
renaming it an exact contact timer.

## Exact production call chain

During every DIG frame the production path is:

```text
EarthmovingPhysicsCore._step_gpu
  -> ContinuousSweepBuilder.build
  -> DeviceFailureZoneBridge.execute
       -> DeviceBulkState.download_region
       -> ToolTerrainIntersectionModel.compute
       -> FailureZoneModel.compute
       -> DeviceBulkState.entrain_host_indices
       -> build_tool_mobile_contact_support       # dominant CPU path
       -> DeviceBulkState.apply_host_indices(material_mask)
  -> GpuBulkOperatorChain.step_mobile
       -> WarpProductionMobileV2Solver.step_resident
            -> compact contact H2D/scatter once per Isaac frame
            -> CFL loop
                 -> shared-face Mobile transport on CUDA
                 -> fused contact/local-source kernel on CUDA
  -> DeviceBucketIntakeBridge.execute
  -> MobileMomentumBudget / SoilForceModel
  -> scalar ledger snapshot
  -> DeviceLargeAvalancheBridge
  -> optional compact residual frontier
  -> scalar ledger snapshot
```

No deposition operator executes in the CUT_AND_FILL branch.

## Demonstrated complexity

The real bucket descriptor contains 36 vertices and 68 interior triangles.
`build_tool_mobile_contact_support` executes once per DIG Isaac frame on CPU:

1. `np.argwhere` enumerates every wet Mobile cell in the dynamically expanded
   FailureZone patch.
2. For every candidate, a serial Python loop calls `_triangle_box_overlap`
   against as many as all 68 triangles.
3. Regardless of the SAT result, `_point_inside_closed_mesh` loops over all 68
   triangles again.
4. Every accepted prism calls `_closest_point_on_triangle` for all 68 faces.

Therefore geometry is
`O(N_candidate * 68) + O(N_accept * 68)`. It is not
`O(N_candidate * 68 * N_substeps)`: geometry is built once per frame. The
accepted compact mask is reused during the CFL loop. The CUDA fused source
kernel itself is full-grid, `O(701*701*N_substeps)`, but its complete measured
frame envelope is only 49.170 ms at the worst recorded frame.

An audit-only CPU microbenchmark using the actual 390F 68-face descriptor
confirmed the source-level scaling:

| Wet candidates | Potential pairs | Exact tests | Total |
|---:|---:|---:|---:|
| 100 | 6,800 | 13,600 | 961.647 ms |
| 500 | 34,000 | 62,220 | 4,426.559 ms |
| 1,000 | 68,000 | 121,820 | 8,103.334 ms |

These measurements corroborate the identified CPU loop. They are not used to
back-calculate an invented last-frame candidate count.

## Static geometry and transfer audit

- CAD topology and descriptor are static, but transformed vertices, triangle
  gathering and face normals are rebuilt once per frame.
- No triangle-AABB cache, BVH, spatial index or cached closed-mesh query exists.
- SAT, ray/triangle and closest-point loops run in Python/NumPy despite the
  production Mobile solver using Warp CUDA.
- Geometry is not rebuilt per Mobile substep.
- The accepted compact contact arrays are uploaded once per frame and reused
  by every substep; they are not reuploaded per substep.

## Acceptance telemetry overhead

The three acceptance flags do not introduce a full-field D2H inside every CFL
substep. `cut-fill-payload-audit` downloads one compact approximately
4.1 m x 4.1 m bucket-neighborhood patch per observed CUT frame and consumes
returned compact/scalar objects; it executes outside `physics_core.step`, not
inside a Mobile substep. The PRE_DUMP full-field checkpoint is boundary-only
and was never reached. The dual-CV observer performs scalar reservoir
reductions after the Mobile and LargeAvalanche operator boundaries plus one
runner finish-step snapshot, never inside the CFL loop.

The old run did not time those two callbacks separately. Their strict combined
upper bound is 61.869 ms—the sum of the complete Mobile and LargeAvalanche
envelopes containing them—versus the 48,596.660 ms activation envelope. The
new `audit_readback_ms` timer isolates the in-core callbacks on the next run
without adding a full-field transfer. Separate launcher timers now isolate the
P0-2B finish-step ledger and P0-2D compact-patch audit.

## Instrumentation boundary

The code now records the requested per-frame phase timings and scalar counts:
candidate count, 68-face pair count, actual SAT tests, ray tests, exact tests,
closest queries, accepted contacts, global Mobile nonzero count, Mobile
substeps, compact H2D bytes/transfers, geometry phase times, fused CUDA source
upper bound and acceptance readback time.

For `run_1786952005`, candidate/test/substep counts at the final 8.0 s frame
remain `NOT_RECORDED_PRE_INSTRUMENTATION`. They are deliberately not inferred.
The last separately flushed contact audit sample is at 6.233333658 s and has
79 geometry-confirmed cells and five Mobile substeps; it is not mislabeled as
the 8.0 s frame.

## Stop boundary

The dominant hot path and its complexity are demonstrated. Per the task stop
rule, no BVH, caching, compact candidate list, Warp geometry kernel or other
optimization has been implemented yet.
