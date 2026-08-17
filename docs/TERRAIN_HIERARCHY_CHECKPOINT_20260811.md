# Terrain hierarchy checkpoint — 2026-08-11

## Outcome

The former blocking terrain path has been replaced. A deposition-triggered
connected failure is no longer sent through a minutes-long, converge-to-final
equilibrium MiniSlope call on the Isaac main loop.

The interactive hierarchy is now:

1. Airborne parcels land incrementally at each physics step.
2. A physical-state detector measures connected unstable extent, slope excess,
   mobilizable volume, persistence in simulation time, and current Mobile
   activity.
3. A sustained large event transfers a finite failure layer from Resting to
   Mobile exactly conservatively and initializes downhill momentum from the
   local gravity/friction state.
4. Mobile Layer evolves the event through finite physical time. The existing
   deposition operator returns slow material below the stop-angle criterion to
   Resting incrementally.
5. MiniSlope runs only after airborne and Mobile activity are quiet. It receives
   instability seeds and advances a bounded number of compact frontier rounds
   per Isaac step; the frontier remains free to propagate outside the seed
   region and across tiles until complete.
6. The task state may wait for `terrain_settled`, but the application continues
   stepping and rendering.

No path lowers the formal 0.05 m terrain resolution, truncates a physically
reachable frontier, or relaxes reservoir conservation.

## Physical transition, not an iteration heuristic

`LargeAvalancheTransitionController` has no MiniSlope iteration-count input.
Activation depends on:

- connected unstable cells and area;
- excess slope above the material start angle;
- mobilizable volume;
- persistence in seconds of simulation time;
- existing Mobile volume and velocity.

The legacy `large_avalanche_iteration_threshold` remains only an offline
MiniSlope diagnostic classification threshold. It is separate from
`numerical_safety_max_iterations`, and neither controls the Resting-to-Mobile
transition. Exceeding the numerical safety limit remains an explicit numerical
non-convergence failure; it is not a performance truncation.

The transition settings in `configs/390f_v2_interactive.yaml` and the Mobile
friction/start/stop sensitivity information in
`configs/literature/iron_ore_condition_scenarios.json` are explicitly labelled
`LITERATURE_BASED_REDUCED_ORDER_UNCALIBRATED`. They must not be presented as
site-calibrated iron-ore prediction.

## GPU backends

Three optional NVIDIA Warp implementations now exist:

- `WarpMobileLayerSolver`: resident height/momentum/work arrays and conservative
  donor-limited flux;
- `WarpCompactActiveEdgeOperator`: directly consumes the accepted
  `CompactActiveEdgeBatch` semantics;
- `WarpTrackSoilOperator`: resident rut/shoulder and conservative
  Resting-to-Mobile operations.

Missing Warp/CUDA is reported as `UNAVAILABLE`; CPU fallback is never reported
as `GPU_OPTIMIZED`. The operators report measured H2D/D2H bytes,
synchronizations, kernel launches, wall time, and resident array names.

The operator-level CUDA implementations are accepted. The current interactive
`BulkStateManager` still owns authoritative NumPy state, so the YAML runtime
switch deliberately remains `DISABLED_UNTIL_EQUIVALENCE_ACCEPTED`; claiming the
entire Isaac pipeline is GPU-resident would be false. A later integration must
move shared state ownership and the remaining CPU consumers across the same
device boundary rather than upload/download complete fields each frame.

## Acceptance results

`tools/run_terrain_hierarchy_acceptance.py` executes localized (129×129),
medium (351×351), and whole-pile (701×701) cases at 0.05 m. Each case reports
`CPU_REFERENCE`, `GPU_OPTIMIZED`, and `LARGE_AVALANCHE_MOBILE_PATH` separately.

The 2026-08-11 run passed all three paths for all cases. For the 701×701,
three-step finite-time comparison:

- CPU Mobile wall time: 1.0251 s, RTF 0.0488;
- GPU path wall time including initialization, acceptance downloads, and three
  external `nvidia-smi` samples: 0.1462 s, RTF 0.3421;
- sampled GPU utilization mean: 28.67%;
- final surface L-infinity error: 4.93e-5 m;
- final surface RMSE: 1.39e-7 m;
- GPU volume balance error: -2.87e-11 m³;
- momentum closure error: 0 kg·m/s;
- H2D/D2H: 19,656,040 / 11,794,008 bytes;
- synchronization count: 21.

The standalone resident-kernel benchmark excludes initialization and external
sampling overhead. Its 701×701 Mobile step is approximately 0.011 s versus
approximately 0.196 s for the CPU reference. It additionally covers the
compact-frontier and TrackSoil operators.

Run the unified acceptance with Isaac's shipped Warp:

```bash
PYTHONPATH=/home/eric/isaacsim/extscache/omni.warp.core-1.5.0+lx64:src:. \
  /home/eric/isaacsim/python.sh tools/run_terrain_hierarchy_acceptance.py --steps 3
```

The extension version path is installation-specific. Inside Kit, use normal
extension loading.

## Interrupted diagnostic

The stopped legacy run is preserved at
`outputs/390f_v2/interactive_runs/run_1786419940`. Its
`INTERRUPTED_NOT_ACCEPTED.json` states that the terrain is not an accepted
physics result. It is retained only as evidence that the old blocking
quasi-static scheduling was computationally infeasible.

## Verification

- Full CPU regression: `257 passed, 3 skipped, 57 subtests passed`.
- The three regular-environment skips are CUDA/Warp availability guards.
- Isaac Python/CUDA unified terrain hierarchy acceptance: `PASS`.
- Warp operator benchmark: all three operators pass localized, medium and
  701×701 whole-pile CPU-reference comparisons.
- Real Isaac Sim 4.5/PhysX headless startup smoke: five frames completed and
  shut down normally in `interactive_runs/run_1786422478`; no autonomous
  excavation or long terrain solve was started.

Primary artifacts:

- `outputs/390f_v2/terrain_hierarchy_acceptance.json`
- `outputs/390f_v2/warp_backend_benchmarks.json`
- `outputs/390f_v2/interactive_runs/run_1786419940/INTERRUPTED_NOT_ACCEPTED.json`

