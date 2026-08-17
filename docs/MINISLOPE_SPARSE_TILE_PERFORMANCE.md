# MiniSlope compact tile-frontier performance

Date: 2026-08-10

## Backend status

`EventDrivenMinimumSlopeAdapter` remains the correctness-approved CPU baseline.
It dynamically discovers reachable cells but evaluates vector phases over the
overall reached bounding rectangle. Its tile state does not drive numerical
work, so it is explicitly named `CPU_OPTIMIZED_REACHABLE_BBOX_BASELINE`.

`SparseTileFrontierMinimumSlopeAdapter` is the separate performance-stage
implementation. Its numerical loop consumes:

- compact active tile IDs;
- unique edge-owner buffers for one direction and checkerboard phase;
- only edges owned by active tiles plus transfer-endpoint stencil activation.

Stable tiles are retired after the post-round instability scan. A later
transfer touching their stencil can reactivate them. The same contiguous
`CompactActiveEdgeBatch` buffers are exposed through `kernel_buffers()` for a
future Warp/GPU backend; no growing bounding rectangle is part of that API.

The CPU implementation is a correctness/architecture baseline for compact
frontiers, not yet a claim that NumPy indexed edge lists beat dense rectangular
vectorization in every case.

## Independent limits

Two different configuration fields are enforced:

- `large_avalanche_iteration_threshold`: diagnostic classification only;
  crossing it produces `LARGE_SUSTAINED_AVALANCHE` and never truncates physics.
- `numerical_safety_max_iterations`: numerical guard. Crossing it raises an
  explicit `NUMERICAL_NONCONVERGENCE` failure and does not return a partially
  relaxed terrain as a valid result.

## 701 x 701 benchmark

Command:

```bash
cd /home/eric/Desktop/mesh
PYTHONPATH=.:src .venv/bin/python \
  tools/run_minislope_sparse_701_benchmark.py
```

Machine-readable output:

`outputs/390f_v2/minislope_sparse_701_benchmark.json`

The benchmark records active cells, ever-active tiles, reached bounding-box
cells, iterations, wall time, volume error, final-height error, slope
quantiles, and changed-region extent against the full-domain CPU reference.

Measured on the current 20-logical-core target machine with tile size 32:

| Case / backend | Active cells | Active tiles | BBox cells | Iterations | Wall time |
|---|---:|---:|---:|---:|---:|
| Near-critical toe / full reference | 491401 | n/a | 491401 | 183 | 12.94 s |
| Near-critical toe / reached bbox baseline | 117 | 2 | 132 | 183 | 0.123 s |
| Near-critical toe / compact tile CPU | 117 | 2 | 132 | 182 | 0.262 s |
| Whole-pile / full reference | 491401 | n/a | 491401 | 2 | 0.147 s |
| Whole-pile / reached bbox baseline | 491401 | 484 | 491401 | 2 | 0.288 s |
| Whole-pile / compact tile CPU | 491401 | 484 | 491401 | 1 | 1.038 s |

The compact backend matches the reference within the configured numerical
acceptance (`height L-inf <= 2e-8 m`, volume error `<= 1e-8 m3`, slope-quantile
L-inf `<= 2e-8`, equal changed-region extent). In the localized toe case its
height L-inf difference is about `1.28e-9 m`; the whole-pile case is exact.

The table also exposes an important performance result: on CPU, compact NumPy
gather/scatter and list management are currently slower than the already-small
132-cell dense bbox. For a whole-domain event, compact lists correctly
degenerate to all 484 tiles and are slower than contiguous full-array NumPy.
No speedup is claimed for those two comparisons. The compact representation is
valuable as the honest sparse work list and future GPU/Warp input; subsequent
optimization must benchmark kernel residency and transfer cost rather than
assuming sparse indexing is automatically faster.
