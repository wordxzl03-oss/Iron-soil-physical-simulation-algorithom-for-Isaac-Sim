# Warp GPU backends

The GPU implementation is optional and explicit. Importing the project does
not require Warp. If either the Python module or requested CUDA device is not
available, `probe_warp()` reports `UNAVAILABLE` with a reason and constructors
raise `WarpBackendUnavailable`; CPU work is never labelled as a GPU pass.

Implemented resident operators:

- `WarpMobileLayerSolver`: closed-boundary conservative Mobile Layer. Resting,
  mobile height, momentum, weights, and work arrays remain on the Warp device.
  Physics arrays are downloaded only through the explicit `download_result()`
  acceptance/debug boundary. CFL and conservation use small scalar reductions.
- `WarpCompactActiveEdgeOperator`: consumes the accepted
  `CompactActiveEdgeBatch` buffers unchanged. The CPU frontier controller still
  owns dynamic tile activation and numerical convergence classification.
- `WarpTrackSoilOperator`: performs rut sinkage, four-neighbour shoulder
  dilation, and equal-volume resting-to-mobile transfer on the device.

Every backend exposes `diagnostics()` with H2D bytes, D2H bytes, transfer
counts, synchronization count, launch count, kernel enqueue wall time, device,
Warp version, and resident array names.

Isaac Sim 4.5 ships Warp as an extension. A standalone shell must expose it:

```bash
export PYTHONPATH=/home/eric/isaacsim/extscache/omni.warp.core-1.5.0+lx64:src:.
/home/eric/isaacsim/python.sh -m pytest -q tests/test_warp_gpu_backends.py
```

Isaac's embedded Python may not include pytest. The standalone benchmark has
no pytest dependency and covers localized, medium, and 701x701 whole-pile
cases against the CPU reference:

```bash
PYTHONPATH=/home/eric/isaacsim/extscache/omni.warp.core-1.5.0+lx64:src:. \
  /home/eric/isaacsim/python.sh tools/run_warp_backend_benchmarks.py
```

The strict three-path hierarchy acceptance additionally samples GPU
utilization externally and reports CPU reference, GPU optimized, and physical
large-avalanche paths separately:

```bash
PYTHONPATH=/home/eric/isaacsim/extscache/omni.warp.core-1.5.0+lx64:src:. \
  /home/eric/isaacsim/python.sh tools/run_terrain_hierarchy_acceptance.py --steps 3
```

The extension version path is installation-specific. Inside a running Isaac
Kit application, use its normal extension loading instead of hard-coding this
path.

Current scope restrictions are deliberate and visible:

- Mobile open-boundary outflow is not yet implemented by the Warp backend.
- Compact MiniSlope is a kernel primitive, not a replacement propagation law.
- Site-calibrated iron-ore TrackSoil parameters remain unavailable; the
  existing reduced-order parameter-status string is carried into GPU reports.
