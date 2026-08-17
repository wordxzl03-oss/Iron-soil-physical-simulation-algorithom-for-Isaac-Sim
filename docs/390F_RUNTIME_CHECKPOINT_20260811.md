# 390F Runtime Checkpoint — 2026-08-11

> Superseded for current runtime status by
> `docs/TERRAIN_HIERARCHY_CHECKPOINT_20260811.md`. The CUDA reboot blocker
> recorded below was historical: Isaac Sim 4.5, PhysX, Warp 1.5 and a five-frame
> headless startup smoke all run normally now. The interrupted long terrain run
> remains unaccepted.

This checkpoint is intentionally not a final three-cycle acceptance report.
No interrupted or CUDA-failed run is counted as a completed physical cycle.

## Completed corrections

- Preserved the correctness-approved CPU compact tile-frontier MiniSlope
  backend and its unrestricted dynamically expanding propagation.
- Corrected the dump-phase zero-release latch: a dry retention evaluation no
  longer permanently marks the payload as released.
- Corrected the real CAD bucket polarity. The accepted reachable breakout,
  lift and swing poses retain approximately 5.63 m3; the dump pose has zero
  retainable capacity. The no-soil arm/track acceptance remains PASS with no
  root or link pose writes.
- Changed the dump swing from 65 degrees to 15 degrees. The measured result is
  five of five airborne parcels inside the authoritative 701 x 701 heightmap;
  the former pose placed four of five parcels outside its left boundary.
- Added an immediate `DUMP_PARCEL_LEFT_TERRAIN_DOMAIN` failure instead of
  allowing out-of-domain parcels to fall until a generic deposition timeout.
- Preserved the DUMP-entry deposition baseline and counted deposition from the
  first actual release frame.
- Separated dump-origin MiniSlope seeds from unrelated historical terrain
  changes. Seeds now come from the authoritative ballistic landing points;
  they remain seeds only, and unstable-frontier propagation is not bounded by
  their extent.

## Latest valid physical evidence before the GPU fault

Run `run_1786417189` reached `BUCKET_RECOVERY` through observation-based gates:

- admitted and released payload: 0.1880513088 m3;
- payload after release: approximately zero;
- airborne parcels: 5/5 inside the heightmap;
- root/link pose writes: zero;
- transition sequence reached DUMP, DEPOSITION and BUCKET_RECOVERY.

That diagnostic run was stopped during an incorrectly over-seeded MiniSlope
event and is not accepted as a complete cycle. Later runs used progressively
more precise event seeds, but the final verification was pre-empted by the GPU
failure below.

## Current hard blocker

Isaac/PhysX reported CUDA 719 followed by CUDA 999. Kernel logs contain NVRM
Xid 31 MMU faults and Xid 154 with `Node Reboot Required`; the NVIDIA UVM
driver explicitly reports a global fatal error requiring an OS reboot.
`nvidia-smi` remaining readable is not sufficient evidence that CUDA or PhysX
is usable.

The machine was not rebooted automatically because that would interrupt the
user's desktop session. After reboot, rerun:

```bash
cd /home/eric/Desktop/mesh
nvidia-smi
/home/eric/isaacsim/python.sh isaac_loader/audit_390f_no_soil.py \
  --config configs/390f_v2_interactive.yaml --headless
./run_390f_interactive.sh --headless --acceptance-cycles 1
```

The one-cycle run must reach `READY_NEXT_CYCLE` before a three-cycle acceptance
is attempted. CPU real-time feasibility and GPU/Warp soil acceleration remain
not passed.
