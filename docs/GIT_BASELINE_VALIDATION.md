# Git Baseline Validation Audit

Audit date: 2026-08-17  
Baseline: `v0.1.0-mobile-v2-baseline`  
Local repository: `/home/eric/Desktop/mesh`

This audit records repository validation without changing production physics to
make the repository appear green.

## Safety and inventory

- The local directory was initially not a Git repository; the empty `.git`
  directory contained no history.
- `main` was established directly on the fetched `origin/main` initial commit.
- Candidate tracked files were scanned by filename and common credential
  signatures. No credentials or private keys were found.
- `.venv`, caches, raw `outputs`, datasets, checkpoints, recordings, training
  runs, handoff archives, and historical duplicated review bundles are ignored.
- Seven compact, named Mobile V2/CUT validation JSON summaries are retained.
- No production physics source was edited during this Git-baseline task.

## Complete test suite

Command:

```bash
.venv/bin/python -m pytest -q
```

Result in 18.29 s:

- 276 passed
- 3 failed
- 16 skipped
- 57 subtests passed

Known failures, preserved without production changes:

1. `test_phase_a_baseline.py::test_generator_is_deterministic_and_covers_reference`
   reports that the historical Phase-A manifest no longer equals the current
   generated source inventory.
2. `test_phase_a_baseline.py::test_validator_accepts_reference_and_rejects_manifest_tampering`
   reports a stale size/SHA-256 entry for
   `src/isaac_bulk_pipeline/runtime/__init__.py`.
3. `test_phase_h_soil_force.py::test_penetration_width_and_cohesion_increase_quasi_static_resistance`
   reports that the current narrow and wide cases have equal cutting
   resistance. SoilForce was not changed in this repository task.

The first two are historical-manifest drift. The third is an existing
SoilForce behavior/test disagreement and is not classified as an environment
failure. All remain explicit follow-up items; none is hidden by regenerating
evidence or tuning physics.

## Focused authoritative contracts

Command:

```bash
.venv/bin/python -m pytest -q \
  tests/test_device_bulk_state.py \
  tests/test_large_avalanche_transition.py \
  tests/test_avalanche_persistence.py \
  tests/test_bucket_geometry_intake.py \
  tests/test_bucket_internal_fill.py \
  tests/test_phase_e_bulk_state.py \
  tests/test_phase_f_bulk_interaction.py \
  tests/test_phase_g_spill_dump.py \
  tests/test_solver_conservation.py \
  tests/test_cut_fill_payload_audit.py
```

Result in 1.43 s: `67 passed, 9 skipped`.

## Mobile V2 acceptance tools

```bash
.venv/bin/python tools/run_mobile_v2_reference_contracts.py
.venv/bin/python tools/run_mobile_v2_frozen_5s.py
.venv/bin/python tools/run_mobile_v2_production_integration.py
```

Results:

- Mobile V2 reference contracts: `PASS` (10/10 cases), 0.17 s wall time.
- Frozen 5 s validation: `PASS`, maximum absolute mass error
  `1.7763568394e-15 m3`, positivity `true`, energy consistency `true`, 7.43 s.
- Production integration: Gate A `PASS`, frozen Gate B `PASS`, 5.55 s.

## Environment boundary

- Python reference environment: local project `.venv`.
- GPU validation used the local NVIDIA RTX 3070 Ti Laptop GPU through Warp.
- Isaac GUI/full-cycle validation was not run because it is outside this
  version-management task and requires the external 390F deployment asset.
- The CI workflow uses Python 3.12, whereas Isaac integration targets Python
  3.10; CI results may expose additional environment-specific differences.

This audit supports the Mobile V2 numerical/Gate A/B baseline only. It does not
change `PRE_DUMP_REACHED: NO` or any morphology `NOT_EVALUATED` status.
