# 390F Iron-Ore Earthmoving Simulator

Real-time reduced-order iron-ore-fines earthmoving simulation for a
390F-class tracked hydraulic excavator in NVIDIA Isaac Sim 4.5.

This repository is the first auditable production baseline for the terrain and
bulk-material physics stack. It includes validated numerical components and
their tests, but it is **not** a claim of completed field calibration, a fully
validated excavation cycle, or DEM-equivalent constitutive fidelity.

## Architecture

```text
Resting terrain (z_base, b_eff)
        -> FailureSurface V3 / FailureZone activation
        -> Mobile V2 granular layer (h_mobile, momentum)
        -> bucket mouth Intake / Payload
        -> Spill / Airborne / Deposition
        -> authoritative free surface H_free = b_eff + h_mobile
        <-> excavator dynamics / SoilForce
```

The production GPU path keeps authoritative terrain state on DEVICE and uses
dirty-region publication for the Isaac visual terrain. External vehicle USD
assets and machine controllers remain deployment inputs; they are not bundled
as large binary assets in this Git baseline.

## Current numerical core

Mobile V2 provides:

- conservative finite-volume transport;
- generalized well-balanced reconstruction;
- two-state shared-face fluxes;
- a first-order Rusanov reference flux;
- positivity-compatible CFL substepping;
- authoritative `z_base`, `b_eff`, `h_mobile`, and horizontal momentum fields;
- explicit conservative entrainment and deposition contracts.

The centered Mobile V1 gravity/pressure source and the later
velocity-selected face heuristic are retained only as rejected historical
approaches; they are not the production numerical core.

## Validation status

Accepted in this baseline:

- FailureSurface V3 and FailureZone activation semantics;
- Mobile V2 reference contracts and frozen 5 s energy/conservation gate;
- Mobile V2 production integration Gate A and frozen Gate B;
- authoritative `b_eff` state migration and checkpoint persistence;
- conservative entrainment, deposition, Airborne-to-V2, ownership,
  FailureZone-to-V2, and LargeAvalanche-to-V2 contracts;
- MiniSlope residual/static semantics;
- 701 x 701 terrain at 0.05 m spacing with machine-precision-scale mass error.

Still open:

- a realistic coordinated stick-retraction plus bucket-curl 390F scoop;
- final tool-to-Mobile momentum coupling validation;
- reaching and evaluating PRE_DUMP morphology;
- needle-forest, triangular-fin, and grid-spike visual classification;
- complete production digging-cycle and GUI full-cycle validation;
- iron-ore parameter calibration against dedicated field measurements.

The latest CUT_AND_FILL audit found
`PRIMARY_BOTTLENECK=MOBILE_NOT_REACHING_MOUTH`: 4.78187563 m3 gross R2M
produced 0.014300715 m3 gross M2P, with a mouth capture ratio of 1.0 and no
Payload spill. This is an open trajectory/transport validation issue, not an
accepted complete-cycle result.

See [CURRENT_BASELINE.md](CURRENT_BASELINE.md) for the authoritative status and
[CHANGELOG.md](CHANGELOG.md) for the baseline version boundary.

## Repository layout

- `src/isaac_bulk_pipeline/`: production and experimental soil-physics code;
- `isaac_loader/`: Isaac Sim 390F runtime integration;
- `configs/`: production and literature-traceable configurations;
- `tests/`: pure Python and runtime contract tests;
- `tools/`: validation, audit, benchmark, and checkpoint utilities;
- `sdk/earthmoving_soil_sdk/`: non-destructive alpha SDK extraction;
- `docs/`: architecture, theory traces, validation reports, and limitations;
- `outputs/mobile_v2_*/*.json`: deliberately selected compact evidence only.

Raw runtime outputs, checkpoints, videos, caches, virtual environments,
training runs, datasets, and duplicated handoff bundles are intentionally not
tracked by ordinary Git.

## Development setup

Python 3.10 is the reference interpreter for Isaac Sim 4.5 integration.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run the repository test suite:

```bash
.venv/bin/python -m pytest -q
```

Run the focused Mobile V2 baseline checks:

```bash
.venv/bin/python tools/run_mobile_v2_reference_contracts.py
.venv/bin/python tools/run_mobile_v2_frozen_5s.py
.venv/bin/python tools/run_mobile_v2_production_integration.py
```

Isaac-dependent commands require a local Isaac Sim 4.5 installation and the
external 390F USD configured for that workstation. Do not commit machine-local
credentials, Isaac caches, or proprietary vehicle assets.

## Reproducibility boundary

Compact validation summaries are versioned. Large `.npz` checkpoints and raw
telemetry are excluded from normal Git and should be retained in controlled
artifact storage with hashes. The baseline tag does not imply that excluded
assets are publicly redistributable.

## License and citation

See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
