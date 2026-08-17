# Minslope 0.1.0 Stage Development Report

## 1. Executive summary

- Stage objective: build a reproducible reduced-order wheel-loader excavation
  simulator, SAC baseline, data-scale study, large-pile rollout, and animation pipeline.
- Outcome: source, 40 unit tests, nine verified final policies, experiment summaries,
  a 309-scoop rollout below 10% remaining volume, and a continuous MP4 replay.
- Delivery date: 2026-07-28.
- Completion status: stage complete; research limitations remain.

## 2. Scope

### Included

- Height-field pile generation and volume-conserving slope relaxation.
- Reduced-order resistance, bucket trajectory, vehicle, and terrain-contact models.
- OBJ geometry integration.
- Gymnasium environments and Stable-Baselines3 SAC training/evaluation.
- Data-scale experiments at 1k, 3k, 6k, and 10k steps.
- Random large-pile selection, supervised rollout below 10%, and continuous replay.
- Tests, reports, reproducibility commands, selected final weights, and examples.

### Excluded

- Full DEM validation, field calibration, production safety certification, and real-machine
  control.
- The locally stored third-party research paper; it is not redistributed.
- Virtual environments, caches, intermediate checkpoints, duplicate/smoke policies,
  bulk validation datasets, and failed render logs.
- Direct GitHub publication; this delivery prepares files but does not create a remote
  repository or release.

## 3. Repository and environment

| Item | Value |
|---|---|
| Project root | `minslope` |
| Version-control revision | unavailable: source directory was not a Git repository at packaging time |
| Runtime | CPython 3.12 |
| Training framework | PyTorch 2.13.0+cu130; Stable-Baselines3 |
| Training hardware | NVIDIA GeForce RTX 5080 Laptop GPU |
| Packaging OS | Windows, Asia/Shanghai |
| Package version | 0.1.0 |

## 4. Architecture and implementation

### Components

| Component | Responsibility |
|---|---|
| `slope_model.py` | pile generation, scooping, slope relaxation, avalanche updates |
| `physics_aware_trajectory.py` | material, machine limits, resistance-aware trajectory |
| `wheel_loader_dynamics.py` | low-speed dynamics and four-wheel terrain fitting |
| `obj_scooping_env.py` | OBJ-derived bucket geometry and capacity |
| `dynamic_obj_scooping_env.py` | vehicle/soil trajectory integration |
| `large_pile_excavation_env.py` | repeated excavation and stopping criteria |
| `wheel_loader_rl/` | reusable configuration, algorithms, environments, evaluation, CLI |
| `run_data_ablation.py` | resumable equal-configuration data-scale experiment |
| `run_best_model_excavation.py` | candidate-pile selection and supervised rollout |
| `render_large_pile_long_animation.py` | continuous terrain and loader replay |

### Important changes in this stage

- Added `--resume` support to the data-scale experiment.
- Added consistent 10,000-step training statistics, including 42 completed training piles
  and 9,782 transitions.
- Added configurable target remaining fraction and peak-height range.
- Added deterministic selection of the largest initial-volume pile among random candidates.
- Added continuous per-scoop state sampling and H.264 output.
- Added GitHub-oriented packaging, documentation, CI, model-zoo, and release manifests.

### Design decisions

- Represent soil as a grid height field for deterministic, inexpensive experiments.
- Use reduced-order physics and four-wheel terrain fitting instead of full multibody/DEM.
- Use SAC for continuous high-level trajectory parameters.
- Keep maximum-scoop candidate supervision explicit; do not present it as direct-policy SAC.
- Publish large weights and demonstrations as Release assets rather than bloating Git history.

## 5. Experiments

| Experiment | Model/config | Input/seed | Output | Status |
|---|---|---|---|---|
| 1k data scale | SAC, fixed config | eval seeds 20272000–20272007 | `ablation/data_scale/steps_001000/metrics.json` | complete |
| 3k data scale | SAC, fixed config | same evaluation set | `ablation/data_scale/steps_003000/metrics.json` | complete |
| 6k data scale | SAC, fixed config | same evaluation set | `ablation/data_scale/steps_006000/metrics.json` | complete |
| 10k data scale | SAC, fixed config | same evaluation set | `ablation/data_scale/steps_010000/metrics.json` | complete |
| 100-pile validation | SAC v2 + supervisor | seed 20261201, 100 piles | `large_pile_validation_100/dataset_summary.json` | complete, bulk data excluded |
| 10% demonstration | 10k SAC + 24 candidates | selection seed 2026072801 | `best_model_large_pile_10pct/summary.json` | complete |

## 6. Results

### Data-scale study

| Steps | Training piles | Training trajectories | Mean load (m³) | Payload efficiency | Mean remaining | Evidence |
|---:|---:|---:|---:|---:|---:|---|
| 1,000 | 4 | 997 | 0.4452 | 0.1484 | 78.52% | measured: `steps_001000/metrics.json` |
| 3,000 | 12 | 2,855 | 0.6551 | 0.2184 | 68.41% | measured: `steps_003000/metrics.json` |
| 6,000 | 25 | 5,975 | 0.6949 | 0.2316 | 66.49% | measured: `steps_006000/metrics.json` |
| 10,000 | 42 | 9,782 | 0.8120 | 0.2707 | 60.84% | measured: `steps_010000/metrics.json` |

All direct-policy success rates were 0% under the 20% remaining termination threshold.
The 10k model was selected as the best of these four because it had the highest mean
payload and lowest mean remaining fraction on the same eight evaluation piles.

### Large-pile validation and demonstration

| Metric | Value | Evidence type and file |
|---|---:|---|
| 100-pile validation successes | 100/100 | measured: `large_pile_validation_100/dataset_summary.json` |
| 100-pile mean remaining | 19.8284% | measured: same file |
| Selected pile peak | 23.5912 m | measured: `best_model_large_pile_10pct/summary.json` |
| Initial volume | 681.8121 m³ | measured: same file |
| Final volume | 67.7221 m³ | measured: same file |
| Final remaining | 9.9327% | measured: JSON and NPZ agree |
| Scoops | 309 | measured: same file |
| Mean payload | 1.9873 m³ | measured: same file |
| Continuous video | 5,562 frames, 18 FPS, 309 s | measured: `excavation_continuous.json` |

The 10% demonstration used a 24-candidate maximum-scoop supervisor around the learned
policy. It demonstrates the integrated planning and execution pipeline, not the unsupervised
policy success rate.

## 7. Verification

| Check | Command/method | Result |
|---|---|---|
| Unit tests | `python -m unittest discover -v` | 40/40 passed in 1.274 s |
| Final policies | `SAC.load(..., device="cpu")` | 9/9 loaded |
| Experiment JSON | Python `json.loads` | 8/8 selected files parsed |
| Rollout NPZ | NumPy load and shape check | actions `(309, 7)`, final 0.09932668 |
| Continuous video | render metadata and extracted frame | 5,562 frames, preview readable |
| Secret scan | filename and text-pattern scan | no candidates found |

## 8. Known limitations and failures

- Height-field soil cannot represent overhangs or individual particles.
- Physics parameters have not been calibrated against field measurements.
- The direct SAC policies did not reach the 20% threshold in the fixed eight-pile study.
- Candidate-action supervision materially improves rollout behavior and must be disclosed.
- The original continuous render attempt failed because H.264 `yuv420p` requires even
  dimensions; the final renderer pads 819×364 to 820×364.
- Windows Smart App Control previously blocked unsigned PyTorch DLLs; the user explicitly
  disabled it. Other systems may require a different PyTorch installation method.

## 9. Reproduction

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### Test

```powershell
python -m unittest discover -v
```

### Resume the data-scale study

```powershell
python run_data_ablation.py `
  --budgets 1000 3000 6000 10000 `
  --eval-piles 8 --device auto `
  --output ablation/data_scale --resume
```

### Reproduce the selected large-pile rollout

```powershell
python run_best_model_excavation.py `
  --model ablation/data_scale/steps_010000/policy.zip `
  --output best_model_large_pile_10pct `
  --seed 2026072801 --candidates 16 `
  --candidate-actions 24 --target 0.10 `
  --peak-min 23.5 --peak-max 24.0
```

Completion criteria:

- `remaining_fraction < 0.10`;
- `terminated == true`;
- `actions.shape == (scoop_count, 7)`;
- JSON and NPZ final remaining values agree.

## 10. Artifact map

| Relative path | Purpose | Package |
|---|---|---|
| `README.md`, `docs/` | public documentation | source |
| `wheel_loader_rl/`, root `*.py` | software and examples | source |
| `test_*.py` | verification | source |
| `simple_wheel_loader.obj`, `wheel_buck.obj` | loader geometry | source |
| `ablation/data_scale/*/metrics.json` | data-scale evidence | source |
| `ablation/data_scale/*/policy.zip` | data-scale policies | assets |
| `rl_runs/**/final.zip` | task-specific policies | assets |
| `best_model_large_pile_10pct/*.json`, `*.npz` | rollout evidence | assets |
| `best_model_large_pile_10pct/excavation_continuous.mp4` | full example | assets |

## 11. Packaging

- Source archive: produced from `release_manifests/github-source.txt`.
- Release asset archive: produced from `release_manifests/release-assets.txt`.
- Both ZIPs contain file-level `SHA256SUMS.txt`; external checksums are generated alongside.
- Excluded categories: secrets, environments, caches, Git internals, third-party PDF,
  intermediate checkpoints, duplicates, smoke runs, bulk validation directories, and logs.

## 12. Next stage

- Calibrate material and resistance parameters against measured piles.
- Improve direct-policy success without candidate supervision.
- Add training checkpoints and optimizer/replay-buffer resume.
- Add Linux GPU CI or a small deterministic CPU training smoke test.
- Replace the placeholder repository URL in `CITATION.cff`.
