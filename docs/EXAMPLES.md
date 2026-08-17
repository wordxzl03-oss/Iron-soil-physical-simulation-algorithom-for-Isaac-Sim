# Examples

## Lightweight examples in the source archive

### Critical-slope soil relaxation

```powershell
python slope_model.py --no-show
```

### Randomized loader trajectory

```powershell
python animate_loader_3d.py `
  --random --domain-randomize --trajectory-seed 9
```

### Trained single-scoop replay

After extracting Release weights:

```powershell
python render_trained_scoop.py `
  --model rl_runs/sac_scooping_baseline/scooping_sac_final.zip `
  --seed 20260808
```

The source archive includes a small result JSON/GIF pair from this example.

## Full large-pile example in the Release asset

Files:

- `best_model_large_pile_10pct/summary.json`
- `best_model_large_pile_10pct/excavation_data.npz`
- `best_model_large_pile_10pct/excavation_continuous.mp4`
- `best_model_large_pile_10pct/excavation_continuous.json`

Measured result:

- selected from 16 random high-pile candidates;
- peak height 23.5912 m;
- initial volume 681.8121 m³;
- 309 scoops;
- final volume 67.7221 m³;
- final remaining 9.9327%;
- continuous video: 5,562 frames at 18 FPS.

Regenerate:

```powershell
python run_best_model_excavation.py `
  --model ablation/data_scale/steps_010000/policy.zip `
  --output best_model_large_pile_10pct `
  --seed 2026072801 --candidates 16 `
  --candidate-actions 24 --target 0.10 `
  --peak-min 23.5 --peak-max 24.0

python render_large_pile_long_animation.py `
  --pile-dir best_model_large_pile_10pct `
  --output best_model_large_pile_10pct/excavation_continuous.mp4 `
  --fps 18 --dpi 64 --frames-per-scoop 18
```

## Dataset generation

```powershell
python generate_loader_dataset.py `
  --count 100 --output-dir loader_dataset_100_real_scale
```

For a compact height-map-only transition dataset suitable for testing another
simulator or learned dynamics model:

```powershell
python generate_heightmap_dataset.py `
  --count 32 --output-dir heightmap_test_dataset
```

This export includes initial, immediate post-cut, and relaxed final height maps,
the height delta, the randomized bucket action, grid coordinates, a JSON
manifest, CSV examples, and a preview image. See the generated dataset README
for the exact coordinate convention.

For continuous large-stockpile sequences modeled after an industrial loading
site, generate `H0` followed by six consecutive stable excavation states:

```powershell
python generate_continuous_heightmap_dataset.py `
  --sequence-count 3 --scoops 6 `
  --workspace-size 25 --grid-spacing 0.05 `
  --output-dir continuous_heightmap_25m_dataset
```

The active-mask arrays define the irregular soil footprint independently of
the rectangular carrier grid used by the height-map representation.

Generated datasets are not committed or included in the curated release because they can
be regenerated and grow quickly. Publish a separately versioned dataset if they become a
stable research artifact.
