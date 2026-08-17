# Model Zoo

The GitHub Release asset archive preserves the paths below. Extract it into the repository
root so existing commands work without path changes.

All nine selected policies were verified with:

```python
from stable_baselines3 import SAC
model = SAC.load("path/to/policy.zip", device="cpu")
```

| Model | Path | Purpose | Size |
|---|---|---|---:|
| Data scale 1k | `ablation/data_scale/steps_001000/policy.zip` | ablation baseline | 5.67 MB |
| Data scale 3k | `ablation/data_scale/steps_003000/policy.zip` | ablation baseline | 5.67 MB |
| Data scale 6k | `ablation/data_scale/steps_006000/policy.zip` | ablation baseline | 5.67 MB |
| Data scale 10k | `ablation/data_scale/steps_010000/policy.zip` | best fixed-eval data-scale model | 5.67 MB |
| Large pile v2 | `rl_runs/sac_large_pile_excavation_v2/large_pile_sac.zip` | 20% large-pile validation | 5.67 MB |
| Dynamic loader fixed | `rl_runs/sac_dynamic_simple_loader_fixed/dynamic_simple_loader_sac.zip` | four-wheel terrain dynamics | 5.68 MB |
| Simple loader OBJ | `rl_runs/sac_simple_wheel_loader/scooping_sac_obj_final.zip` | complete loader geometry | 5.68 MB |
| Bucket OBJ | `rl_runs/sac_obj_scooping/scooping_sac_obj_final.zip` | bucket geometry policy | 5.68 MB |
| Single-scoop baseline | `rl_runs/sac_scooping_baseline/scooping_sac_final.zip` | original SAC baseline | 5.68 MB |

## Recommended model

Use `ablation/data_scale/steps_010000/policy.zip` for the documented data-scale result and
10% supervised large-pile example. It achieved the best mean payload and lowest mean
remaining fraction among the four equal-configuration policies.

## Important qualification

The 10% rollout combines the learned policy with `max_scoop_supervisor.py`, which evaluates
24 nearby candidate actions. Direct-policy data-scale success remains 0% at the 20%
termination threshold.

## Not published

Intermediate checkpoints, smoke-test policies, duplicate experiment outputs, and the
older large-pile v1 policy are intentionally excluded from the curated asset archive.
