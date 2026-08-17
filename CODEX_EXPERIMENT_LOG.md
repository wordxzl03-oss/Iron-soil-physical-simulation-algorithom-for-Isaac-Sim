# Codex 实验工作记录

最后更新：2026-07-28（Asia/Shanghai）

## 当前目标

完成同一配置下的 SAC 数据规模消融实验：

- 1,000 training steps
- 3,000 training steps
- 6,000 training steps
- 10,000 training steps
- 每个模型使用固定的 8 个料堆进行评估
- 记录训练料堆数、训练轨迹数、平均装载量、剩余比例、平均铲数和成功率

## 已完成工作

### 1. 重启前恢复

从 Codex 本地会话、文件修改时间和 JSON 内容确认：

- 1,000、3,000、6,000-step 训练及各自的 8-pile 评估已经完成。
- 原 `reference_10000_eval.json` 是对已有 10,000-step 模型的单独评估。
- 该参考评估只有评估指标，没有 `training_piles` 和
  `training_trajectories`，不能直接作为同口径的消融训练结果。

### 2. 消融脚本续跑支持

已修改 `run_data_ablation.py`：

- 新增 `--resume`。
- 当对应目录同时存在有效的 `policy.zip` 和 `metrics.json`，且
  `timesteps`、评估 episode 数与本次参数一致时，跳过重复训练。
- 将 PyTorch/Stable-Baselines3 等训练依赖改成按需导入，使已有结果的
  汇总不必加载 PyTorch。

### 3. 已生成的汇总产物

- `ablation/data_scale/ablation_results.json`
- `ablation/data_scale/data_scale_ablation.png`
- `ablation/data_scale/reference_10000_eval.json`

已完成结果：

| Steps | Training trajectories | Eval piles | Mean load (m³) | Remaining fraction | Success |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 997 | 8 | 0.4452 | 0.7852 | 0% |
| 3,000 | 2,855 | 8 | 0.6551 | 0.6841 | 0% |
| 6,000 | 5,975 | 8 | 0.6949 | 0.6649 | 0% |

旧的 10,000-step 参考模型评估包含 8 个完整 episode，但不含训练轨迹统计；
它将保留作历史对照，不冒充新的同口径消融结果。

## 当前阻塞

2026-07-28 08:26 重启后，Windows 激活了 Smart App Control /
代码完整性策略。事件查看器中记录：

- Code Integrity Event ID 3077、3033
- 被拦截文件：
  `.venv/Lib/site-packages/torch/lib/torch_cuda.dll`
- `torch_cuda.dll`、`torch_cpu.dll` 和 `shm.dll` 均未带数字签名
- Python 导入 PyTorch 时返回 `WinError 4551`

因此当前 Windows Python 环境不能启动 SAC 训练。项目代码和已有实验文件没有
损坏。

已检查的替代环境：

- Conda base：存在，但未安装 PyTorch。
- WSL：未安装。
- Docker：未安装。

为避免未经许可降低系统安全性，Codex 没有关闭 Smart App Control，也没有擅自
安装 Windows 子系统。

## 恢复后执行命令

当 PyTorch 可以正常导入后，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -u run_data_ablation.py `
  --budgets 1000 3000 6000 10000 `
  --eval-piles 8 `
  --device cuda `
  --output ablation\data_scale `
  --resume
```

预期行为：

1. 自动复用 1,000、3,000、6,000-step 完整结果。
2. 新训练 10,000-step 模型。
3. 评估固定的 8 个料堆。
4. 写入：
   - `ablation/data_scale/steps_010000/policy.zip`
   - `ablation/data_scale/steps_010000/metrics.json`
5. 重新生成四组数据的：
   - `ablation/data_scale/ablation_results.json`
   - `ablation/data_scale/data_scale_ablation.png`

## 完成判据

任务只有在以下条件全部满足后才算完成：

- `steps_010000/policy.zip` 存在且可加载。
- `steps_010000/metrics.json` 是有效 JSON。
- `timesteps == 10000`。
- `evaluation` 恰好包含 8 个 episode。
- `training_piles` 和 `training_trajectories` 均存在。
- 汇总 JSON 包含 1,000、3,000、6,000、10,000 四组结果。
- 没有遗留的训练 Python 进程。

## 2026-07-28 本轮恢复与新增任务

用户已手动关闭 Smart App Control。Codex 于恢复后验证：

```text
torch 2.13.0+cu130
cuda_available True
cuda_device NVIDIA GeForce RTX 5080 Laptop GPU
```

本轮需要依次完成：

1. 运行同口径的 10,000-step SAC 训练。
2. 使用固定的 8 个料堆评估，并记录训练料堆数和训练轨迹数。
3. 将 1,000、3,000、6,000、10,000 四档结果重新汇总。
4. 根据评估结果选择表现最好的模型。
5. 使用最佳模型挖掘一个峰高靠近配置上限的随机大土堆。
6. 持续挖掘到剩余土量严格低于 10%。
7. 保存完整动作、土量变化、最终地形和动画。

计划运行的 10,000-step 命令：

```powershell
.\.venv\Scripts\python.exe -u run_data_ablation.py `
  --budgets 1000 3000 6000 10000 `
  --eval-piles 8 `
  --device cuda `
  --output ablation\data_scale `
  --resume
```

执行策略：

- 已有的 1,000、3,000、6,000-step 完整结果通过 `--resume` 复用。
- 只训练缺失的 10,000-step 模型。
- 训练完成后再根据统一指标选择最佳模型，不预先假定步数最多的模型一定最好。
- 大土堆使用新随机种子并记录种子，保证动画和数值结果可复现。

## 10,000-step 新实验结果

同口径训练和 8-pile 评估已经完成：

```text
timesteps: 10000
training_piles: 42
training_trajectories: 9782
evaluation_piles: 8
success_rate: 0.0
mean_load_m3: 0.8120075894
payload_efficiency: 0.2706691965
mean_remaining_fraction: 0.6083897077
mean_scoops: 237.25
```

产物：

- `ablation/data_scale/steps_010000/policy.zip`
- `ablation/data_scale/steps_010000/metrics.json`
- 更新后的 `ablation/data_scale/ablation_results.json`
- 更新后的 `ablation/data_scale/data_scale_ablation.png`

在四个同口径模型中，10,000-step 模型具有最高平均装载量和最低平均剩余比例，
因此被选为当前最佳模型。

## 最佳模型大土堆挖掘方案

已新增 `run_best_model_excavation.py`，并扩展：

- `validate_large_pile_batch.py`：支持自定义目标剩余比例和峰高范围。
- `render_large_pile_long_animation.py`：按 `summary.json` 中保存的相同环境参数
  精确重放。

随机土堆选择方法：

- 使用固定的选择随机种子生成 16 个候选随机种子。
- 候选峰高范围限制为 23.5–24.0 m，保证属于较大土堆。
- 在候选中选择初始土体体积最大的一个。
- 使用 10,000-step 最佳模型和 24 个策略引导候选动作持续挖掘。
- 终止条件为剩余比例严格小于 0.10。

计划执行命令：

```powershell
.\.venv\Scripts\python.exe -u run_best_model_excavation.py `
  --model ablation\data_scale\steps_010000\policy.zip `
  --output best_model_large_pile_10pct `
  --seed 2026072801 `
  --candidates 16 `
  --candidate-actions 24 `
  --target 0.10 `
  --peak-min 23.5 `
  --peak-max 24.0
```

## 最佳模型大土堆挖掘结果

任务已完成，终止条件“剩余土量严格低于 10%”已满足。

随机候选与选择：

```text
selection_seed: 2026072801
candidate_count: 16
peak_height_range_m: [23.5, 24.0]
selection_criterion: maximum initial volume
selected_requested_seed: 618755297
selected_soil_seed: 1447608138
```

最终实验结果：

```text
model: ablation/data_scale/steps_010000/policy.zip
peak_height_m: 23.5912329840
initial_volume_m3: 681.8121380722
final_volume_m3: 67.7221345658
remaining_fraction: 0.0993266778
remaining_percent: 9.9326677811%
scoop_count: 309
minimum_scoop_label: 205
mean_load_m3: 1.9873462897
terminated: true
truncated: false
```

动画结果：

```text
file: best_model_large_pile_10pct/excavation_to_10pct.gif
format: GIF
resolution: 819 x 364
frames: 927
fps: 12
file_size_bytes: 12516045
final_remaining_fraction: 0.0993266778
minimum_rendered_wheel_clearance_m: 0.02
```

完整产物：

- `best_model_large_pile_10pct/summary.json`
  - 每铲动作、奖励、装载量、剩余比例、入土位置和轨迹。
  - 随机候选列表和选中依据。
- `best_model_large_pile_10pct/excavation_data.npz`
  - 初始/最终地形、观测、动作、奖励、装载量和剩余比例数组。
- `best_model_large_pile_10pct/excavation_to_10pct.gif`
  - 309 铲完整动画，共 927 帧。
- `best_model_large_pile_10pct/excavation_to_10pct.json`
  - 动画帧数、最终比例和渲染轮胎间隙校验。

动画复现命令：

```powershell
.\.venv\Scripts\python.exe -u render_large_pile_long_animation.py `
  --pile-dir best_model_large_pile_10pct `
  --output best_model_large_pile_10pct\excavation_to_10pct.gif `
  --fps 12 `
  --dpi 64
```

## 最终校验

- PyTorch/CUDA 可用。
- 四档消融汇总包含 `1000, 3000, 6000, 10000`。
- 10,000-step 结果包含 `training_piles=42` 和
  `training_trajectories=9782`。
- 随机大土堆初始体积为 681.8121 m³。
- 309 铲后剩余比例为 9.9327%，严格小于 10%。
- 数值结果、回放结果和 GIF 最终帧比例一致。
- GIF 可正常读取，实际包含 927 帧。
- 没有遗留训练、挖掘或动画渲染 Python 进程。

## 慢速动画版本

按用户要求保留原始 12 FPS 动画，并新增慢速版本：

```text
file: best_model_large_pile_10pct/excavation_to_10pct_slow.gif
resolution: 819 x 364
frames: 927
frame_duration: 160 ms
effective_fps: 6.25
file_size_bytes: 21902665
```

慢速版包含与原动画完全相同的 927 帧，播放时间约为原版的两倍。

## 连续操作动画

用户指出慢速 GIF 仍然只是降低帧率，并没有像普通动画一样连续记录操作过程。
检查发现原动画每铲约有 72 个真实车辆动力学状态，但只选择了开始、中间和结束
3 个状态，因此动作有明显跳跃。

已修改 `render_large_pile_long_animation.py`：

- 新增 `--frames-per-scoop` 参数。
- 每铲从真实动力学状态序列中均匀采样。
- 连续插值切土后的地形变化。
- 连续显示 `Approach`、`Cut and fill`、`Curl and lift` 阶段。
- 铲斗收斗角度和动臂举升角度随阶段平滑变化。
- 增加 MP4/H.264 输出，避免数千帧 GIF 体积过大。
- 安装项目虚拟环境依赖 `imageio-ffmpeg==0.6.0`。

第一次 MP4 编码因为画布宽度 819 不是偶数而失败。H.264 的 `yuv420p`
要求偶数宽高，随后为编码器增加 1 像素边缘补齐，输出宽度为 820，内容保持
不变。

连续动画生成命令：

```powershell
.\.venv\Scripts\python.exe -u render_large_pile_long_animation.py `
  --pile-dir best_model_large_pile_10pct `
  --output best_model_large_pile_10pct\excavation_continuous.mp4 `
  --fps 18 `
  --dpi 64 `
  --frames-per-scoop 18
```

连续动画结果：

```text
file: best_model_large_pile_10pct/excavation_continuous.mp4
codec: H.264
resolution: 820 x 364
scoops: 309
frames_per_scoop: 18
total_frames: 5562
fps: 18
duration_seconds: 309
duration: 5 minutes 9 seconds
file_size_bytes: 28705892
final_remaining_fraction: 0.0993266778
```

已抽取视频第 150 秒画面进行视觉检查：

- 当前为第 151/309 铲。
- 标题正确显示 `Approach` 阶段、装载量和实时剩余比例。
- 俯视地形、进场方向和装载机侧视姿态均正常。
- 轮胎渲染间隙保持 20 mm。

预览帧保存在：

- `best_model_large_pile_10pct/continuous_preview.png`
