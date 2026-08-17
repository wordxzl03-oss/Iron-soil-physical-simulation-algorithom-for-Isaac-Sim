# Isaac Sim 散料交互管线架构审查

审查日期：2026-08-07  
审查范围：`/home/eric/Desktop/mesh` 现有代码、配置、WheelLoader USD、测试和已保存运行产物  
阶段：Phase A — repository audit / baseline freeze  
本次动作：完成审查并补齐 Phase A 的 baseline manifest/validator、隐私受限环境清单和 Isaac USD 只读 inventory；随后补装仓库已声明的独立 RL 环境依赖并建立完整非 Isaac test runner。没有修改 P0 主管线、基准 episode 或 WheelLoader USD，也没有开始 Phase B–K 功能。

## 0. 结论先行

当前仓库确实存在一条已实际跑通、可复现的模块化 P0 管线：Isaac articulation 的 bucket link pose 会进入工具运动学、连续扫掠、Heightmap 裁剪、MiniSlope 松弛、动态视觉 Mesh 和日志系统；701×701、0.05 m 的六铲结果两次独立运行位级一致。

但它目前仍是：

```text
逐帧直设车辆根位姿
  + lift/bucket position targets
        ↓
真实 bucket link pose
        ↓
FlatBottomQuad sweep
        ↓
Heightmap column clipping
        ↓
removed-volume sink
        ↓
action-end MiniSlope
        ↓
Visual Mesh + recorder
```

它不是目标系统中的“真实轮式车辆 + ContactView + 守恒散料 + payload + 土体反力”闭环。以下判断必须明确：

- **六铲 P0 几何回归：PASS。**
- **真实 wheel drive：FAIL。** 当前没有 wheel velocity/torque/throttle 命令。
- **正常运行无 teleport：FAIL。** 六铲共逐帧直设根位姿 1440 次。
- **轮胎/底盘站在 Heightmap 上：FAIL。** Heightmap 只有视觉 Mesh，没有 collider。
- **bucket 与 TerrainSupport 的 collision filtering：FAIL。** 当前没有 TerrainSupport，也没有 collision groups。
- **swept volume 与 payload 分离：FAIL。** 当前 `removed_volume` 就是裁掉的地形体积；4 m³ 铲斗单铲会“移除”10–20 m³。
- **完整系统质量守恒：FAIL。** 当前只闭合 terrain/removed/outflow 的 P0 账，payload 等字段没有行为。
- **MiniSlope 作为独立重力松弛器：基本 PASS。** 但它和账本仍依赖需要修正的体积积分契约。
- **Heightmap 离散语义：PARTIAL。** Mesh 已证明 H 是 vertex field，但体积代码仍按 cell-centered 样本求和。
- **Lighting/material：PARTIAL。** 可见性已改善并能运行，但仍嵌在演示和 Mesh adapter 中，不是要求的可配置模块/预设系统。

因此，下一步应是 Phase B 的视觉模块化和真实手动平地驾驶；不能把现有六铲演示继续包装成车辆物理或 payload 仿真。

## 1. 状态判定标准

- **PASS**：仓库中有真实实现，并有本次测试或已保存 Isaac 实跑证据。
- **PARTIAL**：有可执行实现，但仅覆盖目标的一部分，或契约存在已知缺口。
- **FAIL**：目标能力不存在，或当前行为与目标明确冲突。
- **PLACEHOLDER**：只有字段、配置入口或接口名，没有实际状态转移/物理行为。

“存在接口”不计为 PASS。

## 2. 当前实际架构

### 2.1 主管线

正式六铲入口是 [`isaac_loader/interactive_dig_demo.py`](isaac_loader/interactive_dig_demo.py#L392)，shell 入口分别是：

- GUI：[`run_isaac_demo.sh`](run_isaac_demo.sh#L14)
- headless acceptance：[`run_isaac_headless_acceptance.sh`](run_isaac_headless_acceptance.sh#L13)

实际对象装配在 [`interactive_dig_demo.py`](isaac_loader/interactive_dig_demo.py#L435) 和 [`interactive_dig_demo.py`](isaac_loader/interactive_dig_demo.py#L455)：

```text
configs/project_25m.yaml
        │
        ├── HeightmapIO ──────────────→ TerrainGrid / H_current
        ├── wheel_loader.usd ─────────→ SingleArticulation
        ├── RobotAdapter ─────────────→ joints + bucket-link pose
        ├── ToolDescriptorLoader ─────→ FlatBottomQuadProxy_L0
        ├── ToolKinematicsAdapter ────→ ToolState in terrain frame
        ├── ContinuousSweepBuilder ───→ mask + cut_surface
        ├── ExcavationOperator ───────→ clipped H + removed volume
        ├── TerrainStateManager ──────→ H_before/H_excavated/H_stable
        ├── MassLedger ───────────────→ terrain/removed/outflow balance
        ├── MinimumSlopeAdapter ──────→ action-end H_stable
        ├── DynamicMeshAdapter ───────→ derived visual USD Mesh
        └── ActionRecorder ───────────→ H、joint/tool、action、volume logs
```

职责分离已经有一个正确的骨架：RobotAdapter 不含挖掘几何，ToolDescriptor 不依赖视觉 Mesh，MiniSlope 被 adapter 隔离，视觉 Mesh 不是权威状态。这些都应复用。

### 2.2 当前还不存在的目标视图/闭环

下列目标层在仓库中没有实现：

```text
TerrainState(H_resting/mobile/payload/parcels)
├── ContactView / TerrainContactBackend
├── PlannerView
├── FailureZoneModel
├── MobileLayerSolver
├── BucketIntakeModel
├── Spill / AirborneParcel / Deposition
├── SoilForceModel → PhysX force feedback
└── continuous loading state machine / planner
```

当前车辆和土堆只通过“读取 bucket pose 后修改 H”单向关联；土体没有向 PhysX bucket/vehicle 返回力，也没有给车轮提供 Heightmap 支撑。

## 3. 当前相关文件树

下面是实际主管线和审查证据树。仓库根目录还包含大量早期 RL、OBJ、数据生成和渲染实验；它们不在当前六铲正式调用链中。

```text
mesh/
├── architecture_audit.md                 # 本报告
├── pyproject.toml                         # 项目/RL 依赖与 pytest dev 依赖
├── requirements.txt                      # 独立 .venv 依赖入口
├── .github/workflows/tests.yml            # 完整非 Isaac CI 入口
├── configs/
│   ├── project_25m.yaml                  # 当前 701×701 主配置
│   ├── project.yaml
│   ├── minislope.yaml
│   ├── bucket_{small,medium,large}.yaml
│   ├── bucket_markers.yaml
│   └── terrain_{128,256,512}.yaml
├── continuous_heightmap_25m_closed_dataset/
│   └── sequence_000_H0_initial_m.csv     # 当前 H0 输入
├── isaac_loader/
│   ├── interactive_dig_demo.py           # 当前正式六铲入口
│   ├── load_and_test.py                  # steering/lift/bucket 简单 smoke
│   ├── probe_usd_inventory.py            # Phase A Isaac headless 只读探针
│   ├── build_loader_usd.py               # WheelLoader USD 生成器
│   ├── wheel_loader.usd                  # 当前二进制 USD 资产
│   └── README.md
├── src/isaac_bulk_pipeline/
│   ├── audit/
│   │   └── usd_inventory_schema.py       # 纯 Python inventory 契约
│   ├── config/
│   │   └── loader.py
│   ├── robot/
│   │   └── robot_adapter.py
│   ├── tools/
│   │   ├── tool_descriptor.py
│   │   ├── tool_descriptor_loader.py
│   │   ├── tool_kinematics_adapter.py
│   │   └── marker_validator.py
│   ├── interaction/
│   │   ├── continuous_sweep.py
│   │   └── excavation_operator.py
│   ├── terrain/
│   │   ├── terrain_grid.py
│   │   ├── heightmap_io.py
│   │   ├── terrain_state.py
│   │   └── mass_ledger.py
│   ├── solvers/
│   │   ├── base_solver.py
│   │   └── minimum_slope_adapter.py
│   ├── visualization/
│   │   └── dynamic_mesh_adapter.py
│   └── runtime/
│       ├── simulation_controller.py
│       └── action_recorder.py
├── tests/
│   ├── test_*.py                         # 现有模块化纯 Python 测试
│   ├── test_phase_a_baseline.py
│   ├── test_phase_a_environment.py
│   ├── test_usd_inventory_schema.py
│   ├── isaac_smoke_robot_tool_adapters.py
│   ├── isaac_smoke_dynamic_mesh.py
│   ├── isaac_smoke_25m_mesh.py
│   └── isaac_smoke_full_pipeline.py
├── outputs/
│   ├── modular_25m_six_scoop/episode_0005/       # reference
│   ├── modular_25m_six_scoop_repro/episode_0001/ # independent repeat
│   ├── modular_25m_six_scoop_gui/episode_0004/   # 最新完整 GUI run
│   ├── modular_25m_reproducibility_check.json
│   ├── phase_a_baseline_manifest.json
│   ├── phase_a_environment.json
│   ├── phase_a_usd_inventory.json
│   ├── phase2_isaac_smoke.json
│   ├── phase5_25m_mesh_smoke.json
│   └── phase5_isaac_pipeline_smoke.json
├── slope_model.py                         # legacy MiniSlope 核心，仍被 adapter 调用
├── tools/
│   ├── phase_a_baseline.py                # 完整 hash + semantic validator
│   └── collect_phase_a_environment.py     # 隐私受限环境采集器
├── bucket_soil_coupling.py                # legacy prototype，不在主管线
├── src/isaac_bulk_pipeline.zip            # 静态归档，不是源码真值
├── run_isaac_demo.sh
├── run_isaac_headless_acceptance.sh
├── run_unit_tests.sh                      # 51 项模块化 P0/Phase-A suite
└── run_full_repository_tests.sh           # 51 项模块化 + 50 项 legacy/RL/OBJ
```

根目录的 `wheel_loader_rl/`、训练脚本和许多旧 `test_*.py` 是另一组历史代码。`run_unit_tests.sh` 仍只运行 `tests/` 中的模块化管线测试；新增的 `run_full_repository_tests.sh` 将模块化 suite 与历史 RL/OBJ suite 分进程执行，并显式排除会启动 SimulationApp 的 Isaac smoke/probe。Isaac Python 与独立 RL `.venv` 继续隔离，避免升级 Isaac 自带的 NumPy/Gymnasium/Torch。

## 4. 六铲 baseline freeze 审查

### 4.1 建议的唯一 reference

将以下目录作为 P0 几何交互 reference：

```text
outputs/modular_25m_six_scoop/episode_0005
```

独立重复运行：

```text
outputs/modular_25m_six_scoop_repro/episode_0001
```

最新完整 GUI 运行：

```text
outputs/modular_25m_six_scoop_gui/episode_0004
```

三个运行的 `H0_H6.npz`、`action_log.json` 和 `volume_log.csv` 均一致。不要使用会被后续运行覆盖的 `outputs/modular_25m_six_scoop_summary.json` 作为冻结锚点；它目前指向 GUI `episode_0004`。

### 4.2 已验证数值

| 项目 | Reference 结果 |
|---|---:|
| H0–H6 shape / dtype | `(7, 701, 701)` / `float32` |
| 网格间距 | `0.05 × 0.05 m` |
| carrier span | `35 × 35 m` |
| `H > 0.25 m` 的活动土堆包围盒 | 约 `27.55 × 22.90 m` |
| H0 peak | `8.101713 m` |
| 动作数 | 6 |
| 总 physics-frame log | 1440 |
| effective excavation events | 661；各铲 `87,126,125,104,104,115` |
| 总 P0 几何移除体积 | `86.84703958624432 m³` |
| closed-boundary outflow | `0 m³` |
| P0 ledger 数值误差 | `-1.8474111129762605e-13 m³` |
| headless runtime | `345.0666 s` |
| peak process RSS | `5038.43 MB` |

逐铲移除量：

```text
20.44041262969698
15.720377833446427
15.129730178094626
10.374785433541199
10.341418803300193
14.840314708164897 m³
```

MiniSlope 迭代数：

```text
5227, 4860, 5222, 8156, 7248, 7923
```

这些结果可从 [`run_summary.json`](outputs/modular_25m_six_scoop/episode_0005/run_summary.json) 复核。它们证明 P0 column-clipping + relaxation 的确定性，不证明 payload 或真实土体物理。

### 4.3 位级复现和关键 SHA-256

`episode_0005` 与重复运行的 H0–H6 位级一致，最大绝对差为 `0.0 m`。机器可读证据是 [`modular_25m_reproducibility_check.json`](outputs/modular_25m_reproducibility_check.json)。

```text
dbd94a1198a537c24b34ce474624a06c782fcc57b49e743c4bfe08e1e2433c89  H0_H6.npz
b946ae38beb39b0aa2a406d96223a17b1f2d1be99d3fe6decc20ed13bbf5d46e  action_log.json
c8461abfdd32d8912c6051465bac4b0e7f3f20e52d1343fce7c07be4c30ec5a2  volume_log.csv
```

关键输入/入口当前哈希：

```text
41936d66f211db6258b08ed88f2a66485a11a427f5ca85a89808a9fd335a4a7e  configs/project_25m.yaml
4a29d946eda51d3c607771c09389e0c5b44a80e942f263c2bd782ca2be9a4df2  configs/minislope.yaml
5f1a3b10102d1113ca865ac2b60443eca300b994b3b81330bbf73b82f663ef0b  isaac_loader/wheel_loader.usd
e16a9c3c757cc809cc023e4642ac6ddfb2cc4e085ee4e430edc464d79c04b458  sequence_000_H0_initial_m.csv
a4825a2047df100455276208a36f504a4752381759b2d7139826c687393ed881  isaac_loader/interactive_dig_demo.py
e00031f0dc333005fd102d4ce8af98edade8de31433ada56352a23f05d73f0f1  runtime/simulation_controller.py
9d67a9a612591882587161060124b7c8bd942ac33c4adb79085542130dc7d75a  slope_model.py
```

### 4.4 Freeze 状态

核心 baseline freeze 现为 **PASS**：

- [`phase_a_baseline_manifest.json`](outputs/phase_a_baseline_manifest.json) 保存 reference 全部 32 个文件和 34 个关键输入、配置、USD、入口及主管线源码的 size/SHA-256；
- [`tools/phase_a_baseline.py`](tools/phase_a_baseline.py) 的 `validate` 已实际复核 H0–H6 shape/dtype/finite/nonnegative、6 actions、1440 frames、661 events、分铲事件数、动作连续性、逐铲 removal 和所有哈希，结果 `ok: true`；
- manifest SHA-256 为 `59f6b81b16c9bc09378d3940ced7bc2b8fd930a7a0b52d615ee1e1560d4c065f`；
- [`phase_a_environment.json`](outputs/phase_a_environment.json) 保存 Ubuntu、Python/NumPy、Isaac/Kit 精确 build 和 GPU/driver/VRAM，且不收集环境变量、hostname、username、绝对路径或 GPU UUID；
- [`phase_a_usd_inventory.json`](outputs/phase_a_usd_inventory.json) 是 Isaac headless 实际读取二进制 USD 后产生的物理清单，不再只依赖 builder 推断。

环境清单确认：Ubuntu 22.04.5、CPython 3.10.12、NumPy 2.2.6、Isaac `4.5.0-rc.36+release.19112.f59b3005.gl`、Kit build `106.5.0+release.162521.d02c707b.gl`、RTX 3070 Ti Laptop 8192 MiB、driver 580.173.02。

仍需保留两个诚实限制：仓库 `.git/` 为空且没有 `git` 可执行文件，所以 revision 被明确记录为 `unavailable`；Phase A 使用数据和机器可读 inventory 作为证据，没有 H0/drive/dig/post-dig/dump 的截图，这组视觉截图属于 Phase B 验收。

## 5. 当前六铲执行路径

### 5.1 初始化

1. [`load_config`](src/isaac_bulk_pipeline/config/loader.py#L404) 读取 `project_25m.yaml`。
2. [`HeightmapIO.load`](src/isaac_bulk_pipeline/terrain/heightmap_io.py#L63) 将源文件明确从 legacy `H[x,y]` 转成 canonical `H[y,x]`。
3. 创建 `World`、默认 ground plane、灯光和 WheelLoader reference。
4. `SingleArticulation` 绑定 `/World/WheelLoader/rear_chassis`。
5. `RobotAdapter` 验证 robot/articulation/tool Prim 并读取实际 7 DOF。
6. 参数式 `ToolDescriptor` 绑定 bucket link 到统一 Tool Frame。
7. `DynamicMeshAdapter` 从 H0 创建一个固定拓扑 visual Prim。
8. 创建 `TerrainStateManager`、`MassLedger`、`MinimumSlopeAdapter`、`ActionRecorder` 和 `SimulationController`。

### 5.2 每铲固定阶段

六铲横向位置由 [`lateral_offsets`](isaac_loader/interactive_dig_demo.py#L536) 固定为：

```text
0.0, -3.0, 3.0, -1.5, 1.5, 0.6 m
```

每铲共 240 帧：

| Phase | 帧数 | root Y / joint 逻辑 | cutting |
|---|---:|---|---|
| `calibration` | 12 | 根不前进，lift −8°，curl −5°；测斗刃后再修正 root Z | OFF |
| `approach` | 24 | 根前进 0.7 m | OFF |
| `penetration` | 70 | 根继续前进 3.2 m，lift/curl 固定 | ON |
| `curl_and_lift` | 64 | 根不前进，lift −8°→20°，curl −5°→46° | ON |
| `retreat` | 70 | 根后退 4.2 m，lift/curl 保持 | OFF |

### 5.3 每 physics frame

[`_run_phase`](isaac_loader/interactive_dig_demo.py#L263) 的真实顺序是：

```text
smooth hard-coded phase interpolation
→ _set_loader_pose(root)
→ ArticulationAction(all joint positions; only lift/curl overwritten)
→ world.step()
→ RobotAdapter.get_joint_state()
→ RobotAdapter.get_tool_link_pose_world()
→ ToolKinematicsAdapter.update()
→ SimulationController.process_tool_state()
→ script-level ActionRecorder.record_frame()
```

当 `cutting_enabled=True` 且存在前一 ToolState 时，controller 执行：

```text
ContinuousSweepBuilder.build()
→ ExcavationOperator.apply(H_current)
→ TerrainStateManager.apply_excavation()
→ MassLedger.record_excavation()
→ DynamicMeshAdapter.update()（按显示 stride）
```

动作结束时 [`SimulationController.end_action`](src/isaac_bulk_pipeline/runtime/simulation_controller.py#L232) 执行：

```text
MinimumSlopeAdapter.solve_sequence(H_current)
→ reject non-converged result
→ apply H_stable
→ update ledger
→ full visual mesh update
→ save H_before/H_excavated/H_stable/trajectory/action log
→ H_stable becomes next action H_current
```

因此高度变化确实由实际读回的 bucket link pose 和 lift/curl joint 运动共同决定；但 chassis 的位移来自 root pose 直设，挖掘不受轮胎动力、bucket/soil contact 或土体反力约束。

## 6. 所有 `set_world_pose` / teleport

全仓库生产 Python 只发现一个车辆运行期逻辑入口：

- [`_set_loader_pose`](isaac_loader/interactive_dig_demo.py#L188)
- 单 articulation API：[`loader.set_world_pose`](isaac_loader/interactive_dig_demo.py#L196)
- 兼容 fallback：[`loader.set_world_poses`](isaac_loader/interactive_dig_demo.py#L198)
- 每帧调用点：[`_run_phase`](isaac_loader/interactive_dig_demo.py#L299)

两种 setter 是同一路径的互斥 API fallback，不是两次调用。它在所有五个 phase 的每一帧执行，因此：

```text
240 calls/action × 6 actions = 1440 root-pose writes/episode
```

这比“动作之间 teleport”更严重：正常前进、后退本身也完全是 kinematic relocation。

frame log 中动作边界的 tool-link 单帧跳变分别约为：

```text
3.889, 6.332, 4.834, 3.590, 2.692 m
```

初始化 USD 时使用的 `AddTranslateOp()` 只是资产建模，不属于运行期 teleport；camera 的 `set_camera_view()` 也不是车辆 teleport。其他生产 Python 未发现 `set_local_pose`、`set_joint_positions` 或同类 chassis pose 写入。

## 7. WheelLoader articulation 与真实 drive 判断

### 7.1 实际 DOF

当前二进制 [`wheel_loader.usd`](isaac_loader/wheel_loader.usd) 与 Isaac 实跑日志均给出 7 个 DOF，顺序如下：

Phase A 现已用 [`probe_usd_inventory.py`](isaac_loader/probe_usd_inventory.py) 在 Isaac Sim headless 中直接 `Stage.Open` 该二进制资产；机器可读结果见 [`phase_a_usd_inventory.json`](outputs/phase_a_usd_inventory.json)。它确认 28 prim、1 articulation root、8 rigid bodies、7 joints，inventory 自身 SHA-256 为 `1bc354b5793ae3ccab39654d7ad0f9a68158af9ec34b3a1d5b5525a54429fc04`。

| Joint | Body0 → Body1 | Axis | Limit | Drive stiffness | damping | maxForce |
|---|---|---:|---:|---:|---:|---:|
| `articulation_joint` | rear → front chassis | Z | −38°…38° | 400,000 | 80,000 | 2,000,000 |
| `rear_left_wheel_joint` | rear → rear-left wheel | Y | ±1,000,000° | 0 | 8,000 | 25,000 |
| `rear_right_wheel_joint` | rear → rear-right wheel | Y | ±1,000,000° | 0 | 8,000 | 25,000 |
| `front_left_wheel_joint` | front → front-left wheel | Y | ±1,000,000° | 0 | 8,000 | 25,000 |
| `front_right_wheel_joint` | front → front-right wheel | Y | ±1,000,000° | 0 | 8,000 | 25,000 |
| `lift_joint` | front → lift arm | Y | −12°…52° | 700,000 | 100,000 | 3,000,000 |
| `bucket_joint` | lift arm → bucket | Y | −45°…65° | 600,000 | 80,000 | 2,000,000 |

全部为 angular `DriveAPI`、`type="force"`、默认 target position 0。生成源码见 [`_revolute_joint`](isaac_loader/build_loader_usd.py#L75)。

刚体质量：

| Link | Mass |
|---|---:|
| rear chassis | 5200 kg |
| front chassis | 3100 kg |
| four wheels | 4 × 240 kg |
| lift arm | 850 kg |
| bucket | 980 kg |
| **total** | **11,090 kg** |

当前没有显式 COM/inertia tensor、悬架 DOF、velocity limit 或 power limit；惯量依赖 PhysX 自动计算/default。2–3 MN·m 的 lift/bucket/steering effort 与高 stiffness 尚无标定证据，只能标记为 `UNCALIBRATED`。

### 7.2 当前是否具备真实 wheel drive

结论：**FAIL。资产有可驱动轮关节，但当前运行路径没有真实 wheel drive。**

原因：

1. [`_apply_joint_targets`](isaac_loader/interactive_dig_demo.py#L204) 每帧创建一个覆盖全部 DOF 的零 position 数组，只把 lift/bucket 两项改为非零。
2. 没有 `joint_velocities`、`joint_efforts`、throttle、brake、wheel torque 或 wheel target speed。
3. `articulation_joint` 同样一直收到 0 位置目标，主演示不能实际转向。
4. wheel stiffness 为 0，position target 不提供推进；damping 只会产生阻尼/制动性质的响应。
5. [`load_and_test.py`](isaac_loader/load_and_test.py#L34) 只测试 articulation/lift/bucket position targets，也不驱动车轮。

六铲日志中四轮确有被动转动，最高约 3.4 rad/s；这是被逐帧移动的车体与默认 ground 接触后带动的结果，不是 wheel command 的证据。

## 8. 当前 collision setup

### 8.1 WheelLoader colliders

当前 USD 有 11 个 `CollisionAPI` Prim：

```text
/WheelLoader/rear_chassis/body
/WheelLoader/rear_chassis/cab
/WheelLoader/front_chassis/body
/WheelLoader/rear_left_wheel/wheel
/WheelLoader/rear_right_wheel/wheel
/WheelLoader/front_left_wheel/wheel
/WheelLoader/front_right_wheel/wheel
/WheelLoader/lift_arm/left
/WheelLoader/lift_arm/right
/WheelLoader/lift_arm/cross
/WheelLoader/bucket/mesh
```

- 轮胎是半径 0.78 m、宽 0.48 m 的 cylinder collider。
- bucket collider 使用 `convexHull`，见 [`_bucket_mesh`](isaac_loader/build_loader_usd.py#L107)。视觉上开放的斗口会被凸包碰撞近似封闭。
- 当前二进制 USD 中没有显式 PhysicsMaterial/friction/restitution、CollisionGroup 或 filtered pairs。
- build script 会在能导入 `PhysxSchema` 时尝试关闭 self-collision、写 contact/rest offset，见 [`build_loader_usd.py`](isaac_loader/build_loader_usd.py#L155)；但这个 import 是 optional，当前落盘 USD 未包含这些 PhysX 属性，生成结果不够确定。

Isaac inventory 的实测摘要为：11 colliders、0 collision groups、0 filtered-pair owners、0 physics materials、0 authored PhysX-attribute owners；同时确认 8 个刚体均有显式 mass，但 COM 和 diagonal inertia 均未 authored。

### 8.2 Terrain / ground

- 场景有一个 Isaac 默认静态 ground plane，见 [`interactive_dig_demo.py`](isaac_loader/interactive_dig_demo.py#L421)。
- `project_25m.yaml` 明确 `collision_enabled: false`。
- [`DynamicMeshAdapter.initialize`](src/isaac_bulk_pipeline/visualization/dynamic_mesh_adapter.py#L109) 会主动拒绝 terrain collision。
- 完成 episode 的 validation 也要求 terrain Prim **没有** `CollisionAPI`。

所以当前实际行为是：

| Pair | 当前行为 | 目标判断 |
|---|---|---|
| wheel/chassis ↔ default ground | 可接触 | 仅平地基础 |
| bucket ↔ default ground | 可接触 | 尚未过滤 |
| wheel/chassis ↔ visual Heightmap pile | 无接触，可能穿过 | FAIL |
| bucket ↔ visual Heightmap pile | 可穿过 | 因 pile 无 collider，不是正确 filtering |
| Wheel ↔ TerrainSupport | TerrainSupport 不存在 | FAIL |
| Bucket ↔ TerrainSupport | group/filter 不存在 | FAIL |

若现在直接给 visual Mesh 加 collision，bucket convex hull 会与地形发生物理阻挡，和自定义挖掘模型 double-count/冲突。正确做法必须是 Phase C 的独立 ContactView 与 collision groups，不能把 Visual Mesh 改成动态刚体碰撞层。

## 9. Terrain visual mesh 与 H 离散语义

### 9.1 当前 visual setup

[`DynamicMeshAdapter`](src/isaac_bulk_pipeline/visualization/dynamic_mesh_adapter.py#L78) 的真实行为：

- Stage 强制 metre、Z-up；
- H 的每一个样本直接成为一个 `(x,y,z)` vertex；
- 701×701 vertices 生成 700×700 quads、980,000 triangles；
- 固定对角线为 `(a,b,c)` 和 `(a,c,d)`；
- topology/Prim 只创建一次，后续替换 points，按 cadence 更新 normals；
- `doubleSided=true`、无 subdivision、无 RigidBody/CollisionAPI；
- 绑定粗糙 `UsdPreviewSurface`：roughness 0.92、metallic 0；
- 配置色为 `[0.44, 0.23, 0.07]`；
- 当前 visual carrier 仍是 35×35 m 的矩形顶面网格；不规则性来自零边界内的 H，而不是非矩形 topology；该 Mesh 也不是带底面/侧壁的 watertight solid。

`affected_bbox` 目前只进入 metrics；更新时仍替换全部 491,401 个 points，法线更新时也计算全图。

当前灯光在演示脚本的 [`_configure_scene_lighting`](isaac_loader/interactive_dig_demo.py#L130) 中直接创建 Dome、Distant Sun 和 overhead fill。它解决了“场景太暗”的即时问题，但还没有 `LightingManager`、`outdoor_day/overcast/strong_sun` presets 或截图验收。

### 9.2 关键离散语义缺陷

H 已被 Mesh 明确定义为 **vertex field**：`nx×ny` 个 H 样本对应 `nx×ny` 个顶点，实际 footprint 只有 `(nx-1)dx × (ny-1)dy`。

但 [`TerrainGrid.compute_volume`](src/isaac_bulk_pipeline/terrain/terrain_grid.py#L171) 仍使用：

```text
sum(H) * dx * dy
```

这等价于把每个顶点当成一个完整 cell-centered column，和当前三角 topology 不一致。按固定三角形 `(a,b,c)`、`(a,c,d)`，每个 quad 的严格线性曲面积分应为：

```text
V_quad = dx*dy/6 * (2*a + b + 2*c + d)
```

对一个边界非零的平坦 701×701 场，旧方法会多算：

```text
701² / 700² - 1 = 0.285918%
```

当前 H0–H6 四周恰好全为 0，因此 reference 上 sample-sum 与严格 triangle integration 只差约 `2.3e-13 m³`，本次 frozen regression 数字不受影响；这只是数据边界掩盖了通用错误。

其他相关缺口：

- `valid_mask` 的 shape 与 vertex field 相同，但文档称其为 cells；Mesh topology 完全忽略 mask；
- MiniSlope 拒绝带洞 mask；
- H 被强制非负，当前隐含“local Z=0 是材料基底”；没有独立 bedrock/base-height field；
- 在进入扩展 TerrainState 前，必须正式声明 H 为 vertex field，并增加统一 `TerrainVolumeIntegrator`；Excavation、MiniSlope、MassLedger 和 resolution tests 必须共用它。

## 10. 当前 ToolDescriptor / kinematics

### 10.1 已实现

[`ToolDescriptor`](src/isaac_bulk_pipeline/tools/tool_descriptor.py#L25) 定义统一 Tool Frame：

- origin：cutting-edge centre；
- +X：工具左到右；
- +Y：rear 指向 mouth/cutting edge；
- +Z：局部向上/斗底法向；
- `tool_to_link_matrix`：`T_link_from_tool`。

它包含 cutting edge、bottom profile、左右边界、interior profile、nominal width/capacity，并支持 parameters、USD markers、JSON/NPZ 三种来源。现有 unit tests 验证了 small/medium/large tool 不需修改核心 kinematics/sweep 代码。

当前主配置实际是：

```text
source                 parameters
actual proxy           FlatBottomQuadProxy_L0
nominal width          3.2 m
mouth depth            2.4 m
rear height            1.25 m
interior height        0.90 m
nominal capacity       4.0 m³
tool link              /World/WheelLoader/bucket
```

[`ToolKinematicsAdapter`](src/isaac_bulk_pipeline/tools/tool_kinematics_adapter.py#L106) 使用实际 bucket-link pose，计算 tool→terrain 变换、各代理点及有限差分 linear/angular velocity。

### 10.2 目标差距

- 配置和类强制只允许 `FlatBottomQuadProxy_L0`。
- `ContinuousSweepBuilder` 实际只使用 rear-bottom 和 cutting-edge endpoints 组成的四点平底 quad。
- 没有真实 mouth polygon/admission plane。
- `interior_profile_local` 只是一个二维中心截面，不是封闭 interior volume。
- capacity 只被校验和记录，任何 interaction 都不读取它。
- 无 teeth、curved bottom、payload COM、free-surface 或 retention geometry。
- tool velocity 来自 pose finite difference，不是 Isaac rigid-body velocity。
- `begin_action()` 清除 sweep history，但不会清除 kinematics velocity history；动作间 teleport 会产生虚假的大 tool velocity，只有完整 reset 才清除。

最直接的实证是：nominal capacity 为 4.0 m³，而六次单铲 removal 均为 10–20 m³。因此当前 removal 绝不能重命名为 payload。

## 11. 当前 ContinuousSweep / ExcavationOperator

### 11.1 ContinuousSweepBuilder

可复用部分：

- 根据平移距离、旋转角和 tool radius 自适应插值；
- 0.05 m 网格下平移步长限制为网格间距的 0.5；
- quaternion slerp 处理旋转；
- 对每个插值 pose 栅格化平底 quad 的两个三角形；
- 已有快速大步与慢速小步无缝一致、旋转采样测试。

目标差距：

- 输出语义是 `affected_mask + cut_surface`，直接服务于删除，不是纯 candidate intersection；
- 缺 penetration depth、candidate volume、local terrain normal/slope、cutting-edge velocity；
- 每 physics step 分配 mask 和 cut_surface 两个完整 701×701 数组，没有 local active domain；
- 异常 pose jump 没有最大 sampled-pose budget；
- 只扫平底 quad，不使用 side walls、mouth 或 interior。

合理演进方式是保留插值/栅格化核心，新增或包装 `ToolTerrainIntersectionResult`，并保证这一层绝不修改 H。

### 11.2 ExcavationOperator

[`ExcavationOperator.apply`](src/isaac_bulk_pipeline/interaction/excavation_operator.py#L61) 的完整语义是：

```text
candidate = affected_mask & finite(cut_surface)
target = max(cut_surface, minimum_height)
changed = old - target >= minimum_cut_depth
H_new[changed] = min(H_old, target)
removed = sum(H_old - H_new) * cell_area
```

这是真实运行的 legacy P0 backend，但与目标架构直接冲突：

- swept/cut candidate 立即永久修改 resting terrain；
- 所有被裁掉的体积成为系统 sink；
- 没有 failure wedge；
- 没有 resting→mobile 转换；
- 没有 conservation finite-volume transport；
- 没有 bucket mouth flux/capacity；
- 没有 pushing、side flow、spill、deposition；
- 没有 SoilForce。

当前测试只用同一个 `sum(deltaH)*area` 再计算一次 removal，不是独立的 triangle-volume oracle。

该模块应保留并明确命名为 legacy/regression backend，不能继续作为正式 `BulkMaterialInteractionModel`。

## 12. 当前 MiniSlope

正式接口是 [`TerrainRelaxationSolver`](src/isaac_bulk_pipeline/solvers/base_solver.py#L69)，实际实现是 [`MinimumSlopeAdapter`](src/isaac_bulk_pipeline/solvers/minimum_slope_adapter.py#L16)。legacy 入口只有：

```text
slope_model.relax_critical_slope
```

调用点见 [`MinimumSlopeAdapter._legacy_call`](src/isaac_bulk_pipeline/solvers/minimum_slope_adapter.py#L150)。

已实现并可复用：

- canonical `H[y,x]` 到 legacy `H[x,y]` 的显式 adapter transpose；
- closed/open boundary；
- `step/solve/solve_sequence/reset`；
- sequence frame/memory budget；
- closed-domain local crop 优化和 whole-grid convergence check；
- 非收敛结果拒绝 commit；
- 当前项目 `solve_trigger=action_end`，符合“event-driven resting relaxation”的目标职责。

当前主配置：38° critical angle、15,000 max iterations、2 mm neighbouring-height-excess tolerance、closed boundary、`1e-6 m³` conservation tolerance、sequence disabled。参数是模型/场景参数，不是已标定铁矿粉真值。

缺口：

- legacy solver守恒未加权 `sum(H)`，adapter 和 local-crop diagnostics 也使用当前 sample-sum volume；
- 没有统一 `TerrainVolumeIntegrator`；
- 没有 `theta_start > theta_stop` 的 material hysteresis；
- 没有每次调用 runtime/profile 字段；
- config 仍允许 every-step trigger，正式系统应让它只处理 resting/deposited geometry，而不承担 mobile transport。

MiniSlope 本身没有执行切削、入斗或力模型，这是正确的；后续必须保持这一职责边界。

## 13. 当前 TerrainState / MassLedger

### 13.1 TerrainStateManager

当前 [`TerrainState`](src/isaac_bulk_pipeline/terrain/terrain_state.py#L14) 只有：

```text
H_initial
H_current
H_before_action
H_excavated
H_stable
action_index
total_removed_volume_m3
```

它能正确保证 action N+1 从 action N 的 `H_stable` 开始，并已由六铲 artifact 和多动作 unit tests 验证。这是 P0 可复用行为。

目标状态中的 `H_resting`、mobile height/momentum、PayloadState、airborne parcels、MaterialScenario、timestamp、compaction/rut fields 全部不存在。

另一个事务缺口是：controller 在每个 cutting frame 已经修改 H 和 ledger；若 action-end MiniSlope 不收敛，controller 会拒绝 commit，但没有自动回滚到 `H_before_action`。

### 13.2 MassLedger

当前账本主量是 geometric volume，密度只用于明确标注的 estimate。这一点是正确的：`removed_mass_estimate_kg` 没有被声称为 measured payload mass。

但 payload/spill/deposited/exported/airborne 字段目前是 **PLACEHOLDER**：

- 只有 dataclass 字段和 reset-to-zero；
- 没有 transfer API；
- 没有 mobile reservoir；
- 没有 capacity/non-negative validation；
- `_recompute_error()` 不读取 payload、spill、exported 或 airborne；
- 当前等式只是：

```text
initial terrain
- current terrain
- removed sink
- boundary outflow
+ deposited
= numerical error
```

因此 `~1e-13 m³` 只证明 P0 removal ledger 自洽，不能证明：

```text
V0 = resting + mobile + payload + airborne + exported + boundary outflow
```

## 14. 当前 logging

[`ActionRecorder`](src/isaac_bulk_pipeline/runtime/action_recorder.py#L30) 已真实实现：

- `H_initial.npy`；
- 每动作 `H_before_action_N`、`H_excavated_N`、`H_stable_N`；
- 每动作 tool trajectory；
- 可选、有界 MiniSlope sequence；
- action JSON、volume CSV；
- 每 physics frame joint names/positions/velocities；
- tool-link world pose、tool terrain pose；
- phase、cutting flag、affected cells、incremental removed volume；
- effective excavation event 子集；
- `H0_H6.npz` 和 episode metadata。

当前 frame log 共 2.99 MB，effective-events JSON 又复制约 1.38 MB。所有 frame 保存在 Python list 中并多次重写 JSON，长周期作业不具备流式可扩展性。

目标 schema 缺少：

- VehicleCommand 和 actuator targets；
- root pose/velocity、roll/pitch/yaw；
- joint effort/torque、wheel slip/contact load；
- tool velocity、penetration、rake/mouth geometry；
- failure/mobile/payload/spill/deposition/airborne；
- SoilForce 和 application point；
- time/distance/energy/work；
- planner goal/candidates/path/cost；
- git revision、精确 Isaac build 和完整 environment manifest；
- H_after_deposition；
- 多阶段截图。

其他 schema 问题：

- `finalize_episode()` 无论动作数都硬编码输出名 `H0_H6.npz`；
- 不校验 frame timestamp 单调、frame ID 唯一和 action 顺序；
- controller 虽持有 recorder，逐帧记录仍由演示脚本手工调用，不是一个原子主管线步骤。

## 15. 测试与证据状态

### 15.1 本次实际运行

执行：

```bash
./run_unit_tests.sh
```

结果：模块化管线与 Phase A 审计工具的纯 Python 测试 **51/51 PASS**，最新复跑 3.381 s。原 39 项继续通过；新增 12 项覆盖 baseline 生成/验证/篡改拒绝、环境清单隐私/失败契约，以及 USD inventory schema。

另实际执行：

```bash
.venv/bin/python tools/phase_a_baseline.py validate
```

结果为 `ok: true`、32 个 episode 文件和 34 个 tracked 文件全部验证通过。USD probe 也已通过 Isaac headless 实跑并输出 `PHASE_A_USD_INVENTORY_OK`。

修复历史 RL 依赖后另实际执行：

```bash
./run_full_repository_tests.sh
```

结果为 **101/101 PASS**：模块化 P0/Phase-A suite 51 项（3.381 s），根目录 legacy/RL/OBJ suite 50 项（5.80 s）。后者包含原 `unittest discover` 会静默漏掉的 5 个 pytest function tests。独立 `.venv` 当前使用 Gymnasium 1.3.0、Stable-Baselines3 2.9.0、PyTorch 2.13.0+cu130；`pip check` 无损坏依赖，沙箱外 CUDA 实测能够识别 RTX 3070 Ti。

### 15.2 已保存 Isaac evidence

- 7-DOF robot/tool adapter smoke：[`phase2_isaac_smoke.json`](outputs/phase2_isaac_smoke.json)
- 701×701 dynamic mesh smoke：[`phase5_25m_mesh_smoke.json`](outputs/phase5_25m_mesh_smoke.json)
- 256×256 full-pipeline smoke：[`phase5_isaac_pipeline_smoke.json`](outputs/phase5_isaac_pipeline_smoke.json)
- reference 六铲：[`episode_0005`](outputs/modular_25m_six_scoop/episode_0005)
- repeat 六铲：[`episode_0001`](outputs/modular_25m_six_scoop_repro/episode_0001)
- 最新完整 GUI 六铲：[`episode_0004`](outputs/modular_25m_six_scoop_gui/episode_0004)
- 完整 baseline manifest：[`phase_a_baseline_manifest.json`](outputs/phase_a_baseline_manifest.json)
- 隐私受限环境清单：[`phase_a_environment.json`](outputs/phase_a_environment.json)
- Isaac 实际 USD inventory：[`phase_a_usd_inventory.json`](outputs/phase_a_usd_inventory.json)

本次审查没有再次启动一条耗时约 345 s 的 Isaac 六铲进程；两次既有独立 headless 运行和最新 GUI 完整运行已经提供相同 H/action/volume 证据。本次新增的 USD probe 则确实在 Isaac headless 中重新执行。

### 15.3 仍需后续阶段验收的范围

- 原先 5 个 repository-wide import errors 已通过在独立 `.venv` 安装声明依赖修复；不把这些 RL 依赖安装进 Isaac Sim 自带 Python。
- `configs/terrain_128/256/512.yaml` 指向当前不存在的 `data/H_initial_*.npy`；现有 config test 没有验证这些输入实际可加载。
- 当前无 Isaac 手动驾驶、真实 wheel drive、terrain contact、collision filtering、effort/energy 或 screenshot acceptance test。

## 16. 能力矩阵

| 能力 | 状态 | 证据/说明 |
|---|---|---|
| canonical `H[y,x]` 与 metre/Z-up | PASS | Grid/I/O/adapter tests |
| Heightmap 是权威地形，Mesh 是派生视图 | PASS | state + DynamicMesh 契约 |
| H 明确为 vertex field | PARTIAL | Mesh 如此使用，Grid 没有显式离散类型 |
| topology-consistent volume integration | FAIL | 仍为 `sum(H)dxdy` |
| 701×701 / 0.05 m 不规则闭合零边界 pile | PASS（P0 数据） | H0 peak 8.10 m，边界 0 |
| Dynamic visual Mesh 固定 topology/Prim | PASS | Isaac smoke + six-scoop summary |
| Visual Mesh 无 rigid/collision authority | PASS | 配置、代码和 runtime validation |
| 独立 LightingManager + presets | PARTIAL | 灯光能用，但仅为 demo helper |
| 独立 TerrainMaterialAdapter | PARTIAL | matte material 能用，但嵌在 Mesh adapter |
| RobotAdapter 读取实际 joints/tool link | PASS | Isaac smoke 和 1440-frame log |
| ToolDescriptor 可配置、多来源、tool-independent | PASS（L0） | unit tests |
| mouth/interior/capacity 的真实交互语义 | FAIL | capacity/interior 仅字段/截面 |
| adaptive continuous tool sweep | PASS（L0 geometry） | continuity tests |
| pure ToolTerrainIntersection result | PARTIAL | 有 candidate mask/surface，但语义仍面向删除 |
| ExcavationOperator 符合目标散料模型 | FAIL | 直接 column clipping |
| MiniSlope adapter/axis/boundary/sequence | PASS（P0） | solver tests + six actions |
| MiniSlope 使用统一严格 volume integrator | FAIL | 仍使用 sample sum |
| 六次 H 连续、可重复 | PASS（P0） | bitwise repeat |
| 目标 TerrainState（rest/mobile/payload/parcels） | FAIL | 只有 H action snapshots |
| 完整多 reservoir MassLedger | FAIL | 多数字段是 placeholder |
| assumed density 不冒充 true mass | PASS | 明确 `estimate`/not measured |
| P0 action/frame logging | PASS | arrays + JSON/CSV |
| World Model/RL-ready complete schema | FAIL | vehicle/material/force/energy/planner 缺失 |
| 真实 PhysX articulation asset | PASS | 7 DOF、有限 maxForce |
| 真实 wheel drive | FAIL | 无 velocity/effort/throttle command |
| articulation steering in main demo | FAIL | 目标固定为 0 |
| 手动 keyboard controller | FAIL | 不存在 |
| 正常车辆移动无 teleport | FAIL | 逐帧 1440 次 root pose write |
| Terrain ContactView | FAIL | 不存在 |
| wheel/chassis Heightmap support | FAIL | 只有 default ground |
| Bucket/Terrain collision filtering | FAIL | groups/filter 不存在 |
| wheel sinkage/rut disabled且有未来接口 | PARTIAL | 当前确实不变形，但接口/telemetry也不存在 |
| force/torque/slip/slope/energy telemetry | FAIL | 不存在 |
| SoilForce applied to Isaac | FAIL | 不存在 |
| continuous drive→dig→dump→return cycle | FAIL | 固定六铲 phase + teleport |
| classical attack/path planner | FAIL | 不存在 |
| World Model / RL 未提前训练 | PASS | 当前主管线未做这些工作 |

## 17. 已实现、可复用模块

以下模块应保持主体，不应为“重构漂亮”而推翻：

1. `TerrainGrid` 的 axis、单位和坐标变换部分。
2. `HeightmapIO` 的显式 `xy↔yx` 边界处理。
3. `RobotAdapter` 的 articulation/joint/tool-link 观测边界。
4. `ToolDescriptorLoader` 的 parameter/marker/file 三来源框架。
5. `ToolKinematicsAdapter` 的 link→tool→terrain 变换。
6. `ContinuousSweepBuilder` 的自适应 pose 插值和栅格化核心。
7. `TerrainRelaxationSolver` 抽象与 `MinimumSlopeAdapter`。
8. `DynamicMeshAdapter` 的固定 topology、单 Prim 更新模型。
9. `TerrainStateManager` 的 P0 action continuity 行为和对应 regression。
10. `ActionRecorder` 的 H/action/trajectory 基础文件契约，后续应版本化扩展。
11. 已冻结的六铲 P0 输出、完整 Phase A manifest/inventory/environment 证据和 51 项模块化测试。

## 18. 仅为接口、占位或 legacy backend 的部分

| 项目 | 实际状态 |
|---|---|
| `MassLedger.payload/spill/deposited/exported/airborne` | 字段 + reset，未进入完整余额或 transfer |
| `ToolDescriptor.nominal_capacity_m3` | 校验/序列化而已 |
| `interior_profile_local` | 二维截面，不是容量代理 |
| `MeshConfig.collision_enabled` | 只能为 false；不是 Contact backend |
| `DynamicMeshAdapter.affected_bbox` | 指标字段；更新仍是全图 |
| `TerrainGrid.valid_mask` | 坐标判断可用；Mesh/volume topology 语义未闭合，MiniSlope 不支持 holes |
| `PipelineConfig.extra_sections` | 仅保留未知 YAML；没有未来模块消费者 |
| wheel joint DriveAPI | 资产 hook；主管线没有 propulsion command |
| `RobotAdapter.reset()` | 只调用 articulation `post_reset`，不是手动控制状态机 |
| `ExcavationOperator` | 可执行 legacy column-clipping backend，不是正式散料 interaction |
| root `bucket_soil_coupling.py` | legacy prototype，不在模块化主管线 |
| `src/isaac_bulk_pipeline.zip` | 静态归档，不是当前代码源 |
| lighting helper / inline PreviewSurface | 可执行视觉修补，不是 LightingManager/TerrainMaterialAdapter |

以下目标类连占位接口都不存在：

```text
TerrainVolumeIntegrator
TerrainContactBackend / TerrainContactAdapter
TriangleMeshContactBackend / PhysXHeightFieldContactBackend
VehicleCommand / LoaderLowLevelController / ManualLoaderController
WheelTerrainContactSample / WheelTerrainOperator
expanded TerrainState / PayloadState / MaterialScenario / MaterialParcel
ToolTerrainIntersectionResult
FailureZoneModel / MobileLayerSolver / BucketIntakeModel
BucketRetentionSpillModel / AirborneParcelModel / DepositionOperator
BulkMaterialInteractionModel / SoilForceModel
LoaderOperationStateMachine / PlannerView / AttackPointPlanner / PathPlanner
```

## 19. Phase B 具体修改计划

Phase B 的边界是：**视觉可读 + 人能在默认平地真实驾驶 WheelLoader**。Phase C 才实现 Heightmap ContactView 和上坡；Phase E 以后才扩展散料状态。Phase B 不应偷偷加入假的 contact、payload 或 dump。

### 19.1 Phase A freeze 前置条件：已完成

进入 Phase B 前要求的四项无算法变更工作现已完成：

1. 32 个 reference 文件 + 34 个输入/config/USD/核心源码的完整 SHA-256 manifest；
2. 可重跑的 semantic/hash baseline validator；
3. 隐私受限的 runtime environment manifest；
4. Isaac headless USD inventory probe 和实际 JSON 清单。

P0 主管线、H 和期望数值均未改变。Phase B 开发必须保留这些历史证据；若有意修改 manifest 中冻结的旧入口或资产，必须把“历史 provenance 漂移”和“H/action regression”分别报告，不能静默重写 reference。

### 19.2 计划新增文件

```text
src/isaac_bulk_pipeline/vehicle/
├── __init__.py
├── vehicle_command.py
├── loader_low_level_controller.py
└── manual_loader_controller.py

src/isaac_bulk_pipeline/visualization/
├── lighting_manager.py
└── terrain_material_adapter.py

configs/
├── vehicle.yaml
├── visual.yaml
├── physics.yaml
└── logging.yaml

isaac_loader/manual_loader_demo.py
run_isaac_manual.sh

tests/
├── test_vehicle_command.py
├── test_loader_low_level_controller.py
├── test_manual_loader_controller.py
├── test_lighting_manager.py
├── test_no_runtime_teleport.py
├── isaac_smoke_manual_vehicle.py
└── isaac_smoke_visual_presets.py
```

### 19.3 计划修改文件

- `config/loader.py`：增加 typed vehicle/control/visual/physics/logging config；每个参数标注 `KNOWN_GEOMETRY`、`ASSUMED`、`UNCALIBRATED`、`NUMERICAL` 或 `VISUAL_ONLY`。
- `build_loader_usd.py`：让 `PhysxSchema` 缺失时明确失败；确定性写入 finite drive/effort/velocity、COM/inertia 或其来源；去掉 ±1e6° 的伪 continuous wheel limit；给轮胎和 Phase B 平地绑定明确 friction material。
- `wheel_loader.usd`：只通过生成脚本再生成，保存 inventory diff。
- `robot_adapter.py`：增加 root pose/body velocity、实际 effort/DOF inventory 读取；保持 tool-independent。
- `dynamic_mesh_adapter.py`：把 shader authoring 委托给 `TerrainMaterialAdapter`；保持 fixed topology 和 no-collision 契约。
- `visualization/__init__.py`：导出视觉 adapters。
- `action_recorder.py`：向新 schema 增加 VehicleCommand、actual actuator targets、root pose/velocity；保持现有 v2 episode 可读。
- `interactive_dig_demo.py`：优先保持冻结不动，继续作为明确标注的 legacy P0 regression；新 lighting/material 由新 manual entrypoint 使用，不把 teleport 路径当作 Phase B normal run。
- `run_isaac_demo.sh`：Phase B 通过后默认启动 manual demo。
- `run_isaac_headless_acceptance.sh`：继续运行冻结六铲 regression，确保 H hash 不变。

### 19.4 VehicleCommand 与低层控制

```python
@dataclass
class VehicleCommand:
    throttle: float
    brake: float
    steering: float
    lift: float
    bucket_curl: float
```

实现要求：

- 输入归一化并 clamp；
- throttle/steering/lift/curl 各自有 dt-based slew limit；
- keyboard callback 只更新 command state；
- physics loop 每帧消费 command；
- throttle → 四轮 sparse joint velocity/effort command；
- brake → 有限制动力/零速度目标；
- steering → `articulation_joint` rate-limited position target；
- lift/curl → 各自有限 position/velocity/effort target；
- 使用 sparse `joint_indices`，不能再用全零 position 数组覆盖无关 DOF；
- wheel direction、radius、target speed、torque、velocity、power/effort limit 全部配置化；
- initialization/reset 可以设置一次 root pose，正常 physics loop 禁止任何 pose setter；
- emergency recovery 必须是显式 debug path 并写日志。

按键契约：W/S throttle，A/D articulation，I/K lift，J/L curl/dump，Space brake，R 清空 command state，ESC 停止 controller。

### 19.5 视觉模块

`LightingManager`：

- DomeLight + DistantLight + optional fill；
- `outdoor_day`、`overcast`、`strong_sun` presets；
- 所有 intensity/exposure/color/rotation 配置化；
- 能读取并验证已 author 的 USD 属性。

`TerrainMaterialAdapter`：

- iron-ore-fines-like 深红褐/深棕视觉材质；
- high roughness、low metallic；
- 不使用 displacement 写回 H；
- shader 和 visual-only 参数从 DynamicMesh 解耦。

截图验收：H0、drive、dig、post-dig 和 bucket dump pose。Phase B 的“dump”只能标注为姿态截图；没有物料卸载，真正 dump 必须留到 Phase G。

### 19.6 Phase B unit tests

至少验证：

1. `VehicleCommand` clamp、dead-key release、brake priority。
2. 四路 command slew rate 与 dt 无关性。
3. DOF name→index 映射和左右轮方向。
4. sparse action 不会给 wheel 发送 position=0，也不会覆盖无关 DOF。
5. finite velocity/effort/steering/lift/curl limits。
6. keyboard callback 不访问 articulation/root pose。
7. normal-loop AST/runtime guard 中没有 `set_world_pose/set_world_poses`。
8. visual presets 的 schema、值域和 deterministic authoring。
9. 当前 51 项（含原 39 项 P0）tests 继续通过。
10. frozen `H0_H6.npz` hash 不变。

### 19.7 Phase B Isaac runtime tests

在 **默认平地** 上测试，因为 Heightmap support 属于 Phase C：

1. 按住/模拟 W：四轮得到非零受限 command，车辆 root 连续向前移动；pose setter count 必须为 0。
2. 松开 W、按 Space：命令回零并减速，不能靠直设速度/位置停止。
3. A/D：真实 `articulation_joint` 角度改变且不越限。
4. I/K、J/L：lift/bucket 实际 joint state 跟随且 effort/velocity 有限。
5. 保存 command→target→joint state→root motion 对照日志。
6. 检查无 NaN/Inf、无 physics explosion、无无限 effort。
7. 自动记录 controller runtime 的 mean/p50/p95/max；不先伪造性能结论。
8. 三种 lighting preset 各保存 USD 属性和截图。
9. 重新运行 headless 六铲 regression，确认 legacy H/action/volume 完全不变。

Phase B 不能用“wheel joint 在转”作为单独验收。必须同时证明有 wheel command、有限 effort、轮地接触和连续 root displacement，而且正常循环没有 pose setter。

### 19.8 Phase B 不做的内容

- 不让车辆开上 visual Heightmap；Phase C 才有 ContactView。
- 不给 visual Mesh 直接加 collider。
- 不实现 wheel rut/sinkage。
- 不修改 `ExcavationOperator` 的散料算法。
- 不实现 payload、spill、deposition 或真实 dump。
- 不施加 SoilForce。
- 不写固定六铲替代 manual control。
- 不训练 RL/World Model。

## 20. Phase B 之后的明确依赖

Phase B 只解决“能看清、能在平地真实开”。下一阶段的顺序不能交换：

```text
Phase B real manual vehicle on default ground
→ Phase C separate ContactView + collision filtering
→ Phase D physics/energy telemetry
→ Phase E authoritative multi-reservoir TerrainState
→ Phase F failure/mobile/intake/deposition
→ Phase G spill/dump
→ Phase H soil force feedback
```

尤其不能在 Phase C 前声称车辆站在土堆上，也不能在 Phase F/G 前把 `removed_volume` 改名为 payload。

## 21. Phase A 最终状态

| Phase A 项目 | 状态 |
|---|---|
| current architecture/file tree 审查 | PASS |
| six-dig execution path 审查 | PASS |
| teleport 全量定位 | PASS |
| actual joints/drives/masses 审查 | PASS |
| current collider/filter 审查 | PASS |
| Terrain/Tool/Excavation/MiniSlope/Ledger/Logging 审查 | PASS |
| 模块化 P0 + Phase A audit unit regression | PASS，51/51 |
| 两次六铲位级复现证据 | PASS |
| 完整 baseline manifest/validator | PASS，32 episode + 34 tracked files |
| runtime environment manifest | PASS，精确 Isaac/Kit/GPU/driver，隐私受限 |
| Isaac USD inventory probe | PASS，headless 实跑并保存 JSON |
| screenshot evidence | DEFERRED，Phase A 使用数据证据；视觉截图纳入 Phase B |
| repository-wide non-Isaac tests | PASS，101/101（51 modular + 50 legacy/RL/OBJ） |
| Phase B 实现 | NOT STARTED（符合本次要求） |

Phase A 所要求的主管线审查、baseline freeze、模块化 regression、全仓库非 Isaac regression 和实际 USD inventory 现已完成。Isaac smoke/runtime 验收仍按阶段独立执行，不能由纯 Python tests 代替。正确的下一动作是按第 19 节进入 Phase B，并先实现 additive 的 visual/manual flat-ground 路径，保持 frozen P0 entrypoint 不动。

## 22. 2026-08-07 Phase F–K additive status update

第 21 节是 Phase-A 冻结时的历史结论，不再代表当前实现范围。Phase B、C authoring、E 已有独立证据；F–K 的纯机制现已 additive 实现，未修改 P0 六铲入口或 baseline 文件。详细模块、契约、测试和未闭合的 Isaac runtime gate 见 `docs/PHASE_F_K_COMPLETION_REPORT.md`。

当前总判定必须拆开：F–K pure CPU acceptance 为 PASS；World Model/RL 前最终验收为 BLOCKED。原因不是把未实现接口包装成成功，而是平台禁止创建新的 Isaac/GPU 进程，因此 Phase D 新源码三坡度重跑以及 H/I/J/K 的物理闭环证据尚不能生成。`outputs/phase_f_to_k_acceptance.json` 是此边界的机器可读记录。
