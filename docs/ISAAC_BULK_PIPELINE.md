# Isaac Sim 散料高度场交互管线

## 权威状态和坐标约定

- 权威地形始终是以米为单位的二维 `Heightmap`。
- 新管线统一使用 `H[y, x]`：axis 0 为 terrain-local +Y 的 row，axis 1
  为 terrain-local +X 的 column。
- 世界/地形变换使用 NumPy 列向量形式 `p_world = T_world_from_terrain @ p_terrain`。
- USD Mesh 只是 `Heightmap` 的派生显示，不从 Mesh 反推状态。
- 动态显示 Mesh 不配置 `CollisionAPI`，初始化后不重建 Prim 或拓扑。

现有 `slope_model.py`、数据集和早期交互演示使用 `H[x, y]`。兼容旧数据时，
必须在 `HeightmapIO` 或 `MinimumSlopeAdapter` 边界显式转置；进入核心状态后
禁止继续混用两种方向。

## 仓库审查与复用

| 现有文件 | 复用内容 | 接入阶段 |
| --- | --- | --- |
| `slope_model.py` | `relax_critical_slope` 及其守恒诊断 | Phase 4，经 solver adapter 转置接入 |
| `isaac_loader/interactive_dig_demo.py` | 正式 701×701 六铲入口 | P0 已完整迁移到主管线 |
| `isaac_loader/wheel_loader.usd` | articulation 和工具 link 验证模型 | Phase 2 |
| `bucket_soil_coupling.py` | 铲斗几何扫掠原型和测试案例 | 已 deprecated，仅作离线回归参考 |
| 现有 NPZ/CSV 数据集 | 旧 `H[x,y]` 回归数据 | 通过显式 legacy IO 转换 |

## 模块依赖

```text
YAML config ──> HeightmapIO ──> TerrainGrid ──> DynamicMeshAdapter
                                  ^                    ^
                                  |                    |
RobotAdapter -> ToolDescriptor -> ToolState -> ContinuousSweep
                                                |
                                                v
                                      ExcavationOperator
                                                |
                                                v
                                      RelaxationSolver
                                                |
                                                v
                                TerrainState + MassLedger
                                                |
                                                v
                                      DynamicMeshAdapter
```

禁止的反向依赖包括：Mesh → Heightmap、Excavation → Robot/Prim 路径、
Excavation → MiniSlope、Solver → USD。

## 分阶段文件计划

### Phase 1：基础接口

已新增：

- `src/isaac_bulk_pipeline/config/loader.py`
- `src/isaac_bulk_pipeline/terrain/terrain_grid.py`
- `src/isaac_bulk_pipeline/terrain/heightmap_io.py`
- `src/isaac_bulk_pipeline/visualization/dynamic_mesh_adapter.py`
- `configs/project.yaml`、`configs/terrain_128.yaml`、`terrain_256.yaml`、
  `terrain_512.yaml`
- Phase 1 单元测试和 Isaac smoke test

### Phase 2：工具适配

已完成：

- `robot/robot_adapter.py`：统一输出关节位置、关节速度、时间戳，以及换算成米的
  `T_world_from_tool_link`；不包含挖掘逻辑。
- `tools/tool_descriptor.py`：定义与视觉 Mesh 解耦的不可变 Tool Frame 代理点。
- `tools/tool_descriptor_loader.py`：支持 `parameters`、语义 `markers`、离线
  `JSON/NPZ file` 三种来源；marker 模式不读取视觉 Mesh 或 Bounding Box。
- `tools/marker_validator.py`：校验 8 个语义 marker、左右宽度、后部到斗口方向、
  局部向上和右手坐标系。
- `tools/tool_kinematics_adapter.py`：实现
  `tool local -> tool link -> world -> terrain local -> heightmap grid` 链路，输出
  代理点、线速度、角速度，并诊断统一/非统一尺度。
- `configs/bucket_small.yaml`、`bucket_medium.yaml`、`bucket_large.yaml`：同一算法
  仅换配置即可改变工具尺寸；另有 `bucket_markers.yaml` 验证 USD marker 模式。

统一 Tool Frame 原点位于切削刃中心，`+X` 从左向右、`+Y` 从斗后指向斗口、
`+Z` 局部向上。现有 `wheel_loader.usd` 的 bucket-local 轴为 `+X` 向斗口、
`+Y` 向左、`+Z` 向上，因此显式使用：

```text
T_link_from_tool =
[[ 0,  1, 0, 1.30],
 [-1,  0, 0, 0.00],
 [ 0,  0, 1, 0.00],
 [ 0,  0, 0, 1.00]]
```

验证命令：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
/home/eric/isaacsim/python.sh tests/isaac_smoke_robot_tool_adapters.py
```

结果：模块测试 `24/24` 通过；Isaac 4.5 冒烟测试加载真实 articulation 的 7 个
DOF，驱动 `lift_joint`/`bucket_joint` 后 Tool Frame 原点移动约 `1.90 m`，marker
宽度为 `3.2 m`，尺度奇异值约为 `[1, 1, 1]`。机器可读结果保存在
`outputs/phase2_isaac_smoke.json`。

### Phase 3：连续扫掠和挖掘

已完成：

- `interaction/continuous_sweep.py` 根据平移距离与旋转弧长自适应采样，强制
  `translation_step <= 0.5 * min(dx,dy)`，旋转采样同时满足
  `angular_step * tool_radius` 的网格约束。
- 每个中间 Tool Pose 将斗底四边形栅格化为 terrain-local `cut_surface`，所有
  采样用逐单元最小 Z 合并，因此高速或大时间步不会留下离散断裂槽。
- `interaction/excavation_operator.py` 只执行
  `H_new = min(H_old, z_cut)` 的 2.5-D column clipping；不引用 Robot、Prim、
  Visual Mesh、MiniSlope 或 USD。
- 移除体积严格由 `sum(H_old-H_new)*dx*dy` 计算，并输出改变掩码和半开 bbox。

验证覆盖：高速一步/20 个慢速子步、纯旋转、small/medium/large、最小切深，
以及相同物理区域的 128/256/512 分辨率收敛。高速/慢速扫掠掩码重合率超过
`99%`，差异只位于不共享采样相位时的一格栅格边界。

### Phase 4：MinimumSlope 闭环

已完成：

- `solvers/base_solver.py` 定义可替换的 `TerrainRelaxationSolver` 和
  `RelaxationResult`。
- `solvers/minimum_slope_adapter.py` 封装现有
  `slope_model.relax_critical_slope`。核心状态始终是 `H[y,x]`，只在 adapter
  调用边界显式转为旧 `H[x,y]`，同时把 `(dx,dy)` 原样传入，不隐藏单位转换。
- 现有 solver 会复制输入、使用 NumPy CPU、支持不同 dx/dy、闭合矩形边界、
  全局区域、最大迭代和容差，但原接口不能直接返回每轮状态。adapter 的
  `solve_sequence` 因此逐轮调用并按配置保存真正独立的数组副本。
- 闭合边界检查 `volume_before ~= volume_after`。开放模式每轮添加固定高度
  ghost ring，把内部损失记录为 `boundary_outflow_m3`，并检查
  `before ~= after + outflow`。
- `SimulationController` 拒绝把 `converged=false` 的结果提交为稳定状态。
- 普通运行的 `sequence_enabled=false`，只保留独立的松弛前/后状态。调试序列由
  `sequence_stride`、`sequence_max_frames`、`sequence_dtype` 和
  `sequence_memory_limit_mb` 共同限制；写盘逐帧执行，不对任意长度序列做
  无上限 `np.stack`。
- 25 m 项目对超坡区域及保守传播 halo 调用同一个 legacy MiniSlope，再进行全图
  坡度和体积复核。若全图不收敛，Controller 仍拒绝提交。

`configs/minislope.yaml` 保存默认参数；具体项目可覆盖迭代上限、容差、序列
采样和边界条件。该结果是降阶几何松弛，不被描述为真实铁矿粉物理真值。

### Phase 5：状态、记录和运行控制

已完成：

- `terrain/terrain_state.py` 显式管理 `H_initial`、`H_before_action`、
  `H_excavated`、`H_stable` 和权威 `H_current`；下一动作从上一动作的稳定副本
  开始，Reset 精确恢复初始数组。
- `terrain/mass_ledger.py` 记录当前体积、累计移除、边界流出和数值误差，并为
  payload/spill/deposited/exported/airborne 预留字段。密度输出始终标记为
  `density-based estimate`，绝不声明为实测 bucket payload。
- `runtime/action_recorder.py` 写出 metadata、每动作前/挖掘后/松弛序列/稳定
  高度图、Tool trajectory、JSON action log 和 CSV volume log。
- `ActionRecorder` 记录每一个 physics frame，而不只记录发生切削的帧；完整日志
  与 `effective_excavation_events.json` 辅助子集分开保存。
- `runtime/simulation_controller.py` 组合 ToolState、Sweep、Excavation、Solver、
  Dynamic Mesh、State、Ledger 和 Recorder；支持 `every_step`、
  `every_n_steps`、`action_end` 三种松弛触发方式。
- `tests/test_multi_action_state.py` 连续运行十次动作，检查第 N+1 次确实从第 N
  次稳定结果开始、所有文件存在、无 NaN/Inf、体积账本闭合、两次运行逐位
  可复现，并验证完整 Reset。

25 m 视频参考土堆使用 `configs/project_25m.yaml`：701×701 的 35 m 承载网格、
0.05 m 间距、名义 25 m 闭合不规则 footprint。源数据是明确标记的旧
`H[x,y]` CSV，因此配置要求在 HeightmapIO 边界显式转换。

### P0 六铲入口合并

`isaac_loader/interactive_dig_demo.py` 现在是模块化主管线的正式 GUI/headless
入口。每铲调用 `controller.begin_action()`，240 个 physics frame 都从真实
Articulation 读取关节和 bucket link 位姿，再通过工具运动学、连续扫掠、列裁剪、
状态/账本和动态 Mesh；动作末尾调用 `controller.end_action()`。因此状态链严格为
`H0 → H_stable_000 → ... → H_stable_005`，没有第二套 `HeightfieldSoil` 运行时。

新模块化物理配置的工具代理为 `ExtrudedProfileBucket_L1`。冻结 Phase-A 的
`project_25m.yaml` 仍保留经过 SHA 锚定的 L0 replay contract。L1 的切削刃、平面斗底、lip、
mouth polygon、top edge、back/side walls、闭合挤出 interior 与几何容量来自同一
`BucketGeometryDescriptor`；参数、marker 与显式 fallback 均记录 source/quality。
曲面斗底、铲齿、多截面和可靠 USD mesh 提取尚未实现。

## 最终验证命令

纯 Python 模块与十轮状态测试：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

当前结果为 `39/39` 通过。与新管线相关的现有土堆、六次连续挖掘、动力学和
旧 MiniSlope 回归另有 `35/35` 通过。

真实 Isaac 4.5 全链路无界面验证：

```bash
/home/eric/isaacsim/python.sh tests/isaac_smoke_full_pipeline.py
```

可视化运行同一条链路：

```bash
./run_isaac_demo.sh
```

701×701 六铲无界面验收：

```bash
./run_isaac_headless_acceptance.sh
```

2026-08-07 的完整验收结果位于
`outputs/modular_25m_six_scoop/episode_0005`：六铲均经模块化 Controller 提交，
`H0_H6.npz` 为 `(7,701,701)`/`float32`；完整帧 1440 条，有效切削事件 661 条；
逐铲移除体积为 20.440413、15.720378、15.129730、10.374785、10.341419、
14.840315 m³，总计 86.847040 m³；闭合边界流出为 0，体积账误差为
`-1.85e-13 m³`。运行用时 345.07 s，进程 RSS 峰值 5038.43 MB。动态 Mesh Prim
全程复用且没有 CollisionAPI。

相同 seed 的第二次完整运行位于
`outputs/modular_25m_six_scoop_repro/episode_0001`，耗时 337.94 s、RSS 峰值
5078.44 MB。两次的 H0–H6 逐位相同，六铲体积和迭代次数一致；机器可读比对为
`outputs/modular_25m_reproducibility_check.json`。

最终验证使用 256×256、0.05 m 网格。真实 articulation 产生 8 次有效高度场
修改和 148 个自适应 Tool Pose，移除 `15.135835635 m³`；MinimumSlope 在
3659 次迭代后收敛；体积误差为 `-1.42e-14 m³`；同一个 65,536 顶点 Mesh
Prim 被复用且没有 CollisionAPI；Reset 恢复初始高度图。机器可读结果位于
`outputs/phase5_isaac_pipeline_smoke.json`，完整动作文件位于结果 JSON 指向的
episode 目录。

25 m 数据的真实 Isaac Mesh 验证：

```bash
/home/eric/isaacsim/python.sh tests/isaac_smoke_25m_mesh.py
```

701×701、0.05 m 的 35 m 承载域成功创建 491,401 顶点和 980,000 三角形；
名义 25 m 闭合土堆峰值为 8.101713 m、边界最大高度为 0；一次固定拓扑更新
耗时约 6.38 ms，Prim 被复用且没有 CollisionAPI。结果保存在
`outputs/phase5_25m_mesh_smoke.json`。

## 当前模型边界

- 挖掘仍是 2.5-D 列裁剪，不是 3-D 布尔、DEM 或 MPM。
- 工具计算代理是平面斗底、横向挤出 interior 的 L1 reduced-order geometry，
  不是完整真实 mesh、曲面斗底或铲齿模型。
- 已有 reduced-order 土体反作用力、容量截断、retention/spill 与卸料机制，但
  尚无铁矿粉标定、完整内部自由面/死料演化或轮胎—散料碰撞验证。
- 一次扫掠的几何移除量不是铲斗有效载荷；当前烟测刻意让代理穿过高土堆，
  因而移除量可以大于 nominal bucket capacity。
- Dynamic Mesh 是纯视觉派生物且没有 CollisionAPI；机器人目前只与静态地面
  发生 PhysX 接触。
