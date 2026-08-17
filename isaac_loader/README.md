# Isaac Sim 4.5 模块化散料交互管线

`interactive_dig_demo.py` 是冻结的 P0 六铲几何回归入口。它不再调用
`bucket_soil_coupling.HeightfieldSoil`；每个物理帧均经过：

```text
RobotAdapter → ToolKinematicsAdapter → ContinuousSweepBuilder
→ ExcavationOperator → TerrainStateManager/MassLedger
→ MinimumSlopeAdapter（动作结束）→ DynamicMeshAdapter/ActionRecorder
```

它保留了历史的逐帧 root-pose 运动，仅用于保证 H0--H6 位级可复现；不能
当作 Phase-B 以后的正常车辆控制路径。实际计算工具明确为
`FlatBottomQuadProxy_L0`：3.2 m 宽平底四点四边形。该 legacy 路径没有实现
曲面斗底、铲齿、内部容量、Payload、卸料、DEM 或 MPM。

## 运行

Phase-B 真实 Articulation 手动驾驶（正常循环不写 root pose）：

```bash
cd /home/eric/Desktop/mesh
./run_phase_b_manual.sh
```

按键：W/S 前后驱动，A/D 铰接转向，I/K 动臂，J/L 铲斗收卷/卸料姿态，
Space 制动，R 只清空命令，Esc 退出。

Phase-C/D 三坡度物理支撑和遥测验收：

```bash
cd /home/eric/Desktop/mesh
./run_phase_cd_acceptance.sh
```

它使用隐藏静态 0.10 m ContactView 和独立 0.05 m RenderView，每个坡度使用全新
Isaac 进程。Bucket↔TerrainSupport 被过滤；Wheel/Chassis↔TerrainSupport 保留。
该验收不挖土，不写 Heightmap。

Phase F–K 的新主管线位于 `src/isaac_bulk_pipeline/`。其中
`ContinuousLoadingCycleCoordinator` 把散料、卸料、土体反力结果和连续状态机接到
同一个 `TerrainState`；`IsaacSoilForceAdapter` 只使用 Isaac Sim 4.5 的公开
`RigidPrim.apply_forces_and_torques_at_pos` 接口。当前没有把这些纯代码验收冒充成
Isaac 实跑：所需 H/I/J 集成运行证据明确记录为 BLOCKED/PENDING，详见
`docs/PHASE_F_K_COMPLETION_REPORT.md`。

以下两个命令只是冻结 P0 六铲回归：

GUI：

```bash
cd /home/eric/Desktop/mesh
./run_isaac_demo.sh
```

无界面完整验收：

```bash
cd /home/eric/Desktop/mesh
./run_isaac_headless_acceptance.sh
```

等价的直接命令为：

```bash
/home/eric/isaacsim/python.sh \
  /home/eric/Desktop/mesh/isaac_loader/interactive_dig_demo.py \
  --project-config /home/eric/Desktop/mesh/configs/project_25m.yaml \
  --scoops 6 --mesh-update-stride 4 --hold-frames 0 --headless \
  --output-dir /home/eric/Desktop/mesh/outputs/modular_25m_six_scoop
```

默认项目使用现有 WheelLoader USD、701×701 的 `H[y,x]` 高度图、0.05 m 间距、
35 m 承载网格和约 25 m 的闭合不规则土堆。六铲固定为 240 帧/铲，总计 1440
个完整物理帧。每一铲都执行 `begin_action()`/`end_action()`，下一铲严格从上一铲
的 `H_stable` 继续。

场景入口会显式创建天空环境光、太阳方向光和顶部补光，不依赖 Isaac/Kit 的默认
灯光。土堆绑定高粗糙度的哑光 PreviewSurface，并在 25 米 GUI 中让法线与点更新
同步，避免塑料感高光和白色条纹。修改代码后必须关闭已经打开的旧进程再重新运行，
现有 USD Stage 不会热更新。

## 输出契约

每个成功 episode 包含：

- `H_initial.npy`；
- `H_before_action_000...005.npy`；
- `H_excavated_000...005.npy`；
- `H_stable_000...005.npy`；
- `tool_trajectory_000...005.npy`；
- `joint_state_log.json`（所有物理帧）；
- `effective_excavation_events.json`（仅有效切削帧的辅助视图）；
- `action_log.json`、`volume_log.csv`、`metadata.json`、`run_summary.json`；
- `H0_H6.npz`，键为 `heightmaps_m`，shape `(7,701,701)`，dtype `float32`。

已验证结果位于 `outputs/modular_25m_six_scoop/episode_0005`。动态地形 Mesh 始终
复用 `/World/Terrain/DynamicSurface`，没有 `CollisionAPI`；它是高度图的派生
可视化，不参与机器人碰撞。

## 模型与性能边界

- 土体是 2.5-D 列裁剪和几何安息角松弛，不是真实铁矿粉物理真值。
- 配置的 38° 是项目级安息角；2 mm 是 5 cm 网格上的邻格高度数值容差，体积
  守恒仍单独按 `1e-6 m³` 检查。
- 普通六铲默认关闭中间松弛序列，只保留独立前/后状态。调试序列受 stride、
  max frames、dtype 和内存预算共同限制。
- 轮胎—散料反力、悬空/洞穴、颗粒飞散和卸料不在本阶段范围内。
