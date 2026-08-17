# 轮式装载机铲装降阶阻力模型开发文档

## 1. 文档目的

本文档说明项目中轮式装载机铲装降阶模型的设计、实现和验证方法。模型面向以下
用途：

- 快速生成大批量铲装训练数据；
- 在不运行 DEM 的情况下估算铲装阻力；
- 生成满足地面、坡脚、阻力和执行器限制的参考轨迹；
- 为 Isaac Sim 中的铲车 articulation 提供目标轨迹和外力；
- 作为后续 PBD/DEM 高保真仿真的初始轨迹和参数基线。

该模型是准静态、控制导向的工程近似，不是颗粒尺度物理仿真。

## 2. 代码结构

| 文件 | 作用 |
|---|---|
| `slope_model.py` | 高度图、随机土堆、坡度松弛及基础几何铲取 |
| `physics_aware_trajectory.py` | 降阶阻力、机器限制和受力约束轨迹规划 |
| `generate_loader_dataset.py` | 土堆与轨迹域随机化、数据集和动画生成 |
| `animate_loader_3d.py` | 铲斗几何和基础三维动画 |
| `plot_trajectory_diagnostics.py` | 切削刃高度、切深和阻力诊断图 |
| `test_physics_aware_trajectory.py` | 阻力与规划器自动测试 |
| `test_slope_model.py` | 高度图、体积守恒和坡度稳定测试 |
| `isaac_loader/wheel_loader.usd` | 可导入 Isaac Sim 的铰接铲车模型 |

## 3. 模型总体流程

```text
随机土堆参数
      ↓
生成非对称高度图
      ↓
临界坡度松弛并标定稳定峰高
      ↓
随机化矿石材料和机器能力
      ↓
从场地边界沿进铲航向搜索坡脚
      ↓
地面接近 → 贯入 → 受力抬升 → 收斗 → 举升
      ↓
按轨迹切深更新土壤高度图
      ↓
再次执行坡度松弛
      ↓
保存高度图、轨迹、阻力、参数和动画
```

## 4. 坐标系与单位

- 使用 SI 单位：m、s、kg、N、Pa；
- 高度图数组索引为 `(soil_x, soil_y)`；
- 三维绘图和铲斗路径使用 `(plot_x, plot_y, z)`；
- `plot_x = soil_y`，`plot_y = soil_x`；
- 零航向沿土壤坐标正 `y` 方向进铲；
- 正航向向土壤坐标正 `x` 方向旋转；
- 默认场地为 30 × 30 m；
- 默认矿堆占地约 20 × 20 m；
- 默认网格分辨率为 0.5 m；
- 默认稳定峰高为 8–10 m。

## 5. 土堆模型

### 5.1 高度图

土堆表示为二维高度场：

\[
h_{ij}=h(x_i,y_j)
\]

单元体积为：

\[
V_{ij}=h_{ij}\Delta x\Delta y
\]

总土体体积为：

\[
V=\sum_{i,j}h_{ij}\Delta x\Delta y
\]

### 5.2 随机形状

`random_pile()` 将多个随机旋转的椭圆高斯峰叠加，并允许少量负幅值凹陷：

\[
h(x,y)=\sum_n A_n
\exp\left[-\frac{1}{2}
\left(\frac{x_n'^2}{\sigma_{x,n}^2}
+\frac{y_n'^2}{\sigma_{y,n}^2}\right)\right]
\]

随后使用径向衰减形成有限占地。随机量包括：

- 峰数量；
- 峰位置；
- 长短轴尺度；
- 平面旋转角；
- 峰幅值；
- 局部凹陷；
- 总体空间展宽；
- 随机种子。

### 5.3 临界坡度松弛

相邻网格中心距离为 \(L\)，高度差为 \(d\)。若：

\[
|d| > \tan(\theta_\mathrm{crit})L
\]

则从高单元向低单元移动：

\[
\Delta h=
\frac{|d|-\tan(\theta_\mathrm{crit})L}{2}
\]

直到所有离散坡度满足临界坡度，或达到最大迭代次数。更新对每一对单元等量增减，
因此保持总体积。

当前矿石堆的休止角域随机化范围为 40–48°。

### 5.4 稳定峰高反标定

直接将原始峰高设置为 8–10 m 后，坡度松弛会使峰高下降。因此数据生成器采用
迭代反标定：

1. 生成峰值为 1 的固定随机形状；
2. 乘以当前幅值并执行坡度松弛；
3. 计算目标峰高与稳定峰高之比；
4. 修正原始幅值；
5. 重复 12 次。

最终记录：

- `peak_height_m`：目标稳定峰高；
- `stabilized_peak_height_m`：实际稳定峰高；
- `raw_amplitude_m`：松弛前所需幅值。

## 6. 坡脚检测

旧版轨迹根据最高点向后偏移固定距离，可能从坡面中部开始铲装。当前算法从场地
边界向土堆峰值追踪射线，并寻找首次超过坡脚高度阈值的位置。

坡脚阈值为：

\[
h_\mathrm{toe}=
\max(0.35,\ 0.055h_\mathrm{peak})
\]

算法步骤：

1. 根据随机航向和横向偏移确定目标点；
2. 计算从目标点反向到场地边界的最大距离；
3. 在边界到目标点之间采样 240 个点；
4. 找到高度首次超过 \(h_\mathrm{toe}\) 的位置；
5. 使用其前一个地面采样点作为进铲点。

接近段验证字段：

- `toe_height_threshold_m`；
- `max_approach_terrain_height_m`；
- `approach_clear`。

## 7. 降阶铲装阻力

### 7.1 设计依据

当前实现采用 Fundamental Earthmoving Equation（FEE）思想的控制导向降阶
近似。它保留阻力随切深、宽度、密度、摩擦、黏聚和速度增加的主要趋势，但没有
完整求解三维破坏楔体。

总阻力为：

\[
F_\mathrm{res}
=F_\mathrm{cohesion}
+F_\mathrm{wedge}
+F_\mathrm{payload}
+F_\mathrm{dynamic}
\]

### 7.2 黏聚切削项

\[
F_\mathrm{cohesion}=cbz
\]

其中：

- \(c\)：等效黏聚力，Pa；
- \(b\)：铲斗宽度，m；
- \(z\)：当前切深，m。

### 7.3 破坏楔体与被动土压力项

定义被动土压力系数：

\[
N_p=\tan^2\left(\frac{\pi}{4}+\frac{\phi}{2}\right)
\]

阻力项为：

\[
F_\mathrm{wedge}
=\frac{1}{2}\rho gbz^2N_p
\]

其中：

- \(\rho\)：矿石松散堆积密度，kg/m³；
- \(g\)：重力加速度；
- \(\phi\)：内摩擦角。

### 7.4 斗内载荷摩擦项

\[
F_\mathrm{payload}
=\mu_b\rho gV_\mathrm{load}
\]

其中：

- \(\mu_b\)：矿石与铲斗的等效摩擦系数；
- \(V_\mathrm{load}\)：当前累计装载体积。

### 7.5 动态阻力项

\[
F_\mathrm{dynamic}
=k_v\rho bzv^2
\]

其中：

- \(k_v\)：速度阻力系数；
- \(v\)：切削刃贯入速度。

### 7.6 当前域随机化范围

| 参数 | 范围 |
|---|---:|
| 松散堆积密度 | 1750–2300 kg/m³ |
| 等效黏聚力 | 3–14 kPa |
| 内摩擦角 | 40–48° |
| 斗壁摩擦系数 | 0.35–0.55 |
| 速度阻力系数 | 0.8–1.6 |
| 最大允许阻力 | 190–290 kN |
| 地面间隙 | 0.08–0.16 m |
| 贯入阶段最大切削刃抬升 | 1.0–1.7 m |

## 8. 受力约束轨迹

### 8.1 阶段划分

61 点参考轨迹按归一化时间分为：

| 阶段 | 时间范围 | 动作 |
|---|---:|---|
| 地面接近 | 0–22% | 切削刃保持近地面，驶向坡脚 |
| 贯入 | 22–64% | 前进、计算切深与阻力、逐渐抬升和收斗 |
| 完成收斗 | 64–80% | 切削刃近似停留，铲斗完成后倾 |
| 举升退出 | 80–100% | 铲斗举升并略微后退 |

### 8.2 名义抬升曲线

贯入进度为 \(p\) 时：

\[
z_\mathrm{lift}=
z_\mathrm{max}
\,
\left[
\max\left(0,\frac{p-0.28}{0.72}\right)
\right]^{1.5}
\]

实际切削刃高度为地面间隙与名义抬升之和。

### 8.3 阻力限幅

对每个贯入点：

1. 从高度图读取切削刃位置的地表高度；
2. 计算期望切深；
3. 预测阻力；
4. 若超过机器上限，二分搜索允许切深；
5. 抬高切削刃，使实际切深降至允许值；
6. 标记 `force_limited=True`。

因此：

\[
F_\mathrm{res}(k)\leq F_\mathrm{max}
\]

阻力增加时，轨迹自然从水平贯入变为抬升曲线。

### 8.4 收斗

收斗在贯入进度 45% 后开始：

\[
\theta_\mathrm{bucket}
=\theta_\mathrm{max}
\max\left(0,\frac{p-0.45}{0.55}\right)
\]

这近似真实驾驶员在形成足够切深后边前进、边举升、边收斗的复合动作。

## 9. 碰撞处理

当前模型处理：

- 切削刃不得低于地面间隙；
- 接近段不得从坡面中部穿入；
- 铲斗只在坡脚后进入允许切削区域；
- 路径必须位于场地和高度边界内；
- 土壤高度不得为负。

当前模型尚未完整处理：

- 车轮、车架和矿堆的三维碰撞；
- 举升臂和矿堆的碰撞；
- 铲斗侧板的三维接触；
- 外部岩壁、车辆和人员等障碍物；
- 车辆俯仰、侧倾和倾覆稳定性。

这些约束应在 Isaac Sim 中使用真实碰撞几何和 PhysX articulation 进一步验证。

## 10. 高度图切削更新

对每个高度图单元，将其转换到铲斗局部坐标：

- `forward`：沿进铲方向的距离；
- `transverse`：相对铲斗中心线的横向距离。

切深由规划轨迹沿贯入进度插值得到，并使用横向高阶衰减：

\[
w(y)=
\max\left[
0,\ 1-\left(\frac{2y}{b}\right)^6
\right]
\]

移除高度：

\[
\Delta h=
\min\left[h,\ z_\mathrm{cut}(s)w(y)\right]
\]

仅更新满足以下条件的单元：

\[
0\leq s\leq L,\qquad |y|\leq b/2
\]

铲取后再次执行临界坡度松弛。

## 11. 数据集字段

每个 `episode_XXX.npz` 包含：

| 字段 | 说明 |
|---|---|
| `initial_height` | 铲取前稳定高度图 |
| `after_scoop` | 铲取后、塌落前高度图 |
| `final_height` | 坡度松弛后的最终高度图 |
| `removed_height` | 每个网格移除的高度 |
| `bucket_path` | 61 × 3 铲斗参考点 |
| `bucket_pitch_deg` | 61 点铲斗俯仰角 |
| `cut_fraction` | 归一化切削进度 |
| `cutting_edge_z` | 切削刃高度 |
| `cut_depth` | 实际切深 |
| `resistance_n` | 预测铲装阻力 |
| `speed_m_s` | 贯入速度 |
| `force_limited` | 是否触发阻力限制 |
| `grid_spacing_m` | 高度图网格间距 |
| `workspace_size_m` | 场地尺寸 |

`manifest.json/csv` 额外记录：

- 土堆和材料随机参数；
- 轨迹随机参数；
- 机器阻力上限；
- 峰值阻力；
- 限力步数；
- 装载体积和装载率；
- 坡脚接近验证；
- 体积守恒和坡度稳定验证。

## 12. 运行方法

### 12.1 生成数据和动画

```powershell
C:\Users\guweihua\anaconda3\python.exe generate_loader_dataset.py `
  --count 100 `
  --seed 20260727 `
  --output-dir loader_dataset_physics_aware
```

### 12.2 生成诊断图

```powershell
C:\Users\guweihua\anaconda3\python.exe plot_trajectory_diagnostics.py `
  loader_dataset_physics_aware `
  --episode 1 `
  --output trajectory_diagnostics.png
```

### 12.3 运行测试

```powershell
C:\Users\guweihua\anaconda3\python.exe -m unittest -v
```

## 13. 自动验证

当前自动测试覆盖：

- 阻力随切深增加；
- 规划阻力不超过机器上限；
- 切削刃不低于地面间隙；
- 铲斗路径 Z 坐标非负；
- 坡度松弛体积守恒；
- 最终坡度不超过临界坡度；
- 随机土堆可复现；
- 铲取移除体积计算一致。

数据集逐样本检查：

- `volume_conservation_error_m3 < 1e-10`；
- `stable == true`；
- `path_in_bounds == true`；
- `approach_clear == true`；
- `0.60 <= fill_factor <= 1.20`；
- `loaded_volume_m3 > 0`。

## 14. 与 Isaac Sim 的集成

### 14.1 快速方案

在每个 Isaac Sim 物理步：

1. 读取铲斗切削刃位置和速度；
2. 查询高度图获得切深；
3. 调用 `excavation_resistance()`；
4. 将阻力沿运动反方向施加到铲斗刚体；
5. 将对应力矩传递到举升和收斗关节；
6. 按实际切削路径更新高度图或视觉网格。

### 14.2 PBD 校准方案

Omni PhysX 支持 GPU PBD 颗粒和颗粒材料。推荐：

1. 选择小规模矿堆和 0.25–0.35 m 等效粒子；
2. 在相同轨迹下记录 PhysX 接触力；
3. 使用接触力拟合 `cohesion_pa`、`bucket_friction` 和
   `velocity_drag`；
4. 用独立轨迹验证降阶模型误差；
5. 将标定后的模型用于大规模并行训练。

## 15. 已知局限

1. 阻力模型不是完整 FEE，也未显式求解三维破坏楔体；
2. 土堆是 2.5D 高度图，不能表示洞穴、悬垂和大块岩石架桥；
3. 未模拟颗粒粒径分布和岩块破碎；
4. 未模拟轮胎滑移、车辆速度下降和发动机转速；
5. 未将阻力转换为真实液压缸压力；
6. 未检查整车倾覆稳定性；
7. 斗内累计体积是几何近似；
8. 当前参数是工程初值，尚未由真实机器力传感器标定。

因此，模型输出应解释为“满足降阶约束的候选轨迹”，而不是已经通过真实矿区认证
的最优轨迹。

## 16. 后续开发建议

优先级从高到低：

1. 将阻力施加到 Isaac Sim 铲斗并记录关节力矩；
2. 加入轮胎牵引上限和打滑判据；
3. 加入车体俯仰和倾覆稳定性约束；
4. 用 PBD 颗粒仿真标定阻力系数；
5. 根据真实 CAN、液压压力和装载质量数据在线辨识参数；
6. 将轨迹规划改为带约束优化，而非逐点限幅；
7. 加入三维车架、举升臂和障碍物碰撞检测；
8. 对不同矿石类型建立独立材料参数分布。

## 17. 参考资料

- 原项目论文：*Large Scale Robotic Material Handling: Learning,
  Planning, and Control*。
- [Data-Efficient Excavation Force Estimation for Wheel Loaders](https://arxiv.org/abs/2506.22579)
- [Bucket Loading Trajectory Optimization for the Automated Wheel Loader](https://www.osti.gov/servlets/purl/1986059)
- [Omni PhysX PBD Particles](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html)
- [USD Physics API](https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/107.0.3/pxr/UsdPhysics.html)

## 18. 滞回式陡坡与突然塌方模型

为避免单一临界坡度模型把陡坡立即抹平，`slope_model.py` 新增
`relax_hysteretic_avalanche()`，原 `relax_critical_slope()` 保留为论文复现基线。

新模型包含：

- 起滑角 `start_angle_deg` 与停止角 `stop_angle_deg` 两个阈值。二者之间为
  亚稳区，坡面可以暂时保持；
- 基于黏聚力、内摩擦角、体密度和局部覆盖深度的 Mohr-Coulomb 安全系数；
- `disturbance` 扰动场，用于表示铲斗冲击、车辆振动或含水变化；
- 可在连续调用间传递的 `damage` 损伤场，使多次小扰动能够累积后触发失稳；
- 破坏从种子单元向相邻的超休止角网格传播，形成连通塌方区；
- 同步、质量守恒的网格转移，运动区最终松弛至停止角附近；
- 固定随机种子控制局部材料离散性，并可记录塌方历史用于动画。

推荐矿石堆初始参数范围：起滑角 55–72°、停止角 35–45°、内摩擦角
35–45°。黏聚力和扰动幅值必须通过现场或高保真颗粒仿真标定；60°以上坡面
不应被理解为普通干砂的长期稳定休止角。

演示命令：

```powershell
python render_hysteretic_avalanche.py
```

默认生成约 20 m × 20 m、10 m 高的料堆，64°前坡在局部扰动前保持，
随后发生连通塌方并回落至约 38°。输出位于
`hysteretic_avalanche_demo/steep_face_collapse.gif`。

`AvalancheStats.moved_volume` 是每次网格转移绝对体积的累计值，用于比较
塌方事件强弱；同一份材料可能经过多个网格，因此它不是净铲装体积。

## 19. 铲装—塌方耦合演示

`render_scooping_avalanche.py` 将真实尺度铲车动作与滞回塌方模型串联：

1. 铲斗沿地面接近约 64° 的亚稳前坡；
2. 使用 3.2 m 宽铲斗在坡脚保持水平短程贯入，最大贯入 1.5 m；
3. 贯入完成后保持后铰点固定，铲斗回转 50°，切削刃沿圆弧向上；
4. 收斗完成后，举升臂带动后铰点和整个铲斗整体上移；
5. 在切口附近生成铲斗冲击/振动扰动场；
6. 只允许暴露的前坡连通失稳，后坡保持黏聚稳定；
7. 铲斗保持收斗姿态并倒车退出。

运行：

```powershell
python render_scooping_avalanche.py
```

默认输出 `scooping_avalanche_demo/loader_scooping_collapse.gif`。演示参数下
几何装载量约 2.95 m³、峰值降阶阻力约 271 kN。铲斗位姿参考点为后部
铰点，切削刃与铰点距离约 1.74 m。收斗阶段铰点固定，举升阶段铰点才上移；
这避免了错误地绕切削刃翻转或让斗体钻入山体。该阻力尚未反向闭环调整
车辆速度或发动机转速，因此当前动画用于检查运动阶段、土体更新和塌方机制，
不等同于整车动力学仿真结果。

`failure_mask` 是可选的破坏域约束。它可由暴露坡面分割、接触邻域或更高保真
仿真给出，用来防止一次局部铲斗扰动无条件重整整个矿堆。

## 20. wheel_buck.obj 实际网格铲装

`render_obj_loader_scooping.py` 直接读取用户提供的 `wheel_buck.obj`。OBJ中的
三个对象保持独立：

- `loader_frame_world.stl`：车身；
- `loader_boom_world.stl`：车身与铲斗之间的动臂；
- `loader_bucket_world.stl`：铲斗。

脚本根据相邻网格最近表面区域推断两个铰点。当前模型坐标转换后的结果为：

- 动臂根销：`(y, z) = (-0.843, 2.056) m`；
- 动臂—铲斗斗销：`(y, z) = (1.097, 1.231) m`；
- 铲斗切削刃：`(y, z) = (3.025, 0.500) m`。

动作严格分成：

1. 整车沿地面接近；
2. 整车推进 1.35 m，动臂和铲斗保持低位；
3. 车身及动臂停止，铲斗绕实际斗销回转 48°；
4. 铲斗保持收拢姿态，动臂绕根销举升 34°；
5. 整车先倒车 3.0 m，退出塌方危险区；
6. 受扰前坡发生主塌方；
7. 整车保持举升姿态继续倒车退出。

运行：

```powershell
python render_obj_loader_scooping.py
```

默认输出 `obj_loader_scooping_demo/wheel_loader_scooping.gif`。动画同时显示
真实刃口轨迹、动臂根销和斗销。主塌方安排在机构退出危险区之后，避免高度图
回填到动臂占据的空间。每一帧还会检查车身及动臂对高度图的穿透；
超过 0.03 m 时脚本直接报错，不生成一个明显穿模的结果。铲斗本身允许进入
材料域，并负责产生切削体积与局部塌方扰动。

OBJ演示默认料堆已调整为约 42° 坡面、10 m 峰高、约 22.4 m × 22.4 m
占地，外部作业区为 30 m × 30 m。可使用 `--pile-angle` 在 30–55° 范围
调整坡面，例如：

```powershell
python render_obj_loader_scooping.py --pile-angle 38
```

默认42°缓坡演示的装载量约 1.83 m³、峰值降阶阻力约 191 kN，车身和动臂
逐帧最大土体穿透为 0。

### 20.1 斗内堆积和洒落

OBJ演示增加了180个等体积可视颗粒表示铲取质量：

- 颗粒随切削进度逐步进入铲斗，而不是在收斗结束时突然出现；
- 初始颗粒按铲斗内腔构造为低位楔形料层；
- 收斗阶段颗粒随真实铲斗刚体变换运动；
- 靠近斗口的10%颗粒在举升和振动阶段依次释放；
- 释放颗粒按照重力抛物线下落，接触当前高度图后停留在表面；
- 装入、斗内保留和洒落体积使用相同的单颗粒体积核算。

默认缓坡示例为：装入约1.83 m³、斗内保留约1.65 m³、洒落约0.18 m³，
满足 `装入体积 = 保留体积 + 洒落体积`。当前颗粒层用于低成本运动表达，
不是颗粒间接触求解器；颗粒之间没有DEM/PBD碰撞，落地颗粒也暂未反向更新
高度图体积。

## 21. 局部碰撞滑裂楔体模型

为解决“铲斗扰动坡脚后整个土堆都改变坡度”的问题，新增
`relax_localized_failure_wedge()`。旧的全局临界坡度和滞回塌方接口继续保留，
用于论文基线和对照。

新版执行顺序：

1. 使用铲斗峰值阻力和有效位移计算碰撞能量
   `E = F × Δs × η`；
2. 从碰撞点生成指数衰减扰动场
   `D(r) = exp(-r/Ld)`；
3. 根据黏聚力、内摩擦角、体密度和局部坡度计算Mohr–Coulomb安全系数；
4. 沿局部上坡方向建立扇形候选域，并从碰撞邻域搜索连通失稳单元；
5. 使用最大传播距离和剩余碰撞能量共同限制传播；
6. 每个网格最多只允许活动层厚度参与移动；
7. 滑裂楔体外网格保持逐值不变。

OBJ缓坡演示默认参数：

| 参数 | 数值 |
|---|---:|
| 扰动相关长度 | 2.5 m |
| 最大传播半径 | 5.5 m |
| 活动层厚度 | 0.65 m |
| 滑裂楔体半角 | 58° |
| 停止角 | 38° |
| 内摩擦角 | 40° |
| 黏聚力 | 5 kPa |
| 能量传递效率 | 0.22 |

当前样例的实际结果并未达到硬限制：失稳面积约6.50 m²，传播距离约2.13 m，
最大活动深度约0.55 m，局部搬运量约1.17 m³；碰撞能量18.93 kJ，其中消耗
约13.63 kJ。土堆总体积守恒，楔体外高度图保持不变。

动画目录额外生成：

- `local_failure_diagnostics.png`：扰动场、安全系数和高程变化图；
- `local_failure_diagnostics.npz`：对应数值数组、碰撞点和网格参数。

## 22. 连续多轨迹铲装

`render_multi_obj_loader_scooping.py` 在同一个持续更新的高度图上执行多次OBJ
铲车铲装。后一次铲装直接使用前一次“切削+局部滑裂”后的地形，不会为每条
轨迹重置土堆。

默认生成4条横向错开的轨迹，并对以下参数做可复现随机化：

- 横向进铲位置；
- 贯入距离；
- 最大切削深度；
- 收斗角；
- 动臂举升角。

运行：

```powershell
python render_multi_obj_loader_scooping.py --passes 4 --seed 2042
```

当前4铲样例：

| 铲次 | 横向位置 m | 贯入 m | 装载 m³ | 峰值阻力 kN | 局部失稳半径 m |
|---:|---:|---:|---:|---:|---:|
| 1 | -3.36 | 1.25 | 1.85 | 195 | 2.14 |
| 2 | -0.97 | 1.30 | 1.69 | 158 | 2.61 |
| 3 | 1.11 | 1.46 | 2.21 | 206 | 2.55 |
| 4 | 3.42 | 1.21 | 1.74 | 174 | 2.61 |

累计装载约7.49 m³，累计局部滑裂搬运量约2.85 m³，车身和动臂最大土体
穿透为0。输出：

- `multi_scoop.gif`：连续4铲动画；
- `multi_scoop_final_comparison.png`：初始、最终及累计高程变化；
- `trajectories.json`：逐铲参数和结果；
- `multi_scoop_terrain.npz`：初始、每铲结束和最终高度图。

为控制GIF生成时间，多次铲装动画采用抽帧剖面和俯视图；土体更新、碰撞检查
及局部失稳仍使用0.25 m完整网格。

## 23. 简化整车动力学与Gym环境核心

`wheel_loader_dynamics.py` 新增低速运动学二轮车和纵向动力学。车辆状态为：

```text
[x, y, heading, longitudinal_speed, steering, distance, energy, time]
```

坐标约定为航向角0沿世界+Y方向，正转向使车辆朝+X方向转弯。运动学方程：

```text
x_dot   = v sin(heading)
y_dot   = v cos(heading)
yaw_dot = v / wheelbase × tan(steering)
```

纵向动力学：

```text
m v_dot = F_drive + F_brake + F_roll + F_grade
          + F_velocity + F_excavation
```

其中驱动力同时受到三项限制：

- 最大驱动力；
- 发动机/电机功率 `Pmax / |v|`；
- 轮胎—地面附着上限 `μmg cos(grade)`。

地面阻力由滚动阻力、坡度阻力、速度线性阻尼和二次空气阻力组成。铲装阻力
通过已有降阶模型作为外部纵向反力输入。默认整车参数：

| 参数 | 数值 |
|---|---:|
| 整车质量 | 25,000 kg |
| 轴距 | 3.25 m |
| 最大转角 | 32° |
| 最大驱动力 | 220 kN |
| 最大制动力 | 260 kN |
| 最大功率 | 310 kW |
| 地面附着系数 | 0.82 |
| 滚阻系数 | 0.025 |
| 前进/倒车限速 | 4.2 / 2.8 m/s |

`WheelLoaderDynamicsEnv` 提供不依赖Gymnasium安装的兼容接口：

```python
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(action)
```

动作是归一化的 `[drive, steering, brake]`，观测是
`[x, y, heading, speed, steering, excavation_force, slip, energy]`。
以后只需继承 `gymnasium.Env`、声明 `action_space` 和 `observation_space`，
即可转为标准Gym环境。

演示：

```powershell
python simulate_vehicle_dynamics.py
```

默认使用速度闭环完成约1.5 m/s接近、0.6 m/s铲装、停车和约1.5 m/s倒车。
铲装阻力由0逐渐增加到165 kN，控制器相应提高驱动力并保持低速。输出：

- `vehicle_dynamics.gif`：车辆路径、驱动力及阻力动画；
- `vehicle_dynamics_diagnostics.png`：路径、速度、力平衡、能耗和打滑；
- `vehicle_dynamics_data.npz`：完整时间序列。

当前模块尚未包含车身侧倾/俯仰、前后轴载荷转移、铰接转向、液力变矩器、
轮胎沉陷和液压系统动力学。二轮车模型适用于低速路径控制与RL环境原型，
不用于极限稳定性验证。

## 24. 预加速惯性冲料与简化整车 OBJ

`simulate_inertial_obj_scooping.py` 将接近和贯入视为同一个连续动力学事件：

1. 车辆在距离料堆约 4 m 处起步；
2. 接触前用速度闭环提高驱动力，形成冲料速度；
3. 接触后保持部分油门，不再强制跟踪速度；
4. 铲装阻力随贯入深度、切削深度、已装载体积和速度增长；
5. 由整车质量的惯性和纵向合力积分得到自然减速；
6. 车速接近零后，才绕真实斗销收斗，再绕真实动臂根销举升。

贯入阶段使用：

```text
m dv/dt = F_drive(v, throttle) - F_ground(v)
          - F_excavation(depth, penetration, load, v)
```

贯入距离不再预先给定，而由质量、接触速度、牵引力、地面阻力和铲装阻力
共同决定。默认 25 t 样例为：

| 指标 | 当前样例 |
|---|---:|
| 最大冲料速度 | 2.55 m/s（9.2 km/h） |
| 最终贯入深度 | 1.78 m |
| 峰值铲装阻力 | 208 kN |
| 估算装载体积 | 2.78 m³ |

`build_simple_wheel_loader_obj.py` 生成 `simple_wheel_loader.obj`，单位为米，
+Y 为前进方向，Z 为高度。OBJ 用独立对象保存矩形底盘、驾驶室、四个车轮，
以及来自 `wheel_buck.obj` 的动臂和铲斗。

动画中底盘、驾驶室和四轮作为一个车体平移，车轮按累计行驶距离转动；
动臂根销固定于车体，铲斗销随动臂运动。该 OBJ 用于可视化和降阶控制原型，
并非带质量、碰撞体和关节约束的完整 Isaac Sim USD。

```powershell
python build_simple_wheel_loader_obj.py
python simulate_inertial_obj_scooping.py
```

输出包括 `inertial_obj_scooping.gif`、动力学诊断图和可复现的 NPZ 时序数据。

## 25. Perlin/分形噪声料堆域随机化

`slope_model.py` 中的 `random_pile()` 已由少量高斯凸包升级为多尺度分形
梯度噪声地形。接口保持不变，所以数据生成器和带 `--random` 的已有动画会
自动使用新版料堆。

生成过程包括：

1. 低频噪声决定峰顶偏移、主脊线和大尺度不对称；
2. 中高频 octave 形成坡面连续粗糙度；
3. 两个独立噪声场进行 domain warp，生成不规则坡脚；
4. 1～4 个随机尺度的宏观包络形成单峰、偏峰、肩峰或多峰料堆；
5. 高程归一化到指定峰高；
6. 按选定休止角运行临界坡度松弛，保证非负、体积守恒和坡度可用。

主要参数为 `base_cells`、`octaves`、`persistence` 和 `lacunarity`。同一 seed
逐点可复现，不同 seed 会同时改变包络、峰顶、脊线、坡脚和表面纹理。

查看样例：

```powershell
python render_perlin_pile_gallery.py --count 4 --seed 2028 --resolution 0.125
```

输出为 `perlin_pile_gallery/perlin_pile_gallery.png`。

高分辨率展示默认采用 0.125 m 网格，30 m × 30 m 工作区对应 241 × 241
个高程点，绘图时不再降采样到 55 × 55。批量数据生成器的默认网格从
0.5 m 调整为 0.25 m；需要最终高质量数据时可显式指定
`--grid-resolution 0.125`。二维网格间距减半后，单帧单层数组的单元数约
增加到四倍，临界坡度松弛与局部失稳计算时间也会相应增加。

## 26. 100组连续五铲数据

`generate_five_scoop_groups.py` 用于生成100个随机工况，每组在同一个持续
更新的料堆上连续完成5铲，共500次铲装。每组均重新随机化Perlin料堆、
峰高、休止角、轨迹横向位置、冲料目标速度、接触后油门、切深和机构角度。

单铲流程为：

```text
搜索当前横向剖面的坡脚
→ 车辆接触前加速
→ 铲装阻力作用下惯性减速
→ 根据自然停车位置确定候选贯入
→ 3.0 m³斗容硬约束
→ 切除扫掠体积
→ 局部Mohr–Coulomb滑裂与活动层松弛
→ 收斗、举升和退出
→ 将更新后的地形传给下一铲
```

斗容限制通过二分搜索缩短实际贯入距离，保证每铲
`loaded_volume_m3 <= bucket_capacity_m3`，而不是只在生成后剔除超斗容
样本。每组目录保存：

- `five_scoop.gif`：初始状态、每铲贯入和每铲退出后稳定状态；
- `five_scoop_data.npz`：初始/最终高度图、11个阶段高度图和5段动力学；
- `summary.json`：逐铲速度、贯入、装载、阻力和局部失稳指标。

数据集根目录的 `dataset_summary.json` 保存100组汇总。运行：

```powershell
python generate_five_scoop_groups.py --count 100 --resolution 0.25
```

## 27. 四轮地形跟随与车体姿态耦合

车辆不再假定始终位于水平地面。`wheel_loader_dynamics.py` 在每个积分步
根据轴距、轮距、航向和车辆中心位置计算前左、前右、后左、后右四个轮胎
接地点，并通过高度图双线性插值取得四轮地面高度。

准静态车体姿态为：

```text
z_body = mean(z_FL, z_FR, z_RL, z_RR) + wheel_radius
pitch  = atan2(mean(z_front) - mean(z_rear), wheelbase)
roll   = atan2(mean(z_right) - mean(z_left), track_width)
```

其中俯仰角作为当前纵坡角进入动力学：

```text
F_grade = -m g sin(pitch)
F_traction,max = μ m g cos(pitch)
```

因此前轮爬上料堆后会出现车头抬升、车体Z增加、上坡重力阻力增加和自然
贯入缩短。左右轮处于不同高度时会产生侧倾。OBJ动画将相同的车体俯仰、
侧倾和Z平移依次作用于底盘、车轮、动臂和铲斗，机构不会再悬浮于一个始终
水平的车体上。

连续五铲数据中，每一铲都基于上一铲更新后的当前地形重新建立四轮高度
采样器，并额外记录：

- `max_chassis_z_m`；
- `max_abs_pitch_deg`；
- `max_abs_roll_deg`。

当前对称单轨迹样例中，最大车体高度由0.72 m增加到约0.96 m，最大俯仰
约8.34°；加入坡度重力阻力后最终贯入由约1.78 m降到1.70 m。Perlin非对称
料堆的五铲冒烟样例出现约12.1°最大俯仰和11.3°最大侧倾。

该模型属于四点准静态贴地，不含悬架弹簧/阻尼、轮胎压缩、车桥摆动、
轮地脱离、车架扭转和动态载荷转移。极端崎岖地形仍需进一步升级为悬架和
轮胎垂向动力学。

## 28. 连续五铲GIF固定连杆与侧面剖视

旧版批量GIF的俯视示意线错误地连接了车体前缘和切削终点。由于车体位置与
切削终点分别更新，该示意线视觉上会随贯入阶段伸缩；这是绘图错误，不是
动力学或OBJ模型中的伸缩臂。

新版使用 `wheel_buck.obj` 推断的真实根销、斗销和刃口三点。动臂先绕根销
旋转，铲斗再绕斗销旋转，之后对三点施加相同的车体平移、俯仰和侧倾。
因此以下长度在所有帧严格不变：

```text
L_boom   = norm(bucket_pin - root_pin)
L_bucket = norm(cutting_edge - bucket_pin)
```

每组GIF现包含三个同步视图：

1. 俯视高度图、刚性车体、四轮和固定连杆机构；
2. 当前铲装横向位置处的地形侧面剖视；
3. 相对该组初始地形的累计高程变化。

侧视图显示车体高度、前后轮、底盘、根销、斗销、刃口以及当前
`pitch/roll`。贯入帧使用低位机构，退出稳定帧显示收斗和动臂举升后的
机构，并将车辆退回安全距离。

侧视图中的铲斗不再用斗销—刃口线段代替，而是读取
`loader_bucket_world.stl` 的完整三角网格。网格先随动臂根销运动，再绕
斗销执行收斗，最后施加车体高度和俯仰后投影到纵向—垂向平面，因此能够
显示斗底、斗背、开口和齿部的实体轮廓。由于源OBJ是带齿薄壳模型，其侧面
投影边缘会保留真实网格的齿形和局部复杂度。

无需重新运行土体仿真即可重绘已有数据：

```powershell
python generate_five_scoop_groups.py `
  --output-dir five_scoop_groups_100 `
  --rerender-existing
```

## 29. 铲斗—土体微观接触序列

连续五铲GIF不再把一次贯入压缩为单帧。每铲展开为初次接触、25%、50%、
75%和100%贯入，以及退出后的局部稳定状态。中间地形由已保存的最终扫掠
切除场按照刃口实际到达位置逐列激活，不是让整个切除区域同时消失。

新增的接触区放大视图同步显示：

- 接触前坡面虚线；
- 当前土体实体；
- 本铲已经切除的红色区域；
- OBJ铲斗实体及刃口接触点；
- 当前贯入距离与车辆速度；
- 土体反力方向及瞬时阻力大小。

速度、贯入和阻力来自每铲保存的车辆动力学时序；逐步切除形状来自该铲
保存的二维扫掠差值。该重建适合检查切削传播顺序和几何接触，但仍是
2.5-D高度场：放大图不会表现单颗矿石破碎、颗粒旋转、斗齿之间的三维
绕流或高速飞散，这些现象需要DEM/PBD模型。

批量重绘支持区间参数，可并行处理：

```powershell
python generate_five_scoop_groups.py --output-dir five_scoop_groups_100 `
  --rerender-existing --start-group 1 --end-group 25
```

## 30. 固定场景连续100铲长期演化

`generate_hundred_scoop_scene.py` 选用一个固定Perlin料堆，在同一持续更新的
高度图上连续执行100铲。轨迹不是100次重复同一沟槽，而是将料堆正面划分
为5个横向通道，每5铲随机打乱通道顺序并加入横向扰动，循环20轮。

每一铲仍完整执行坡脚搜索、地形跟随车辆动力学、惯性冲料、斗容约束、
局部滑裂和活动层松弛。动画为101帧：初始状态加每铲退出后的稳定状态。

当前固定种子 `1731118847` 的结果：

| 指标 | 数值 |
|---|---:|
| 初始料堆体积 | 347.93 m³ |
| 100铲后体积 | 185.43 m³ |
| 累计装载 | 162.50 m³ |
| 平均每铲装载 | 1.62 m³ |
| 单铲最大装载 | 2.84 m³ |
| 平均接触速度 | 1.91 m/s |
| 平均峰值阻力 | 175.43 kN |

高度场净减少体积与累计装载的差小于数值舍入误差，100铲均未超过3.0 m³
斗容。后期局部通道接近挖空时出现一次约0.012 m³的近空铲，说明固定通道
策略已开始失效，应由后续的可装载区域感知和轨迹重规划替代。

最大瞬时俯仰和侧倾达到约26.4°和27.5°，虽未出现在最终第100铲画面，
但已经超过常规安全作业范围。这不是应接受的轨迹结果，而是长期挖槽后
仍强制沿原横向通道进车造成的风险信号。后续策略应增加姿态终止阈值、
轮荷/倾覆稳定裕度以及通道重新选择。

运行：

```powershell
python generate_hundred_scoop_scene.py
```

输出包括101帧GIF、初末地形与统计图、JSON逐铲指标及NPZ高度图时序。
