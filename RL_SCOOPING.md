# 单次铲装强化学习基线

第一版使用 SAC 根据随机土堆高度图选择一条完整的进铲轨迹，目标是最大化每次
起铲获得的实际物料体积。

动作共 7 维，均归一化为 `[-1, 1]`：

1. 进铲航向
2. 相对土堆峰值的横向偏移
3. 接近距离
4. 贯入长度
5. 最大切深
6. 行驶速度比例
7. 起铲高度

观测包含 `15 × 15` 的低分辨率土堆高度图、峰值位置和高度，以及物料密度、
黏聚力和车辆允许的最大挖掘阻力。每个 episode 是一次完整铲装。

## 训练

```powershell
.\.venv\Scripts\Activate.ps1
python train_scooping_sac.py --timesteps 50000
```

快速验证代码和 GPU：

```powershell
python train_scooping_sac.py --timesteps 200 --eval-episodes 3 --smoke-test
```

结果默认写入 `rl_runs/sac_scooping/`：

- `scooping_sac_final.zip`：最终模型
- `evaluation.json`：独立随机土堆上的装载体积和轨迹
- `checkpoints/`：训练检查点
- `tensorboard/`：训练曲线

评估会在完全相同的随机土堆上同时运行学习策略、随机轨迹和各动作居中的固定
轨迹，并报告学习策略相对两种基线的装载体积提升百分比。

查看曲线：

```powershell
tensorboard --logdir rl_runs\sac_scooping\tensorboard
```

用训练模型在新随机土堆上执行一次铲装并保存动画：

```powershell
python render_trained_scoop.py --seed 20260808
```

默认生成 `rl_scoop_demo/trained_scoop.gif` 和同名 JSON 轨迹摘要。

连续执行 5 铲，每铲都根据更新后的土堆重新决策：

```powershell
python render_trained_five_scoops.py --scoops 5 --seed 20260808
```

## wheel_buck.obj 约束策略

从 `wheel_buck.obj` 读取车架、动臂、铲斗、根铰点、斗铰点、刃口、铲斗宽度和
举升范围，并使用 3.0 m³ 斗容上限重新训练：

```powershell
python train_obj_scooping_sac.py --timesteps 10000
```

用同一个 OBJ 的顶点和三角面渲染策略轨迹：

```powershell
python render_obj_policy_demo.py --seed 20260921
```

使用包含完整车身、驾驶室和四个轮胎的 `simple_wheel_loader.obj`：

```powershell
python train_obj_scooping_sac.py --obj simple_wheel_loader.obj `
  --timesteps 10000 --output-dir rl_runs\sac_simple_wheel_loader

python render_obj_policy_demo.py `
  --model rl_runs\sac_simple_wheel_loader\scooping_sac_obj_final.zip `
  --obj simple_wheel_loader.obj --seed 20260921
```

## 四轮接地动力学策略（推荐）

这一版本修正了 OBJ 轴心参考点，使用 `simple_wheel_loader.obj` 的 2.30 m
轴距，并在每个车辆积分步根据四个轮胎接地点更新车体高度、俯仰和侧倾。对于
非平面四轮高度，车体采用不允许任何轮胎穿过地面的最高刚性支撑平面。

```powershell
python train_dynamic_scooping_sac.py --timesteps 5000 `
  --output-dir rl_runs\sac_dynamic_simple_loader_fixed

python render_dynamic_simple_loader_demo.py --seed 20261001
```

这一版本学习的是高层轨迹参数。后续可把策略拆成多时间步控制，逐步输出油门、
转向、制动、举升和翻斗命令，并加入车辆姿态、侧翻风险和连续多铲后的土堆变化。
