# 一个 World、多个对象、一条 ActiveLoop

`python -m physrsr run --config CONFIG --out RUN` 始终调用 `workflows/loop.py:ActiveLoop`。
配置必须提供 `target / simulator / physic / agent` 四个端点。
三个长条共存于一个 World，由一套机器人和一份 CaP 操作；对象数不改变调度算法。

```text
任务 + 整场景视觉
  → Physic Agent 评估整场景任务不确定性
  → Probe → 同一个 World 中的机器人交互
  → 按 object_id 保留测量与估计
  → 拟合观测重放 → 提交整场景预测模型
  → 一份完整任务 CaP → 整场景模型集合评估
  → 再辨识 / 修订 CaP / 独立验证 / 停止
```

主循环只创建一个模型会话，更新模型版本、共享交互预算和任务候选集合。
场景对象的参数不混用。模型集合表示对同一 World 的不同物理假设，不是多个真实 World。
首轮动作仍从实测任务和视觉出发；没有测量前不执行未辨识模型的预测。

## 观测和机器人接入

多对象适配器的 `observe()` 返回同步观测，调用 `runtime.instrumented.observation(raw)` 过滤：

```python
{
    "time_s": ...,
    "coordinate_frame": "robot_base",
    "objects": {
        "rod_a": {...},
        "rod_b": {...},
        "rod_c": {...},
    },
    "tcp": {...},
    "wrench": {...},
    "joints": {...},
    "gripper": {...},
}
```

每个对象包含原 v2 的视觉字段：`position / quaternion_wxyz / axes / dimensions_m / grasp_quaternion_wxyz / support_clearance_m`。
物体质量、摩擦等私有真值不在观测白名单中。对象 ID 在准备、执行、预测与恢复期间保持一致；丢失或替换对象会报错。
观测归一化后保留 `object` 作为按 ID 排序的第一个对象的兼容视图，它不表示当前抓持对象。
程序通过 `objects[object_id]` 的实时位姿选择动作，不通过切换 World 选择长条。

所有对象共享一份 TCP、关节和腕部传感。腕部载荷不是每个对象独立测得的载荷。
硬件适配器仍使用 `MeasuredRobot`；`perception.object` 可以返回 `objects` 映射及同步时间。
场景机器人必须声明实际公共工作空间：

```python
capabilities["workspace"] = {
    "min_position_m": [xmin, ymin, zmin],
    "max_position_m": [xmax, ymax, zmax],
}
```

它替代单物体附近的默认运动范围。准备稳定性、恢复和安全断点检查覆盖所有物体。
开放夹爪与保存断点要求所有对象均处于支撑面附近；这一保守规则不适用于要求自由落放的任务。
原单物体适配器可以继续只返回 `object`，观测与预测特征合同保持兼容。

## 对象级记忆和整场景预测

Probe 返回的每条可辨识估计必须有已知的 `object_id`：

```python
{"object_id": "rod_b", "name": "mass_kg", "value": ...,
 "unit": "kg", "frame": "scalar", "identifiable": True, ...}
```

`robot.describe()['identification']` 保留统一的测量历史、证据来源和断点，
同时通过 `objects[object_id]['estimates']` 索引对象估计。无对象归属的场景估计会被拒绝。
决策合同为兼容保留 `instances / instance_assessments / target_instance_ids` 等字段，
但主循环恰好只有 `current` 一个条目，代表完整 World，不能新增三个 rod 实例。

多对象预测特征为 `measured_scene_response_v2`：按稳定对象 ID 排序，保留每个对象的 6 维姿态/相对位移响应，
再拼接一份 10 维共享腕部响应。三个对象为 28 维，共享传感信息不按对象重复计权。
单对象使用 `measured_response_v3` 的 16 维投影。均值去除重复时间戳并按时间积分；完整同步观测仍存档。
旧特征版本不与新版本混合使用。
`loop.feature_scales` 仍配置一组 16 维尺度，主循环对各对象使用相同尺度。
模型拟合和独立验证使用相同的对象顺序；不会忽略第二、第三个对象的运动。

`sim.configure` 接收一个 `object_ids` 列表，公共初始场景置于 `task.scene`。
任务适配器的 `validate / execute` 接口不变，但必须在一个物理环境中构造全部对象，
使用模型假设中的各对象物理参数，并返回完整场景轨迹。
`execute(..., purpose='policy')` 的 `success / risk / margin` 必须评价完整任务。
只返回一根长条观测的适配器不能冒充整场景预测，会被 Simulator 拒绝。
物理参数名称、单位、范围仍由该适配器的 `parameter_contracts` 声明。

内置 `three_rod_lift_return` 已实现三根共存长条的物理环境和整任务适配器，见下节。
原 `rod_lift` 保留单杆基线。其他任务可通过 `serve-simulator --adapter MODULE:FACTORY`
与 `serve-measured-target --factory MODULE:FACTORY` 接入。接口 fixtures 不能作为真机实验的证据。

## 旧入口和产物

`run --cases` 在读取输入或调用端点前报错；`run --resume` 也不会偷偷启动旧控制器。
主闭环内部的安全断点继续使用 `resume_probe`；主循环暂不支持进程重启后的整体恢复。
旧 v1 和独立 World 自动闭环、批量命令及导入兼容壳已从包中删除。
已冻结的历史策略和有效实验数据继续保留，不重写其原始验收含义。
显式诊断使用 [debug](debug.md)；只针对同一个 World 执行一次指定操作，不自行选择下一步。

新实验产物统一使用 `physrsr.paths.output_path()`，位于 `/data/wjt/PhyAgent-RSR/output/<实验名>/`。
`task_context/world.json` 记录对象 ID、特征合同及误差尺度；Probe、模型、任务候选和验证仍在对应阶段目录内保存。


## 内置三长条物理任务

- `tasks/rod_scene.py`：任务合同、九个对象级参数、连续保持/放回评分状态。
- `tasks/rod_scene_env.py`：一个 MuJoCo 模型中的三根自由刚体、一张桌面和一套 Panda；保留真实杆间接触。
- `tasks/rod_scene_world.py`：同步几何、原始腕部传感、共享机器人 API 与整场景外部评分。
- `examples/three_rod_lift_return.json`：公开任务与服务端点，无隐藏物理参数、无预置模型。
- `examples/three_rod_world.private.example.json`：仅供 World 服务读取的私有参数示例，不传给 Agent。

对象 ID 固定为 `rod_a / rod_b / rod_c`。每根杆使用 `fixed_mass_com` 惯性模型；
质量、质心与摩擦分别绑定到自己的刚体，九个预测参数名称为 `rod_a__mass_kg` 等。
`parameter_contracts` 同时声明对应的 `object_id / quantity / unit / frame`。
模型修订不会把第一根杆的参数复制给其他杆。

任务要求逐根抓取、抬升并保持近似水平至少 0.5 秒，随后放回各自初始区域，打开夹爪并稳定至少 0.25 秒。
抬升使用几何中心高于桌面 0.04 m、双指真实接触和 `abs(axis_z)<0.25`；名义抬升动作为 0.10 m。
放回要求支撑距离绝对值不超过 3 mm、平面位移不超过 4 cm、线速度不超过 0.01 m/s、角速度不超过 0.05 rad/s，且物体不再被抓持。
多杆同时抓持不能累计独立保持成功。任何一根未保持或未稳定放回，整场景失败。
评分只随物理控制步推进，重复 `observe()` 不累计保持时间。`program_return` 中的自报成功不参与评分。
`world.evaluate_policy` 和 `target.execute_policy` 返回 `task_result`；后者的 `status=completed` 只代表源码运行完成，外部任务成功仍看 `task_result.success`。
终端物体必须已经放下，因此成功裕度使用此前真实保持段的最小高度/水平裕度，而不是终端抬升高度。

分别启动服务（物理运动需要已有 PyRoKi 服务）：

```bash
# 若尚未启动 IK 服务
src/cap-x/.venv/bin/python -m capx.serving.launch_pyroki_server \
  --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand

# 仅 World 服务接收私有配置；不是 --cases
src/cap-x/.venv/bin/python -m physrsr serve-measured-world \
  --task three_rod_lift_return \
  --private-config physrsr/active/examples/three_rod_world.private.example.json
src/cap-x/.venv/bin/python -m physrsr serve-simulator --adapter three_rod_lift_return
src/cap-x/.venv/bin/python -m physrsr serve-agent --out services/three_rod_coding
src/cap-x/.venv/bin/python -m physrsr serve-physic --out services/three_rod_physic

# 单 World 的同一条 ActiveLoop
src/cap-x/.venv/bin/python -m physrsr run \
  --config physrsr/active/examples/three_rod_lift_return.json --out three_rod_run
```

私有 `objects` 配置中，每根杆必须提供 `physics.mass_kg / com_offset / friction`；
可选 `half_size_m`（默认 `[0.15,0.015,0.015]`）、桌面坐标 `table_xy_m`、`yaw_rad`。
桌面坐标仅用于初始布局；公开机器人观测与控制仍使用 `robot_base`。
默认三个位置沿桌面 y 轴相隔 0.18 m，颜色用于视觉区分，与物理参数无关。
重置恢复整个场景，独立 Probe 的恢复检查覆盖三根杆；断点保留搜索进度，不保留旧实时位姿。
预测适配器从 `task.scene` 的公开姿态和尺寸重建整个环境，并使用当前模型参数；
World 私有配置不进入预测服务。固定环境重置种子是控制复现条件，不是另一个真实 World。
