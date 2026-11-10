# Robot-Space M0–M2 实施记录

更新日期：2026-11-10。项目：`/home/wen/experiment/PhyAgent-RSR/active_sysid`。

## 目标

用机器人 Probe 消除完成任务所需的不确定性，更新可交互 Sim，使在 Sim 验证过的开门程序冻结后可以在 World 完成任务。不设计新的 Safe-Probe 算法，不追求恢复真实门锁结构；保留已有机器人控制与 Sim/World 信息边界。

**本轮最小闭环已完成：M0-R 接触可执行 → M1-R 任务条件 → M2-R 交互模型更新 → 同一冻结程序 World 验证。**

## 阶段状态

| 阶段 | 状态 | 本轮完成条件 | 证据 |
|---|---|---|---|
| M0-R：任务接触与初始 Sim | **完成** | 公开机器人示教/Probe 初始化接近与抓握；新 Sim 可复现接触；保留自由运动初始先验 | `runs/robot_grounding_final/m0/completion.json` |
| M1-R：动作前提 Probe | **完成** | 比较相同拉动在旋转前后的响应；功能性条件通过拟合与保留样本 | `runs/robot_grounding_final/m1/task_rule.json` |
| M2-R：任务相关 Sim 更新 | **完成** | 更新有效交互几何，保留 Probe 误差下降；Sim 开门通过；同一冻结程序迁移 World 成功 | `runs/robot_grounding_final/m2/completion.json` |

- [x] 不使用旧 Object-Space M1/M2 参数初始化新模型。
- [x] Probe 只记录 RGB、q/TCP、腕部力和无物体身份的指端接触。
- [x] 不读取 World 的门角、把手角、门栓状态或隐藏约束来更新 Sim。
- [x] 两个参考程序都先完成 Sim 评测和冻结，再运行 World；World 结果未用于后续调参。
- [x] 原始 World 111 个源文件未修改，机器人控制器未重写。

## M0-R：接触链路与模型起点

### 输入及实现

从已验收 R0 的机器人示教中，仅提取 q、TCP、夹爪开度和接触反馈，固化到 `assets/reconstructed_sim/task_grounded_v1/robot_contact_demonstration.json`。这是一条已知接触示教，不是从零视觉定位结果。

用该示教重新执行 World 接近与抓握 Probe。新建任务局部 Sim，复用视觉重建中的机器人与粗门板尺寸，并依据公开接触位置和夹爪开度建立可抓握的杆状代理。它不是原始 World 的完整复制，也不覆盖旧视觉重建；任务之外的房间细节不参与本轮更新。

首次代理门底与地面接触，产生伪阻力；已按原视觉重建中的门板—把手高度差修正。探索日志保留在 `runs/robot_grounding/`，最终证据统一在 `runs/robot_grounding_final/`。

### 结果

| 测量 | 结果 |
|---|---:|
| World 抓握保持阶段双指接触 | 100% |
| Sim 抓握保持阶段双指接触 | 100% |
| Sim/World 夹爪宽度差 | 0.114 mm |
| Sim/World TCP 接触位置差 | 0.102 mm |

模型：`robot_m0`。门可以自由运动，尚无“先转再拉”的条件。

## M1-R：消除动作前提不确定性

### 实际 Probe

抓握不变，比较相同 15 mm 拉动前，分别不旋转、旋转 0.10、0.15、0.35 rad。以上为机器人相对旋转动作，不是读取到的 World 物体角度。

| 前置动作 | TCP 拉动跟随率 | 腕部力变化 | 观测结论 |
|---|---:|---:|---|
| 不旋转 | 0.801 | 16.97 N | 受阻 |
| 相对旋转 0.10 rad | 0.805 | 15.46 N | 仍受阻 |
| 相对旋转 0.15 rad | 1.018 | 1.26 N | 可拉动 |
| 相对旋转 0.35 rad | 1.021 | 0.33 N | 可拉动 |

本轮用“保持双指接触且小拉动跟随率大于 0.95”定义可拉动标签。Probe 给出的任务条件变化区间是 **[0.10, 0.15] rad**。

### Sim 更新

在 `robot_m1` 中加入功能性条件：相对把手运动超过有效阈值后允许门运动。最终模型阈值取区间中点 **0.125 rad**。这是任务局部模型坐标中的有效条件，不是恢复了 World 的真实门栓或解锁结构。

| 预测正确率 | `robot_m0` | `robot_m1` |
|---|---:|---:|
| 拟合 Probe：4 个 | 2/4 | 4/4 |
| 保留 Probe：2 个 | 1/2 | 2/2 |

保留 Probe 改为 10 mm、1.0 s 拉动，其中一个不旋转，另一个先旋转 0.45 rad；它们未用于选取条件区间。

M1 解决的是动作前提，尚未精确匹配所有接触力幅值。

## M2-R：更新对策略有用的交互响应

### 拟合对象

先旋转后，分别执行半径 0.65、0.75、0.85 m 的候选拉动圆弧。只比较公开腕部力变化与 TCP 轨迹，不辨识隐藏铰链/门栓内部结构。

在 Sim 中以五个候选有效半径拟合这些响应：0.65、0.70、0.75、0.80、0.85 m。目标是减少末端位置与腕部响应残差，不是让每个参数等于 World 真值。

最终：

- Sim 的有效拉动半径参数：**0.80 → 0.70 m**。
- 根据更新后 Sim 的响应排序，固定验证程序选择的圆弧动作半径：**0.75 m**。

前者是模型参数，后者是动作参数，两者不是同一个量。门质量、弹簧刚度、阻尼和摩擦仍是本轮保留的先验；没有宣称它们均已完成辨识。

### 保留 Probe 结果

使用未参与参数选择的两条圆弧：0.75 m/0.30 rad，以及 0.65 m/0.10 rad。

| 指标 | M1 模型 | M2 模型 |
|---|---:|---:|
| 腕部力变化的平均绝对预测误差 | 5.882 N | 1.712 N |

本轮两条保留 Probe 的平均误差下降 **70.90%**。长圆弧的改进明显；短圆弧的力幅值误差并非每项都改善，逐条值保存在 `m2/completion.json`。任务状态预测与完整开门另外验收，不用一个平均误差代替任务表现。

## 冻结迁移验证

使用两个自包含固定验证程序，不运行 Coding Agent：

1. `reference_initial_direct_pull.py`：根据初始自由运动 Sim，抓握后直接拉门。
2. `reference_grounded_turn_pull.py`：根据 Probe 更新后的 Sim，先旋转再沿选出的圆弧拉门。

两个程序分别在对应 Sim 通过，**全部冻结后才开始任何 World 测试**。此后没有修改模型或程序。

| 结果 | 初始 Sim：直接拉门 | 更新 Sim：旋转后拉门 |
|---|---:|---:|
| Sim 执行 | 通过 | 通过 |
| Sim 最终门角 | 0.478278 rad | 0.474470 rad |
| World 执行 | 未打开 | **成功** |
| World 最终门角 | 约 0 rad | **0.468982 rad** |
| World 操作阶段双指接触率 | 15.77% | **100%** |
| World 把手接触力幅值之和峰值 | 57.68 N | **22.71 N** |
| World 非预期环境接触力峰值 | 0 N | 0 N |
| World 直接物体驱动 | 0 | 0 |
| World 时间 | 12.8 s | **15.8 s** |
| Sim/World 策略 SHA256 | 一致 | 一致 |

任务成功依据是机器人通过接触将门打开到目标角度 ≥0.30 rad，不要求达到隐藏门栓行程。力数值是实验记录，不等价于经过认证的损伤限值。

这是一组 exact 条件下的固定程序验证，不是 Coding-Agent 多次生成成功率或通用视觉 CaP 的结论。RGB 在本轮用于过程记录；任务定位由共同的接触示教初始化。

总结果：`runs/robot_grounding_final/completion.json`。

## 代码和模型

| 文件 | 职责 |
|---|---|
| `src/active_sysid/grounding/task_interaction.py` | 最小 Probe 程序生成、公开证据归纳、任务局部 Sim 构建 |
| `scripts/run_robot_grounding.py` | 按 M0/M1/M2 复现并更新本文件中的阶段状态 |
| `configs/robot/simulators/robot_grounding_m0_m2.yaml` | 三个新模型与同一机器人接口的绑定 |
| `assets/reconstructed_sim/task_grounded_v1/robot_m0.xml` | 初始接触可执行模型 |
| `assets/reconstructed_sim/task_grounded_v1/robot_m1.xml` | 任务条件更新模型 |
| `assets/reconstructed_sim/task_grounded_v1/robot_m2.xml` | 经 Probe 更新的交互模型 |
| `assets/reconstructed_sim/task_grounded_v1/grounding_provenance.json` | 证据来源、更新量与未辨识先验 |
| `policies/robot_space/reference_grounded_turn_pull.py` | 冻结迁移使用的固定验证程序 |

现有 `SimRobotBackend` 只增加新模型的加载分支；机器人控制器和 World 后端不重写。旧 `initial/m1/m2` 为 Object-Space 历史对照；新主线是 **`robot_m0/robot_m1/robot_m2`**，三个 Stage 默认使用 **`robot_m2`**。

## 复现命令

从包根运行。每次使用新的输出目录，不覆盖已完成证据。

```bash
cd /home/wen/experiment/PhyAgent-RSR/active_sysid

# 复用已记录的公开 Probe，重新执行 M0–M2 的 Sim 更新与验证。
MUJOCO_GL=egl .venv/bin/python scripts/run_robot_grounding.py all \
  --out runs/robot_grounding_replay \
  --probes runs/robot_grounding/probes

# 去掉 --probes 则重新执行 World Probe；也可单独运行 m0/m1/m2。

# 运行更新后 Sim 中的固定参考程序。
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run \
  --stage visual_exact --sim-variant robot_m2 \
  --policy policies/robot_space/reference_grounded_turn_pull.py \
  --out runs/new_reference/sim

# 根据该 Sim 报告冻结策略。
.venv/bin/python -m active_sysid.robot.sim2sim freeze \
  --policy policies/robot_space/reference_grounded_turn_pull.py \
  --sim-summary runs/new_reference/sim/summary.json \
  --out runs/new_reference/frozen

# World 仅执行同一冻结版本。
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.sim2sim evaluate-world \
  --bundle runs/new_reference/frozen --out runs/new_reference/world --video

.venv/bin/python -m active_sysid.robot.stages check
.venv/bin/python -m pytest -q
```

## 已保存录像与最终验证

成功 World 录像：

```text
runs/robot_grounding_final/transfer/grounded_turn_pull/world/evaluator_private/overview.mp4
runs/robot_grounding_final/transfer/grounded_turn_pull/world/evaluator_private/robot0_eye_in_hand.mp4
```

同目录包含真实物理轨迹 `trajectory.npz`、动作记录 `actions.json` 与独立裁判 `metrics.json`。Probe 的公开证据单独保存在 `runs/robot_grounding/probes/`，不混入 World 终评真值。

本轮核心回归：**55 项测试通过**。新增测试只保留新模型版本/来源完整性和更新 Sim 的完整机器人接触开门两个核心检查。原始 World 资产校验仍为 **111/111 一致**。代码未提交 Git。
