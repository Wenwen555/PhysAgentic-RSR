# Robot-Space Sim2Sim 边界（v2）

## 现在的执行路径

```text
已有视觉重建 + 机构模型版本（Initial / M1 / M2）
  → 独立 Sim → Probe 预演 / privileged_exact / visual_exact / visual_randomized
  → 保存 Sim 日志 → 冻结一个完整策略文件 → 原始 World 独立评测

需要实测证据时：
  Probe 程序 → 显式 world_probe → RGB / 本体感受 / 无物体身份的指端接触
  → 证据用于后续 Grounding（不返回隐藏门栓、机构参数或真值位姿）
```

`stages.run` 和 `stages.run-all` **只运行 Sim**，没有自动调用 World 的分支。World 评测不会自动修订策略、随机化 World 或回填 Sim 参数。旧的 World-only Stage 结果不是 Sim2Sim 迁移结果；原 R1 仍是独立的 Oracle 执行器验收。

## 三个模型角色

| 角色 | 加载资产 | 对策略的观测 | 输出 |
|---|---|---|---|
| `development_sim` | `assets/reconstructed_sim/visual_stage3_v1/scene.xml` | 机器人传感器；按 Stage 开放 Sim 自身状态或 RGB | Sim 开发日志和评分 |
| `world_probe` | 原始 `assets/reference_world/unidoor_native/scene.xml` | RGB、机器人本体感受、指端接触，无目标身份 | `probe.json`、`observations.npz`、动作与图片 |
| `world_evaluation` | 同一个未改动的原始 World | 与视觉 Sim 策略相同的传感器接口 | 冻结策略评分；真值轨迹在 `evaluator_private/` |

只有 `privileged_exact` 可读取 `obs['privileged']`，其中数据来自 **Sim 的 MjModel/MjData**。其他两个 Stage 不返回特权状态。原来的 `world_fixed_policy.py` 保留为 Oracle 控制能力诊断，不作为新 CaP，也不进入 Sim 开发链路。

## Sim 来源与当前成熟度

几何来自已有 `output/unidoor_real2sim_20261108/stage3_final/scene.xml`。构建工具将其 84 个依赖打成独立本地包，仅重定位资源路径、设置渲染缓冲、将重建 camera 0 命名为 `overview`，并添加默认关闭的功能性锁约束槽。它不读取原始 World XML，也不复制门栓或原 World 的隐藏耦合机构。

`initial` 保留视觉重建的无锁先验。`m1` 和 `m2` 使用已有 Object-Space 结构/参数修复产物，通过独立功能性锁和标量铰链惯量映射接入。这些参数是**旧 Object-Space 辨识的迁移初值，不是本次双门 Robot-Space 接触辨识结果**。各版本共用相同机器人、控制器和重建几何。

当前实际发现的几何问题已记录在 `assets/reconstructed_sim/visual_stage3_v1/provenance.json`：重建杠杆只有 hub 几何，未包含完整杆体；环境相对机器人的位姿与 World 有明显差异。此版本已经通过加载、关节运动、RGB 和权限测试，**尚未通过重建 Sim 内的完整 R1 抓握开门验收**。不能把原 World 的 R1 成功当作 Sim 的 R1 成功。补几何和配准必须使用重建/标定或许可 Probe 证据，不能读取 World 真值填入。

## 不再通过 Python 私有属性读取 World

新 CaP 在 bubblewrap 子进程中执行。仅挂载 Python 运行环境、已安装依赖、worker 脚本和本次策略副本；不挂载项目、World 资产、原用户 HOME、历史 runs 或网络。主进程通过 JSON/数值数组 RPC 暴露允许的动作和观测。没有 `api._backend` 或 MuJoCo 模型对象。缺少 bubblewrap 时明确停止，不退回同进程执行。

公开观测：`time_s, q, dq, joint_torque_nm, ee_pos_m, ee_quat_wxyz, gripper_width_m, wrist_wrench_ee, finger_contact_normal_n`；显式 `rgb=True` 时增加 RGB。指端接触是对外部接触的合计，不再标注是否接触目标把手。门角、把手角、门栓、闩锁、接触物体 ID 和参数均不属于 World 公开观测。

`finger_contact_normal_n` 是基于仿真碰撞计算的触觉读数；实际机器人需要相应的触觉或接触估计。两侧名为 `overview` 的观测并不保证相同标定：Sim 使用已有重建 camera 0，World 使用原有 overview。腕部相机来自共享机器人模型；本次没有读取 World 相机真值去校准重建相机。力阈值是实验中止条件，不是经过认证的损伤界限。

Coding Agent 的 DevSpace 管理权限与 CaP 子进程不同：管理整个仓库的 Agent 仍有能力主动读取私有文件。因此开发会话应只使用 Sim 日志和公开 Probe 证据，不读取 `evaluator_private/` 或原 World 机构源码来修订策略。子进程隔离不等于限制了用户整个 DevSpace 账户。

## 可复用命令

从 `active_sysid/` 运行；每次使用新的输出目录，避免覆盖证据。

```bash
.venv/bin/python -m active_sysid.robot.stages check

# 三个 Stage 均只使用重建 Sim；initial/m1/m2 是独立的模型版本轴。
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run \
  --stage privileged_exact --sim-variant m2 \
  --policy policies/robot_space/privileged_policy.py --out runs/new_experiment/sim_privileged
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run \
  --stage visual_exact --sim-variant m2 \
  --policy policies/robot_space/visual_policy.py --out runs/new_experiment/sim_visual
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run \
  --stage visual_randomized --sim-variant m2 \
  --policy policies/robot_space/visual_policy.py --seeds 0 1 2 3 4 5 6 7 \
  --out runs/new_experiment/sim_randomized

# 真实交互证据入口；不执行任何自动机构拟合。
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.sim2sim probe-world \
  --policy policies/robot_space/probe_policy.py --out runs/new_experiment/probe_01

# 必须先有同一策略 SHA256 的 Sim 视觉评测报告。
.venv/bin/python -m active_sysid.robot.sim2sim freeze \
  --policy policies/robot_space/visual_policy.py \
  --sim-summary runs/new_experiment/sim_randomized/summary.json \
  --out runs/new_experiment/frozen_policy

# 原始 World、无特权观测、无环境随机化、无自动修订。
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.sim2sim evaluate-world \
  --bundle runs/new_experiment/frozen_policy --out runs/new_experiment/world_final --video
```

`robot world-evaluate --policy ...` 已关闭；改用 `--bundle` 或上述 `sim2sim evaluate-world`。特权 Sim 策略不能直接冻结后向 World 请求特权状态；应先完成使用公开观测的策略。允许冻结失败的 Sim 策略作为负对照，manifest 会明确标记。

## 接口和记录

策略是一个自包含 Python 文件，入口 `run(api)`；可使用环境中已安装的数值/视觉库，不依赖项目内 helper 模块（项目不挂载）。动作仍是 `move_joints, move_ee, move_arc, set_gripper, hold`，`phase` 只记标签。Cartesian 坐标单位米，角度弧度，四元数 wxyz。`overview` 与 `robot0_eye_in_hand` 可读取 RGB。视觉评分要求不同时刻读取图像，但这不证明策略真正利用了像素内容，仍需视觉干预实验。

Sim 模型、源码、配置和运行环境由 `configs/robot/frozen_sim2sim.json` 固定；World 的 111 个源文件单独固定，只有显式 Probe/World 评测入口才校验。旧 `frozen_r1.json` 和 Stage 代码已保留在 `docs/archive/world_only_stage_v1/`，未冒用旧版本号覆盖。

本次实现的是执行与信息边界、Probe 证据出口、Sim 开发和冻结迁移入口。自动选择 Probe、从证据辨识新参数、几何补全和新 CaP 算法不属于本次实现。

## 本次验证证据

`runs/sim2sim_boundary_v2/validation.json` 汇总本次结果；`pytest_final.txt` 记录 53 项测试通过。原 World R1 复测仍成功（0.399582 rad），不转把手的负对照仍失败。`acceptance/sim_three_stages/` 为只读图像和传感器的接口检查，不是开门 CaP 实验；其中 `world_evaluations_performed` 为 0。`acceptance/probe_public/` 仅含公开证据，`acceptance/world_negative_control/` 为同一冻结观察脚本的独立 World 负对照。

## 当前 Robot-Space Grounding 主线

后续 M0–M2 已使用新的公开机器人 Probe 完成任务局部更新，主模型为 `robot_m0/robot_m1/robot_m2`，默认 Stage 改为 `robot_m2`。旧 `visual_stage3_v1` 的 hub/配准缺陷仍保留在历史模型中；新模型通过机器人接触示教与 Probe 证据建立可操作局部代理，没有用 World 真值覆盖旧重建。最新状态、命令与结果见 `ROBOT_SPACE_M0_M2_IMPLEMENTATION.md`。

新的任务验收要求机器人通过接触将门打开到目标角度，不再要求拟合隐藏门栓行程或“真实解锁”内部状态。Oracle R1 诊断仍保留原指标，但不作为新任务模型的结构约束。
