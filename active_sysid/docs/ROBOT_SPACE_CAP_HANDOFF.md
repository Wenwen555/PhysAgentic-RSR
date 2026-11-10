# 新会话交接：Robot-Space CaP

项目：`/home/wen/experiment/PhyAgent-RSR`；包根：`active_sysid/`。

**当前主线已完成机器人 Probe → 任务级条件 → Sim 交互响应更新。完整实施记录以 [ROBOT_SPACE_M0_M2_IMPLEMENTATION.md](ROBOT_SPACE_M0_M2_IMPLEMENTATION.md) 为准。**

## 当前模型

- `robot_m0`：以公开机器人接触示教初始化的任务局部几何；门可直接运动的初始先验。
- `robot_m1`：根据相同拉动在旋转前后的差异，加入“相对转动后可拉动”的任务条件。
- `robot_m2`：在上述条件上，用腕部力和 TCP 轨迹更新有效拉门几何，独立 Probe 验证后作为主开发 Sim。

这三个模型不是旧 Object-Space M1/M2 的改名或参数复制。旧 `initial/m1/m2` 仍保留为历史对照。当前三个 Stage 的默认模型为 `robot_m2`。

核心资产：`assets/reconstructed_sim/task_grounded_v1/`。其中 `robot_contact_demonstration.json` 只包含已验收机械臂示教的 q/TCP/夹爪信息；`grounding_provenance.json` 列出 Probe 来源与真正更新的参数。模型是任务局部代理，不宣称重建了完整房间或真实门栓。

## 先运行

```bash
cd /home/wen/experiment/PhyAgent-RSR/active_sysid
.venv/bin/python -m active_sysid.robot.stages check
.venv/bin/python -m pytest -q
```

三个开发 Stage 仍遵守 Sim/World 边界：`privileged_exact` 只读取 Sim 内部状态；`visual_exact`、`visual_randomized` 只读 RGB、机器人本体反馈和不带物体身份的指端接触。World 只用于显式 Probe 与冻结策略终评。详细权限见 [SIM_WORLD_BOUNDARY.md](SIM_WORLD_BOUNDARY.md)。

## 下一会话的工作

在 `policies/robot_space/` 写自包含 `run(api)` 策略，复用机器人控制器与实验入口，不修改模型/裁判以迁就策略。已有 `reference_grounded_turn_pull.py` 和 `reference_initial_direct_pull.py` 是固定程序验证/对照，不是 Coding Agent 开发的通用视觉 CaP。

```bash
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run \
  --stage visual_exact --sim-variant robot_m2 \
  --policy policies/robot_space/YOUR_POLICY.py --out runs/new_cap/sim
```

策略从 Sim 视觉报告冻结后才进入 World。`observe()['privileged']` 不会在 World 出现。策略运行在隔离进程中，文件必须自包含，不要导入项目内 helper 模块。

## API

`observe(rgb=True, camera='overview'|'robot0_eye_in_hand')`、`phase`、`move_joints`、`move_ee`、`move_arc`、`set_gripper`、`hold`。位置使用米、世界坐标；角度使用弧度；四元数使用 wxyz。`finger_contact_normal_n` 是指端所有外部接触的合计，不是“已抓到目标”的身份标签。

任务目标是利用足够的观测消除策略不确定性，不要求发现真实门锁结构。当前局部试验复用已知接触示教，不声称已经解决从零视觉定位或任意场景泛化。
