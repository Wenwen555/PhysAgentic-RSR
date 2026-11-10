# M3-R：Panda 机器人执行层 R0 + R1

## 交付结果

本包实现 **Panda 真实关节伺服 → 双指接触 → 把手旋转 → 闩锁释放 → 门打开** 的 MuJoCo 执行链路。

- R0：完整机器人模型加载、实际关节运动、双指接触验收。
- R1：固定程序显式完成接近、抓握、转把手、拉门、执行后检查。
- 已接通 RobotRuntime / Adapter，提供独立 Robot-Space policy 评测入口。
- `privileged_exact → visual_exact → visual_randomized` 的 Coding Agent 开发与重新冻结属于后续 R3。本包没有运行或宣称完成这些实验。

这次修改全部为新增文件。上传 ZIP 的 **73 个原文件逐字节保持一致**，包括 `mujoco_backend.py`、`latch.py`、两个 Object-Space MJCF、原配置、原测试、`pyproject.toml` 和 `uv.lock`。

## 1. 安装与运行

将交付 ZIP 解压到 **PhyAgent-RSR 仓库根目录**。ZIP 内的 `active_sysid/` 与现有目录合并。

```bash
unzip PhysAgentic_RSR_M3R_R0_R1.zip -d /home/wen/experiment/PhyAgent-RSR
cd /home/wen/experiment/PhyAgent-RSR/active_sysid

uv venv --python 3.11 .venv-robot
uv pip install --python .venv-robot/bin/python -e . -r requirements-robot.txt
```

机器人环境独立使用 `.venv-robot`；依赖安装不需要修改项目的 `pyproject.toml` 或 `uv.lock`。Panda 模型和全部网格已随包提供，运行时不再下载 Menagerie。

先运行 R0：

```bash
PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python scripts/robot_scene_smoke.py \
  --out runs/m3r_r0
```

再运行 R1 并录制两路视频：

```bash
PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python scripts/robot_manipulation_baseline.py \
  --out runs/m3r_r1 \
  --video
```

退出码 `0` 且 JSON 中 `status="pass"`、`failed_required_gates=[]` 表示该阶段通过。失败会记录实际异常和未通过门槛，并返回非零退出码。没有 `--video` 时执行相同物理程序，只省去渲染。

输出目录包含：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 最终任务结果、每项验收、力/力矩/碰撞/漂移统计、源码和模型 SHA256 |
| `trajectory.npz` | 每个 2 ms 物理步的数值记录；不需要 `allow_pickle=True` |
| `actions.json` | 每个 Robot API 调用、输入、阶段、起止时间及完成/失败状态 |
| `overview.mp4` | 完整 Panda 与门的动作 |
| `contact.mp4` | 指尖与把手的接触近景 |
| `*_first.png`、`*_last.png` | 两路视频首尾状态图 |

`--video` 采用仿真时间采样：25 fps，0.002 s 物理步；每 20 个物理步输出一帧。完整固定程序 7600 步 / 15.2 s，对应 380 帧 / 15.2 s。成功判定读取真实角度、接触和受力，不依据固定步数或动画播放结束。

## 2. 固定程序与控制器边界

任务顺序全部位于 `policies/robot/fixed_open_door.py`：

1. 稳定机器人，读取 R1 privileged 把手位姿。
2. 到达把手前方 0.17 m 的预抓取位姿。
3. 接近，把 TCP 对准抓取点，检查实际位置与姿态误差。
4. 夹爪闭合至 0.006 m 的目标宽度，检查两指实际接触。
5. 围绕把手轴执行通用末端圆弧，目标把手角 0.85 rad。
6. 围绕门铰链执行 0.45 rad 的通用末端圆弧。
7. 保持并重新观测，确认门角达到 0.30 rad。

`RobotExecutionAdapter` 只有末端位姿、通用圆弧、夹爪开度和保持命令；没有 `open_door()`、`grasp_handle()` 或自动完成任务的 Object-Space 适配逻辑。

底层采用阻尼最小二乘 IK、五次平滑时间插值、关节位置 PD 与偏置补偿。IK 使用独立 `MjData`，执行时只更新机器人 actuator 的 `ctrl`。手臂执行器力矩限制为前四轴 ±87 N·m、后三轴 ±12 N·m。夹爪保留 Menagerie 的 tendon / 双指耦合，提高位置伺服刚度以形成有效夹持力。

接触控制的当前实现是 **关节 PD 顺应伺服 + 夹爪位置伺服 + 实测接触力超限中止**。本包的 80 N 是新实验中“机器人与把手各接触力幅值之和”的验收阈值，不是沿用旧 Object-Space 的 20 N 拉力命令，也不是材料损伤界。阈值在物理步完成后检测。

## 3. 场景与 M2 参数的关系

旧把手与 Z 轴共轴、嵌于门面并关闭碰撞，无法直接作为本任务的抓握目标。新的 Robot 场景使用：

- 12 kg Oracle 门叶，原门铰链范围、阻尼与摩擦。
- 伸出门面的可碰撞杠杆把手；转轴为门面法向。
- 把手质量中心位于转轴，采用平衡把手设定，避免新增重力矩使 M1 无弹簧模型自行解锁。
- 共同的新把手惯量、接触几何与摩擦参数。
- 原功能性 latch 语义：把手达到 0.556 rad 才解除 door–frame weld；接近关门且把手回落才重新锁定。
- Panda 与门/把手之间不创建 weld、connect 或 mocap 约束。

这是一个新增 Robot-Space 任务，复用机构的标量参数并增加接触层。改变了把手轴、几何和接触负载，因此旧 M2 的 `0.000151 rad` MAE 应继续作为旧 Object-Space 实验结果保存；Robot-Space 的误差和任务表现由新实验重新测量。

## 4. 三种 Simulator 使用同一机器人执行层

本次上传包没有 M1/M2 的 `runs/` 拟合产物，所以不提供虚构的已拟合参数。`--mechanism-mjcf` 从你本机实际产物导入参数，并自动读取同名 `.repair.json` sidecar。

```bash
# Initial：原始重建模型
PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python scripts/robot_manipulation_baseline.py \
  --mechanism-mjcf worlds/reconstructed/reconstructed_door.mjcf \
  --out runs/m3r_initial

# Structure-only：替换为你实际 M1 输出路径
PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python scripts/robot_manipulation_baseline.py \
  --mechanism-mjcf runs/YOUR_M1_RUN/repaired_door.mjcf \
  --out runs/m3r_structure_only

# Structure + parameters：替换为你实际 M2 输出路径
PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python scripts/robot_manipulation_baseline.py \
  --mechanism-mjcf runs/YOUR_M2_RUN/repaired_m2.mjcf \
  --out runs/m3r_structure_parameters
```

导入的是两个对象关节的阻尼、摩擦、弹簧、armature、范围与 springref，以及门体质量/惯量和 latch 配置。Panda、碰撞几何、把手独立质量/惯量、控制器、相机和任务条件保持同一设定。来源 MJCF 和 sidecar 的 SHA256 会写入结果。

内置 `--scene reconstructed` 使用与 Initial 对应的机构参数。`--scene oracle` 是默认真值执行场景。直接加载与导入同源机制的门体质量一致：Oracle 12 kg、Initial 0.2496 kg。

这些运行用于检查执行层与机制导入；后续 CaP 的 Simulator 对比仍需要相同开发预算下各自开发，再部署到共同 Oracle 评测。

## 5. R0 / R1 的独立验收

| 项目 | R1 判定 |
|---|---|
| Reach | 实际末端与日志中 Reach 目标的位置误差 ≤0.01 m，姿态误差 ≤0.10 rad |
| Grasp | grasp 最后 0.2 s 至少 95% 步中，两指法向力均 >0.5 N |
| Rotate | 实际把手越过解锁阈值，并在旋转阶段观测到 latch 释放 |
| Pull / Verify | 实际最终门角 ≥0.30 rad |
| 持续接触 | rotate + pull 至少 95% 物理步保持双指接触 |
| 平移漂移 | 把手相对 TCP 的抓取点漂移 ≤0.01 m |
| 旋转漂移 | 把手相对 TCP 姿态变化 ≤0.15 rad |
| 受力 | 聚合把手接触力 ≤80 N；手臂力矩处于模型范围内 |
| 碰撞 | 非预期环境接触及自接触的单点力不超过 5 N |
| 对象直接驱动 | 对象 qfrc_applied / qfrc_actuator / xfrc_applied 全程为零；执行器仅连接机器人；无机器人–对象约束、无 mocap |

Reach 在 evaluator 内根据动作日志与实际采样重新计算。最终门角和其余验收均直接来自物理记录，策略自己返回的成功标志只作诊断。

平移/旋转漂移参考为 grasp 中最后一个连续至少 0.2 s 的双指接触区间。它衡量把手与夹爪的相对位姿变化，不是接触表面的滑动速度。

力统计分别记录每指法向力、每个接触点峰值、全部把手接触力幅值之和、把手净合力，以及腕部传感器六维力/力矩。不能将这些不同口径混用。

## 6. 新 Robot-Space CaP 接口

新策略文件导出 `run(api)`。默认 `robot` 观测包含 RGB（显式请求）、机器人关节/速度/力矩、TCP、夹爪和接触反馈，不返回门角、把手角、轴线、机构位姿、latch 或抓取点相对位姿。

```python
observation = api.observe(rgb=True, camera="overview")
api.move_ee(position_m, quaternion_wxyz, duration_s=1.5)
api.set_gripper(width_m=0.006, duration_s=0.8)
api.move_arc(center_m, axis_world, angle_rad, duration_s=3.0)
observation = api.hold(0.2)
```

位置、轴和四元数使用世界坐标；长度 m、角度 rad、时间 s，四元数顺序 wxyz。`move_arc` 围绕调用者提供的轴运动并同步旋转末端，不读取对象真值来补全任务。

评测新的策略：

```bash
PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python scripts/eval_robot_cap.py \
  --policy policies/robot/YOUR_NEW_POLICY.py \
  --observations robot \
  --out runs/m3r_new_policy \
  --video
```

用固定 R1 策略检查这条独立评测入口：

```bash
PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python scripts/eval_robot_cap.py \
  --policy policies/robot/fixed_open_door.py \
  --observations privileged \
  --out runs/m3r_fixed_api_eval
```

`task` 模式按实际开门、受力、碰撞和对象零直驱判定，不要求新策略沿用固定程序的 phase 名称。旧 `api.step(handle=..., door=...)` 不能传给此 API。旧 Stage 3 冻结策略的层级迁移对照和新 Robot-Space CaP 的重新开发属于不同实验。

## 7. 核心验证命令

```bash
uv pip install --python .venv-robot/bin/python pytest

PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python -m pytest \
  tests/test_robot_execution.py tests/test_robot_mechanism.py -q

PYTHONPATH=src:. MUJOCO_GL=egl \
.venv-robot/bin/python -m pytest \
  tests/test_mujoco_backend.py tests/test_m0_baseline.py \
  tests/test_m1_structure.py tests/test_m3_runtime.py \
  tests/test_m2_parameters.py -q -k 'not test_m2_end_to_end'
```

新增核心验证覆盖实际接触开门、关闭接触后无法抓握、不转把手时保持锁定、机器人观测边界和 M2 参数映射。旧回归覆盖原 M0/M1/运行时/参数映射；本次不重新执行完整 M2 参数搜索。

`runs/m3r_reference/` 随包提供本次实测产物。具体数值见其中 R0/R1 的 `metrics.json`，不要将单个固定场景的验收结果当作随机化成功率。

## 8. 资产与实现参考

- [官方 Panda 固定版本](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda)
- [MuJoCo Jacobian 与接触力 API](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html)
- [MuJoCo 碰撞与 MJCF 参考](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html)

全部上游 Panda 文件及 Apache-2.0 LICENSE 随包保留；`worlds/robot/NOTICE.md` 记录派生模型修改。可运行 `python scripts/build_robot_scenes.py` 从所附固定模型重新生成三份 Robot XML。
