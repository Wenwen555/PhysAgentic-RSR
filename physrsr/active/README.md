# PhysRSR Active Identification

任务驱动的主动辨识模块：从任务和视觉输入开始，先由 Physic Agent 分析不确定性并执行 Probe，再生成与修订任务程序。
仿真和真机使用同一份 policy / probe 源码，观测和动作经过公开 API。

自动闭环使用 `run` 命令：首次 Probe 前不生成 CaP，也不运行未辨识模型的预测。
真实交互证据获得后，再生成策略并决定下一轮探索或修订。
单实例状态机提交与验证预测模型；多实例共享源码分支当前使用 World 实测反馈，未包含预测 Simulator 更新验证。
`identification` 命令保留为分阶段检查入口，自动运行不需要外部选择下一阶段。
任务名称、隐藏参数和任务专用动作模板不进入 Physic Agent 的系统提示。
接口和数据流见 [通用辨识说明](docs/identification.md)。
修复与交互的分离、逐实例证据判断、多设置与断点续执行见 [实验调度合同](docs/exploration.md)。

## 目录职责

```text
active/
  __main__.py                 # 统一命令行入口
  core/                       # 数据合同、policy 版本、推断与通用产物工具
  api/                        # HTTP 客户端、服务托管、进程生命周期和调用日志
  agents/                     # Coding Agent / Physic Agent、输出 schema 与源码检查
  runtime/                    # 机器人、仿真与真机适配及隔离执行对接
  tasks/                      # 任务接口、注册与各任务的预测/World 适配
  workflows/                  # 首版生成、轨迹组织、Probe 编写、闭环和单 policy 评估
  prompts/                    # 初版 CaP 与 Physic Agent 的 prompt 模板
  examples/                   # 可运行配置
  docs/                       # 架构、API 与流程说明
```

| 需要修改的行为 | 主要入口 |
| --- | --- |
| 任务、预算和物理模型合同 | `core/contracts.py` |
| policy 存档、哈希与统一执行入口 | `core/policy.py` |
| 模型更新、Probe 决策收益和停止判断 | `core/reasoning.py` |
| HTTP API 与服务启动 | `api/client.py`、`api/server.py`、`api/process.py` |
| 首版代码生成服务 | `agents/coding.py` |
| Physic Agent 分析及 Probe schema | `agents/physic.py` |
| 通用实验设计、估计输出与源码合同 | `agents/identification.py`、`core/identification.py` |
| 机器人及真机观测对接 | `runtime/robot.py`、`runtime/backends.py` |
| v2 模型提交、任务适配器与预测 API | `runtime/predictive.py` |
| 任务选择与注册 | `tasks/registry.py` |
| Rod Lift 参数、判据与预测适配 | `tasks/rod_lift.py` |
| Rod Lift 仿真传感和 World 评分 | `tasks/rod_lift_world.py` |
| 历史 v1 Simulator API | `runtime/simulator.py` |
| 私有 World Probe 与原始任务评估 API | `runtime/world.py` |
| 通用 v2 本体、腕部力/力矩、编码器与真实保持动作 | `runtime/instrumented.py` |
| 初版 CaP 生成 | `workflows/initial_cap.py` |
| 存档观测和命令回放 | `workflows/evidence.py` |
| Probe 编写、分析存档与源码复核 | `workflows/probe_authoring.py` |
| World Probe → 一次共享 CaP 修订 → 配对评估 | `workflows/probe_revision.py` |
| 通用辨识 → 实例估计记忆 → 共享策略生成与执行 | `workflows/identification.py` |
| 在线闭环状态机 | `workflows/loop.py` |
| 多实例、共享源码、实测反馈的自动闭环 | `workflows/batch.py`、`workflows/scoped_loop.py` |
| 实例范围、决策收益理由与断点合同 | `core/exploration.py` |
| 历史 v1 状态机 | `workflows/legacy_loop.py` |
| 一份 policy 的固定质量对照 | `workflows/smoke.py` |

## 使用

从仓库根目录运行。帮助和配置检查不会启动实验：

新实验统一位于 `/data/wjt/PhyAgent-RSR/output/<实验名>/`。
`--out` 和 `--experiment` 接受相对此目录的实验名或目录内的绝对路径，不随工作目录改变。
输入文件参数（例如 `--config`、`--initial-cap`）仍使用其实际路径。

```bash
src/cap-x/.venv/bin/python -m physrsr.active --help
src/cap-x/.venv/bin/python -m physrsr.active validate-config \
  physrsr/active/examples/rod_lift.json
```

准备好 API 服务后，显式选择工作流：

```bash
# 历史 task-only 对照入口，不是自动闭环起点
src/cap-x/.venv/bin/python -m physrsr.active author-initial \
  --config physrsr/active/examples/rod_lift.json --out example_initial

# 整理已有初版评估记录，编写 Probe；不执行 Probe
src/cap-x/.venv/bin/python -m physrsr.active author-probe \
  --experiment example_run --round 1

# 同一份 policy 在三个质心配置下各执行一次，总质量与惯量固定
src/cap-x/.venv/bin/python -m physrsr.active smoke \
  --out example_evaluation --initial-cap /data/wjt/PhyAgent-RSR/output/example_initial --mass-kg 0.27

# 对已有固定质量三配置存档，只处理其中两个失败场景。
# 以下三个命令分别执行 Probe、调用一次 Coding Agent、配对复跑。
# 每阶段拒绝覆盖已有产物；一次修订得到一份共享 policy。
src/cap-x/.venv/bin/python -m physrsr.active probe-world \
  --experiment example_run --round 1
src/cap-x/.venv/bin/python -m physrsr.active revise-cap \
  --experiment example_run --round 1
src/cap-x/.venv/bin/python -m physrsr.active compare-caps \
  --experiment example_run --round 1

# 运行在线主动辨识状态机
src/cap-x/.venv/bin/python -m physrsr.active run \
  --config physrsr/active/examples/rod_lift.json --out example_loop

# 多个独立 World 实例共用一份 policy；从视觉开始，自动执行与修订
src/cap-x/.venv/bin/python -m physrsr.active run \
  --config physrsr/active/examples/rod_lift.json --cases evaluator_only_cases.json \
  --out example_shared_probe_first

# 通用辨识：公共上下文和私有 World 配置分别提供。
# 公共上下文只含语义目标、几何、观测、本体与匿名证据；不需要初版 policy。
src/cap-x/.venv/bin/python -m physrsr.active identification design \
  --experiment example_identification \
  --public-context public_context.json --private-cases evaluator_only_cases.json
src/cap-x/.venv/bin/python -m physrsr.active identification probe \
  --experiment example_identification
src/cap-x/.venv/bin/python -m physrsr.active identification synthesize \
  --experiment example_identification
src/cap-x/.venv/bin/python -m physrsr.active identification evaluate \
  --experiment example_identification

# 用已有交互重新设计辨识实验
src/cap-x/.venv/bin/python -m physrsr.active identification redesign \
  --experiment example_identification --round 2 --review public_review.json
src/cap-x/.venv/bin/python -m physrsr.active identification probe \
  --experiment example_identification --round 2
src/cap-x/.venv/bin/python -m physrsr.active identification synthesize \
  --experiment example_identification --round 2
src/cap-x/.venv/bin/python -m physrsr.active identification evaluate \
  --experiment example_identification --round 2

# 参数已足够，使用任务反馈修订共享控制策略，不增加 Probe
src/cap-x/.venv/bin/python -m physrsr.active identification revise-task \
  --experiment example_identification --previous-round 2 --round 3
src/cap-x/.venv/bin/python -m physrsr.active identification evaluate \
  --experiment example_identification --round 3
```

自动 v2 链路使用 `serve-agent`、`serve-physic`、`serve-simulator`，以及 `serve-measured-world` 或 `serve-measured-target`。
World 的物理参数与评估种子来自私有配置；策略请求只携带同一份源码。
旧 `serve-target`、`serve-sim-target`、`serve-world` 与 `serve-legacy-simulator` 保留为历史 v1 入口。
`execute` 显式执行冻结的最终 policy。具体合同、前置服务和参数见 [架构与 API 说明](docs/architecture.md)。

## Prompt 和产物

- [初版 CaP prompt](prompts/initial_cap.txt)：任务、初始公开观测和机器人合同。
- [自动决策 prompt](prompts/loop_decision.txt)：依据预测与实测证据决定辨识、策略修订、验证或停止。
- [逐实例调度 prompt](prompts/scoped_decision.txt)：显式选择源码修复、交互、续执行或 policy 修订；新版两条自动链路使用此合同。
- [旧轨迹整理 Probe prompt](prompts/physic_probe.txt)：匿名执行证据、物理不确定性、候选交互、成本与 Probe 源码。
- [Probe 后 CaP 修订 prompt](prompts/revise_from_probe.txt)：任务、初版轨迹和实际 Probe 观测，一次生成共享修订版本。
- [通用物理辨识 prompt](prompts/physical_identification.txt)：因素选择、可辨识性、多设置对照、测量模型、数值估计与本体约束。
- [辨识后策略 prompt](prompts/identified_policy.txt)：通过 API 获取当前实例的估计记忆，同一源码选择动作与接触位置。
- Probe-first 实验从 `task_context/`、`active_probe_1/` 开始，随后是任务候选与修订。历史对照可保留 `initial_cap/`。
- 本目录存放实现与使用说明。新实验过程、结果与日志统一记录在 `/data/wjt/PhyAgent-RSR/output/`；项目 `docs/` 保留历史索引。

## 验证

```bash
MUJOCO_GL=egl src/cap-x/.venv/bin/python -m pytest src/cap-x/tests/active -q
```

测试使用 API fixtures、隔离 guest 和惯性参数编译检查，不调用 Coding 模型，不执行物理任务或真机动作。

v2 的腕部力/力矩来自物理传感器读数，含工具自重，需同姿态去皮。
接触力矩容量随接触几何和摩擦变化；未标定时显式为未知。
真机使用 `MeasuredRobot(control, perception, proprioception, setup, capabilities)`，
其感知、状态与 `robot.hold` 必须由真实服务提供，缺失时报告错误。
v1 历史策略仍保留原接口与 RPC 预算。

v2 在 `motion_contract` 中公布定位和姿态容差，并用有限次实测反馈确认到位。
同一精度配置通过 `FrankaControlPrivilegedApi.configure_motion` 传入仿真与真机关节控制器。
传感器分辨率、控制精度和任务目标区域分别建模。已知运动未到位可以返回生成程序作有界恢复；
传感或通信基础设施异常保持独立错误。

外部任务服务可在每个 case 中提供 `endpoint`，实现同一套公开目标 API 和独立 World 评分，
无需改动 Physic Agent。缺省私有配置用于本仓库已有的 Rod Lift World 适配器。
真机通过 `InstrumentedTarget(MeasuredRobot(...))` 提供 `target.execute_probe`、
`target.update_identification` 与 `target.execute_policy`；后者执行同一份冻结源码，
程序调用完成与外部任务评分分别记录。
