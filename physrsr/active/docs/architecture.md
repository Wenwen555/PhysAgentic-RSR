# 自动主动物理辨识闭环

`python -m physrsr run --config ... --out <实验名>` 使用 `workflows/loop.py:ActiveLoop` 自动调度。
默认合同是 `physrsr.robot.v2`。从任务和视觉输入直接分析物理不确定性、编写 Probe。
首轮不生成 CaP，不需要已知物理模型，也不以未辨识的 Simulator 预测指导首轮动作。
`workflows/` 只保留 `loop.py`；单次显式诊断位于 `tools/stages.py`，使用 `debug` 命令，不管理实验轮次或停止判断。

## 实际代码链

```text
任务、视觉 API 的姿态/几何输出 → physic.assess_task → physic.design_probe
  → target.execute_probe → 同一 World 内按 object_id 保留实测估计
  → 整场景模型 / 一份完整任务候选
  → physic.assess_task
      ├─ revise_policy → agent.synthesize_policy → sim.evaluate
      ├─ repair_probe → physic.design_probe → 仅冻结修复源码
      ├─ execute_probe → 显式执行已冻结源码
      ├─ resume_probe → 同源码、同 World 的已确认断点 → 继续试验
      ├─ probe → physic.design_probe → sim.predict_probe
      │          → 收益/风险/成本检查 → target.execute_probe
      │          → 估计与实测观测 → physic.revise_physical_model
      │          → 重放全部拟合交互 → sim.update_model
      │          → target.update_identification → sim.evaluate
      │          → 共享 policy 修订 → sim.evaluate
      ├─ validate → 新动作 Probe → 预测与实际观测比较
      └─ stop → 宿主核对稳健性、当前版本验证与剩余收益 → 冻结
  ← 根据更新后的证据进入下一轮 physic.assess_task
```

`run` 始终进入同一个 `ActiveLoop`，只连接一个 World 和一个预测模型会话。
三个长条属于该 World 的三个对象，不是三个独立实例；观测、记忆及预测接口见 [场景合同](single_world.md)。
`sim.configure` 只初始化模型会话；已有真实辨识证据后才创建模型并预测。
`run --cases` 会在读取文件或调用服务前报错，不再隐式切换算法。
旧 v1、多 World 自动闭环及其 CLI 已删除，不再保留导入兼容壳；历史实验产物不改写。
Physic Agent 接收公共语义目标、几何、预测失败、真实交互和本体信息，不接收 benchmark ID 或私有 World 参数。
第一个任务候选根据实测辨识结果生成。任务名称不用于调度器或 Physic prompt 分支。

Agent 接口、阶段 Schema 与移除入口见 [Agent 合同](agent_contracts.md)。

## 代码职责与 API

| 组件 | 代码与调用 |
| --- | --- |
| 自动调度、预算、停止 | `workflows/loop.py`，目录中唯一工作流 |
| 阶段持久化和事件 | `core/recording.py:SessionRecorder`，可注入 `Recorder` 实现 |
| 记录 API 调用及校验版本 | `api/client.py:RecordedCalls`，不自动重发请求 |
| 单次生成/执行诊断 | `tools/stages.py`，见 [debug](debug.md) |
| 逐实例证据、交互决策理由、断点 | `core/exploration.py`，细节见 [实验调度合同](exploration.md) |
| 物理分析、Probe、模型修订 | `agents/physic.py` → `physic.assess_task / design_probe / revise_physical_model` |
| policy 生成/修订 | `agents/coding.py` → `agent.synthesize_policy` |
| 公共推理及 Schema 校验 | `agents/base.py` |
| 类型合同与 Schema 生成 | `core/agent_models.py`，Pydantic v2 |
| 源码及证据合同 | `core/programs.py`、`core/evidence.py` |
| 模型会话和任务路由 | `runtime/predictive.py` → `sim.configure / predict_probe / evaluate / update_model` |
| 任务接口与选择 | `tasks/base.py`、`tasks/registry.py` |
| Rod Lift 专用规则和参数 | `tasks/rod_lift.py` |
| Rod Lift 仿真传感与 World | `tasks/rod_lift_world.py` |
| 仿真与真机机器人 API | `runtime/instrumented.py` → `target.prepare / describe / observe / execute_probe / update_identification` |
| 通用观测投影 | `core/measurements.py` |
| 同一源码执行与冻结 | `core/policy.py` |

`sim.update_model` 提交实际物理参数、权重和程序可读辨识记忆，不只是修改 `belief.json`。
后续 Simulator 实例化使用新参数，程序通过同一个 `robot.describe()['identification']` 读取当前 World 的对象记忆。
所有预测带 `session_id + model_version`，旧版本请求会被拒绝。
版本不一致时记录会话、调用方法、预期/实际版本，并抛出 `ModelVersionError`；不自动重同步或继续认证。
主循环通过 `handle_stop / handle_policy / handle_probe` 分派动作；各处理函数仍执行相同的证据和预算检查。
参数 schema 来自任务适配器声明的名称、单位、坐标系与范围，不固定为 COM 或摩擦。
提交前，宿主用原 Probe 源码、交互发生时的记忆和全部拟合观测检查新模型；解释不了就不提交。
验证数据不进入模型拟合。验证失败要求新的拟合交互，模型更新后旧验证记录失效。

`measured_response_v2` 包含物体三维位移、转角、物体/TCP 相对平移变化、腕部力和力矩变化、最大升高、载荷峰值与六个有符号的腕部变化分量。
完整观测仍保存。相对平移不是滑移真值，原始腕部载荷不是已去皮物体载荷；数值辨识由生成的 Probe 按测量方程执行。
十六个 `loop.feature_scales` 是工程误差尺度，不声称已校准的统计置信区间。

## 显式启动

导入或查看帮助不会执行实验。以下服务分别启动；物理 Simulator 还需要已有依赖和 IK 服务。

```bash
src/cap-x/.venv/bin/python -m physrsr serve-simulator
src/cap-x/.venv/bin/python -m physrsr serve-agent --out services/coding_run_new
src/cap-x/.venv/bin/python -m physrsr serve-physic --out services/physic_run_new
src/cap-x/.venv/bin/python -m physrsr serve-measured-world --private-config /path/to/private_world.json

src/cap-x/.venv/bin/python -m physrsr run \
  --config physrsr/active/examples/rod_lift.json --out active_loop_new

# run 冻结后，用 execute 在同一个 World 显式执行完整任务。
# execute 也可读取已有历史冻结产物；这不恢复旧调度器。
src/cap-x/.venv/bin/python -m physrsr execute \
  --bundle /data/wjt/PhyAgent-RSR/output/active_loop_new/final \
  --target-url http://127.0.0.1:8123
```

私有 World 文件只由目标服务读取，不放进公开配置或 Agent 上下文。
真机使用 `serve-measured-target --factory lab.integration:create_robot`，可信工厂返回
`MeasuredRobot(control, perception_api, proprioception_api, setup_api, capabilities)`。
感知、本体、保持和复位由真实 API 提供，不以模拟信号或主机 sleep 替代。
真实 setup API 应报告 `setup_cost_s`；缺少成本时保存已发生观测并以 `cost_unknown` 停止，不以零成本交付。

## 接入其他任务

`PredictiveSimulator` 只接收外部注入的 `TaskAdapter` 列表，不默认导入或选择某个任务。
内置任务名和默认选择集中在 `tasks/registry.py`，CLI 组合根负责创建并注入适配器。
每个任务的物理绑定、成功判据、传感映射和评分放在 `tasks/`，不放进通用预测/机器人服务。

```bash
src/cap-x/.venv/bin/python -m physrsr serve-simulator \
  --adapter rod_lift --adapter lab.pushing:create_adapter
```

每个可信适配器提供 `task_id`、`mechanisms`、`parameter_contracts`、`validate(task, model)`，以及
`execute(task, model, policy, seed, belief, purpose)`。
任务预测返回 `success/risk/margin/duration_s/observations`；Probe 返回 v2 运行观测与风险。
任务几何、物理绑定和原始成功谓词由适配器维护，机器人合同与真机保持一致。
新任务不需要修改调度器、Physic prompt 或模型参数 schema。
内置物理环境包括单杆 `rod_lift` 和三杆共存的 `three_rod_lift_return`；三杆任务在一个环境中记录各杆的抓持、保持与放回状态，整体评分由 `tasks/rod_scene_env.py` 实现。
其他任务仍须提供实际 Simulator/目标适配器。

预测服务按模型、程序和 seed 顺序执行并汇总；真实 World 操作和模型提交也保持串行。
每个三杆假设在完整场景中评估，不将共存对象拆成不同 World。

## 停止、版本与产物

候选需满足合理模型集合内的成功、风险和裕度要求，当前模型版本有足够的新动作预测证据，且继续探索净收益低，才生成 `final/`。
Physic Agent 返回 `stop` 本身不足以交付；预算耗尽、恢复失败、预测不符和模型不足均分别记录。
有限候选的计算决策价值与 Scientist 对新程序结构的收益估计分别存档，不冒充精确的信息价值。
AST 检查排除同源码/注释改名；它不能形式化证明动作的动力学独立性。

一个实验从 `task_context/`、`active_probe_1/` 开始，随后生成任务候选、修订与模型记录，
以及 API 请求响应 `events.jsonl`、进度、停止原因和最终冻结源码。
Probe 的分析、prompt、源码、预测、执行观测与 belief 保存在同一个 `active_probe_n/` 内。
所有产物位于 `/data/wjt/PhyAgent-RSR/output/`；有效历史归档也已迁入，冻结文件保留原文。

旧 v1 状态机已删除；`serve-legacy-simulator` 仅保留旧 API 服务能力，供显式单阶段工具使用。
历史固定实验工具已从当前包删除，已有冻结对照保留。模型集合停止规则不是实际环境成功率的置信保证。
