# 自动主动物理辨识闭环

`python -m physrsr run --config ... --out <实验名>` 使用 `workflows/loop.py:ActiveLoop` 自动调度。
默认合同是 `physrsr.robot.v2`。从任务和视觉输入直接分析物理不确定性、编写 Probe。
首轮不生成 CaP，不需要已知物理模型，也不以未辨识的 Simulator 预测指导首轮动作。
`identification design/probe/synthesize/evaluate/redesign/revise-task` 保留为分阶段检查和重放入口；自动运行不需要手动选阶段。

## 实际代码链

```text
任务、视觉 API 的姿态/几何输出 → physic.assess_task → physic.design_identification
  → target.execute_probe → 每个实例独立保留的实测估计
  → grounded model / first shared task candidate
  → physic.assess_task
      ├─ revise_policy → agent.synthesize_identified_policy → sim.evaluate
      ├─ repair_probe → physic.design_identification → 仅冻结修复源码
      ├─ execute_probe → 显式执行已冻结源码
      ├─ resume_probe → 同源码、同实例的已确认断点 → 继续试验
      ├─ probe → physic.design_identification → sim.predict_probe
      │          → 收益/风险/成本检查 → target.execute_probe
      │          → 估计与实测观测 → physic.revise_predictive_model
      │          → 重放全部拟合交互 → sim.update_model
      │          → target.update_identification → sim.evaluate
      │          → 共享 policy 修订 → sim.evaluate
      ├─ validate → 新动作 Probe → 预测与实际观测比较
      └─ stop → 宿主核对稳健性、当前版本验证与剩余收益 → 冻结
  ← 根据更新后的证据进入下一轮 physic.assess_task
```

`sim.configure` 只初始化模型会话；单实例模式在已有真实辨识证据后才创建模型并预测。
`--cases` 模式复用共享 Probe、实例记忆、共享 CaP 合成和 World 评估链，依据实测失败自动修订。
此分支不调用预测 Simulator 的模型更新与独立验证；`final_cap/` 表示全部配置的功能验收，区别于单实例 `final/` 的模型停止合同。
Physic Agent 接收公共语义目标、几何、预测失败、真实交互和本体信息，不接收 benchmark ID 或私有 World 参数。
第一个任务候选根据实测辨识结果生成。任务名称不用于调度器或 Physic prompt 分支。

## 代码职责与 API

| 组件 | 代码与调用 |
| --- | --- |
| 自动调度、预算、停止 | `workflows/loop.py` |
| 历史 task-only 对照生成 | `workflows/initial_cap.py` → `agent.generate_initial_cap`，不是闭环起点 |
| 多实例共享程序闭环 | `workflows/batch.py` → `workflows/scoped_loop.py`，调用相同 Physic/Coding/目标 API |
| 逐实例证据、交互决策理由、断点 | `core/exploration.py`，细节见 [实验调度合同](exploration.md) |
| 物理分析、Probe、模型修订 | `agents/physic.py` → `physic.assess_task / design_identification / revise_predictive_model` |
| policy 修订 | `agents/coding.py` → `agent.synthesize_identified_policy` |
| 模型会话和任务路由 | `runtime/predictive.py` → `sim.configure / predict_probe / evaluate / update_model` |
| 任务接口与选择 | `tasks/base.py`、`tasks/registry.py` |
| Rod Lift 专用规则和参数 | `tasks/rod_lift.py` |
| Rod Lift 仿真传感与 World | `tasks/rod_lift_world.py` |
| 仿真与真机机器人 API | `runtime/instrumented.py` → `target.prepare / describe / observe / execute_probe / update_identification` |
| 通用观测投影 | `core/measurements.py` |
| 同一源码执行与冻结 | `core/policy.py` |

`sim.update_model` 提交实际物理参数、权重和程序可读辨识记忆，不只是修改 `belief.json`。
后续 Simulator 实例化使用新参数，程序通过同一个 `robot.describe()['identification']` 读取当前实例记忆。
所有预测带 `session_id + model_version`，旧版本请求会被拒绝。
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

# 显式执行多实例 sim2sim 共享策略验收；私有参数仅由 World 服务读取。
src/cap-x/.venv/bin/python -m physrsr run \
  --config physrsr/active/examples/rod_lift.json --cases /path/to/evaluator_only_cases.json \
  --out shared_probe_first_new

# 单实例 run 冻结后，用 execute 显式执行最终任务；--cases 自身会在私有 World 评估候选。
# execute 也支持批量入口验收后生成的 final_cap；切换 URL 即切换仿真/真机。
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
内置物理环境仍为 Rod Lift；其他任务须提供实际 Simulator/目标适配器，接口测试不代表已完成其物理环境。

## 停止、版本与产物

候选需满足合理模型集合内的成功、风险和裕度要求，当前模型版本有足够的新动作预测证据，且继续探索净收益低，才生成 `final/`。
Physic Agent 返回 `stop` 本身不足以交付；预算耗尽、恢复失败、预测不符和模型不足均分别记录。
有限候选的计算决策价值与 Scientist 对新程序结构的收益估计分别存档，不冒充精确的信息价值。
AST 检查排除同源码/注释改名；它不能形式化证明动作的动力学独立性。

一个实验从 `task_context/`、`active_probe_1/` 开始，随后生成任务候选、修订与模型记录，
以及 API 请求响应 `events.jsonl`、进度、停止原因和最终冻结源码。
Probe 的分析、prompt、源码、预测、执行观测与 belief 保存在同一个 `active_probe_n/` 内。
所有产物位于 `/data/wjt/PhyAgent-RSR/output/`；有效历史归档也已迁入，冻结文件保留原文。

旧 v1 状态机保留在 `workflows/legacy_loop.py`；旧预测服务显式使用 `serve-legacy-simulator`。
历史首版 `smoke` 对照继续使用原协议。模型集合停止规则不是实际环境成功率的置信保证。
