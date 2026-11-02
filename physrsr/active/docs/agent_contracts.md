# Agent 与 Workflow 合同

当前主链路只有一套 Physic Agent 合同和一套 CaP 合成合同。
阶段职责不同，分别使用任务决策、Probe 设计、物理模型修订和 CaP 合成 Schema；没有第二套 Probe 设计输出。

| 模块 | 职责 |
| --- | --- |
| `agents/physic.py` | 任务物理不确定性评估、Probe 设计/源码修复、物理模型假设修订 |
| `agents/coding.py` | 根据辨识和预测反馈生成/修订可执行 CaP |
| `agents/base.py` | JSON Schema 校验、共同推理后端、最多三次结构/源码审查 |
| `core/programs.py` | v2 源码合同、Probe 断点接口要求、排除无动作 CaP |
| `core/evidence.py` | 已有 Evidence ID 的提取、引用检查和观测摘要 |
| `core/exploration.py` | 决策范围、证据需求和交互理由的确定性检查 |
| `workflows/loop.py` | 选择和执行已请求阶段、预算、物理记忆、Simulator 提交、候选评估及最终停止认证 |
| `tools/stages.py` | 单 World 的单次生成/执行诊断，复用相同 Agent RPC、Schema 和 prompt |
| `physrsr/coding/agent.py` | 底层 Codex CLI 推理传输与隔离，不选择物理因素或阶段 |

Agent 不调用机器人或 Simulator，不执行它刚生成的源码，不提交 belief，不自行认证停止。
Workflow 不再维护一套物理实验 Schema 或另一种 CaP 合成实现。
Agent 模块不导入 Workflow，Physic/Coding 也不相互导入各自的领域实现。

## 主接口

| RPC | Python 实现 | Schema / prompt |
| --- | --- | --- |
| `physic.assess_task` | `PhysicAgent.assess_task` | `DECISION_SCHEMA` / `world_decision.txt` |
| `physic.design_probe` | `PhysicAgent.design_probe` | **唯一** `DESIGN_SCHEMA` / `physical_identification.txt` |
| `physic.revise_physical_model` | `PhysicAgent.revise_physical_model` | 由 Simulator `parameter_contracts` 构造 / `physical_model.txt` |
| `agent.synthesize_policy` | `CodingService.synthesize_policy` | `SYNTHESIS_SCHEMA` / `identified_policy.txt` |

`PhysicService` 仅是 `PhysicAgent` 的同类名别名，供 HTTP 服务装配使用，不提供第二套逻辑。
`physic.assess_task` 同时处理首次评估、后续任务不确定性重评估和 `stop` 请求；宿主仍独立检查停止条件。
它要求一个 `current` World 的预测决策合同，不再根据输入字段选择旧决策 Schema/prompt。

新 Probe、设计修订和源码修复都调用 `physic.design_probe`：

- `purpose=fit`：辨识实验，要求读取同源码 `resume_state` 并提供安全断点路径。
- `purpose=validation`：独立验证设计，Schema 不变，拟合/验证来源分离由主 Loop 维护。
- `authoring_mode=repair_probe` 加 `previous_probe_source/design_review`：只修源码，不自动执行。

`debug design-probe / repair-probe` 复用这一接口。没有单独的 `revise_probe` 协议。
候选设计、测量方程、可辨识边界、本体约束、解释和恢复都来自同一 `DESIGN_SCHEMA`。

## 校验与证据

所有推理响应先按实际输出 Schema 做本地校验，包括必填字段、类型、边界和额外字段；再做语义校验。
Probe 的观测事实只能引用上下文已提供的 Evidence ID，所选因素必须在物理分析中声明。
主 Loop 和单次诊断工具同样校验冻结的 Probe 响应；`schema.json`、prompt、分析与源码在对应阶段归档。

证据 ID 只从已给出的初始、实测、预测记录及宿主列出的可用证据中取得。
审查文本或生成响应里自行写出的 ID 不能成为新证据。引用空、重复或不存在的 ID 会被拒绝。
模型修订只接收拟合上下文，引用必须恰好覆盖给定拟合交互；独立验证报告不会被传入其 prompt。
新模型仍须通过主 Loop 的拟合轨迹回放后才能提交。

失败响应只触发最多三次结构/源码修复。修复请求明确标明未执行、没有新测量；耗尽次数时报错，不替换模板程序。
结构校验不证明物理方程正确；实际可辨识性、动作风险、预测能力和任务成功仍需独立执行与验证。

## 移除与保留的入口

旧 `PHYSIC_SCHEMA`、`physic_probe.txt`、`agents/identification.py`、`workflows/probe_authoring.py` 和 `author-probe` 命令已移除。
`physic.propose_probe / revise_probe / design_identification / revise_predictive_model` 不再接受调用。
旧 Coding 的 `agent.propose / agent.revise_model` 混合物理接口已移除。
主 CaP 方法由 `agent.synthesize_identified_policy` 改为 `agent.synthesize_policy`。
调用方需改用上表接口，不会静默回退到旧协议。

旧 task-only、固定多 World 对照、轨迹重放和单次共享修订工具已从当前包移除，其专用 Coding Schema/prompt 也已删除。
Coding Service 仅提供 `agent.synthesize_policy`；旧 `agent.generate_initial_cap / agent.revise_cap_from_probe` 请求会被拒绝。
显式诊断改用 [debug](debug.md)，不管理实验轮次、实例选择、模型更新或任务认证。
已有实验及冻结源码不改写，历史合同只保留在实验归档中。
