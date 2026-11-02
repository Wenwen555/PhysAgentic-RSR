# 按实例调度实验与断点续执行

通用调度入口是 `workflows/scoped_loop.py`；`workflows/batch.py` 保留公开入口与历史答案恢复工具。
单实例预测闭环 `workflows/loop.py` 使用相同的决策、源码与机器人进度合同。
合同校验位于 `core/exploration.py`，Physic Agent 的 prompt 是 `prompts/scoped_decision.txt`。

## 决策和执行分别记录

`physic.assess_task` 收到 `decision_contract=scoped_exploration_v1` 时逐实例返回证据需求：
`quantity_or_condition`、`operation_decision`、`met`、`evidence_ids`，以及 `sufficient` 和理由。
宿主检查每个实例恰好出现一次、引用属于当前实例、已满足的需求有证据，且 `sufficient` 与需求一致。
因素名称、精度、实验设置和估计方程由 Agent 选择；宿主不预设质心、摩擦或任务名称。
该检查验证来源与结构，不声称形式化证明物理判断或信息价值。

| action | 源码生成 | 新物理交互 | 含义 |
| --- | --- | --- | --- |
| `repair_probe` | 修订已有源码，保留实验 ID | 无 | 初始化、实现、恢复或协议缺陷；产物保持未执行状态 |
| `probe` | 设计新实验 | 仅指定实例 | 新的关键物理歧义，需要不同操作决策之间的辨别 |
| `execute_probe` | 无 | 仅指定实例 | 显式执行已冻结源码，包括刚修复但尚未执行的版本 |
| `resume_probe` | 无，哈希相同 | 仅指定实例 | 从已确认的同实例断点继续同一实验程序 |
| `revise_policy` | 合成或修订共享 CaP | 无新 Probe | 对证据足够的实例评估任务候选，其余实例继续辨识 |
| `stop` | 无 | 无 | 停止探索或报告不足，不自行认证任务成功 |

单实例预测模式另外支持 `validate`，要求稳健候选与新的独立预测证据；验证数据不拟合模型。
验证暂停的断点与子段单独保存，不混入辨识记忆或拟合证据；整个程序完成且所有子段均通过，
才记一次有效独立验证。模型版本变化后，旧验证断点不能作为新版本的验证继续使用。
修复源码不会触发 `target.prepare` 或 `target.execute_probe`，也不新增测量证据或消耗 Probe 预算。
修复后如需验证，必须另有 `execute_probe` 决策。新源码哈希不自动继承旧源码的断点。

每个物理交互必须说明：未解决的歧义、已有证据为何不能决定、至少两个可观察结果，
以及对应的不同操作决策。还必须给出正的预计决策收益与逐实例成本（含准备、恢复），总成本为其和。
已对当前决策足够的实例不会因为其他实例尚未完成而被一起重跑。

共享源码与逐实例记忆分别维护。证据充分的部分实例可以先验证同一份候选；
后续共享 policy 修订仍回归评估所有当前证据足够的实例。
只有同一份冻结源码覆盖所有要求实例且原 World 判定全部成功，批量入口才生成 `final_cap/`。
缺失的评估记录不视为成功。单实例 `final/` 继续要求模型稳健性和独立预测证据。

## 多设置实验和安全断点

`run(robot)` 通过公开 API 获取当前执行信息：

```python
execution = robot.describe()['probe_execution']
state = execution['resume_state'] or {'attempted': [], 'next_setting': None}
remaining = execution['remaining_rpc_calls']
```

其中 `program_sha256` 绑定冻结源码，断点还绑定宿主维护的匿名实例范围；
`max_rpc_calls`、`remaining_rpc_calls` 是该次执行的真实 RPC 上限与余量。
上限不超过 192，可以由机器人能力合同声明更小的值。
生成程序在每个设置前核算开销，预留恢复、观测和保存断点的调用。
同一个程序可以在一次执行内试多个设置，也可以在预算不足时主动结束并续执行。

程序应先恢复至稳定支撑、打开夹爪并真实保持/观测，最后：

```python
robot.checkpoint(state)
return {'status': 'budget_exhausted', 'estimates': estimates, 'trials': trials}
```

也可返回 `status='paused'`。断点状态最多 4 KiB，只保存已试设置、搜索区间、下一设置等进度；
完整观测由宿主存档。重新准备 episode 后必须重读实时姿态并重新建立所需校准，不能复用过期 TCP 或去皮信号。

共享执行器检查实测支撑距离、时间分离的物体/TCP 稳定性和夹爪开度。
只有程序正常结束、断点后没有再执行动作，并明确返回暂停状态，才确认 `resumable=true`。
夹持中、恢复失败、程序异常、超限或源码哈希变化均不能直接续执行。
这些是准备与进度合同的检查，不替代独立接触传感或任务评分。

`target.update_identification` 将本实例的 `probe_progress` 交给目标；
下一次仍通过 `target.execute_probe` 执行同一份源码，并由 runtime 在 `probe_execution.resume_state` 提供对应进度。
仿真、预测回放和 `MeasuredRobot` 真机接口共用同一机制，无需新增底层硬件控制动作。

## 预算、存档与命令

`loop.max_probes` 在批量 CLI 中表示所有实例累计的物理 Probe 执行次数，续执行也占一次；
`max_rounds` 限制决策/源码修订轮数，两者不混为同一个计数。
`max_probe_cost`、`probe_cost_weight`、`min_net_value` 同样传给批量状态机。
拟议收益是未校准的操作效用预测，宿主检查正收益、成本与剩余预算；实际成本按执行与准备计入。
缺失成本、未确认结束的执行或实际超支会明确停止，不能当作零成本重试。

首次执行保持 `active_probe_n/world_execution/<instance>/`；
同源码后续执行进入 `world_execution/attempts/002/<instance>/` 等目录，原始结果不覆盖。
`execution_index.json` 记录目标实例、源码哈希、执行次数和结果路径。
`instance_memory/` 保存逐实例派生记忆；`exploration_state.json` 保存批量控制器进度；
`progress.json` 将源码修复、科学实验设计和物理执行分别记录。

下列命令只有显式运行才启动相应阶段：

```bash
# 只修源码，不执行
src/cap-x/.venv/bin/python -m physrsr identification redesign \
  --experiment RUN --round 2 --previous-round 1 --review review.json --repair

# 只执行指定实例；新源码的首次执行不加 --resume
src/cap-x/.venv/bin/python -m physrsr identification probe \
  --experiment RUN --round 2 --instances instance_b

# 保持同一 round/源码，从该实例已确认的断点继续
src/cap-x/.venv/bin/python -m physrsr identification probe \
  --experiment RUN --round 2 --instances instance_b --resume

# 恢复批量状态机；私有 World 列表和任务目标必须一致
src/cap-x/.venv/bin/python -m physrsr run \
  --config config.json --cases private_cases.json --out RUN --resume
```

批量中间阶段可以只有部分实例的执行/评估记录，状态机从各实例历史重建证据。
最终交付仍要求完整评估。这里的续执行是实验搜索进度的延续，
物理起点由真实 `episode.prepare` 服务负责，不声称恢复同一个连续动力学状态。
