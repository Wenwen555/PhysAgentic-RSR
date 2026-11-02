# 单 World 实验决策与断点续执行

唯一自动闭环是 `workflows/loop.py:ActiveLoop`，入口为 `run --config CONFIG --out RUN`。
它维护一个 World、一套机器人、一份整场景策略和一个预测模型会话。
旧 v1 和多 World 自动闭环、批量 CLI 及兼容导入已删除。单次诊断使用 [debug](debug.md)，不管理实验阶段。

决策 prompt 为 `prompts/world_decision.txt`，合同校验在 `core/exploration.py`。
合同保留 `instances / episode_id / target_instance_ids` 字段，但主循环只有 `current` 一个条目，代表整个 World。
物理参数按场景内 `object_id` 区分，见 [场景合同](single_world.md)。

## 决策与执行

`physic.assess_task` 说明当前操作的证据需求：`quantity_or_condition / operation_decision / met / evidence_ids`。
宿主检查引用是否可用、已满足需求是否有证据、`sufficient` 是否与全部需求一致。
因素、精度、设置和估计方程由 Agent 选择，不按任务名称选择固定 Probe。
该检查验证来源与结构，不声称证明物理判断正确。

| action | 源码生成 | 物理交互 | 含义 |
| --- | --- | --- | --- |
| `probe` | 新实验 | 是 | 解决会改变操作决策的关键歧义 |
| `repair_probe` | 修复已有源码 | 无 | 不产生新测量，不自动执行 |
| `execute_probe` | 无 | 是 | 执行已冻结或修复后的源码 |
| `resume_probe` | 无，哈希相同 | 是 | 从同一 World 已确认断点继续 |
| `revise_policy` | 合成或修订整场景 CaP | 无新 Probe | 根据充分证据生成候选，交给 Simulator 评估 |
| `validate` | 独立动作 | 是 | 检查当前模型的独立预测能力，不参与拟合 |
| `stop` | 无 | 无 | 请求停止，仍须宿主核对完整停止合同 |

物理交互必须说明未解决歧义、现有证据为何不足，以及至少两个导致不同操作决策的可观察结果。
预计收益和准备、交互、恢复成本必须为正，并受共享预算限制。
代码错误、协议缺陷和准备失败不自动构成新的物理不确定性。

模型更新先重放拟合观测，再提交物理参数与版本，重新预测已有候选。
只有稳健候选、当前版本足量独立验证、没有待解决预测不符且剩余探索净收益低，才冻结 `final/`。
预算耗尽、`stop` 声明或程序自报成功均不能替代这些要求。
新模型使旧验证失效；验证暂停的子段和断点单独保存，全部完成且通过才计为一次验证。

## 多设置实验与安全断点

生成程序通过公开 API 读取本次冻结源码的进度：

```python
execution = robot.describe()['probe_execution']
state = execution['resume_state'] or {'attempted': [], 'next_setting': None}
remaining = execution['remaining_rpc_calls']
```

RPC 上限不超过 192，可由机器人能力声明更小的上限。程序必须预留恢复、观测和保存进度的调用。
需要暂停时，先恢复所有对象的稳定支撑、打开夹爪并保持/观测，最后：

```python
robot.checkpoint(state)
return {'status': 'paused', 'estimates': estimates, 'trials': trials}
```

断点最多 4 KiB，只保存搜索进度，不能复用旧实时 TCP 或去皮信号。
正常结束、明确暂停、断点后没有新动作且处于稳定支撑/开放夹爪边界，才确认可续执行。
源码哈希、World 范围或验证模型版本不匹配时拒绝续执行。
`resume_probe` 是当前进程内的试验继续；`run` 暂不支持进程重启后的整体 `--resume`。

## 预算与存档

`max_rounds` 限制决策轮数，`max_probes` 限制物理执行次数（含续执行），源码修复不消耗物理执行次数。
`max_probe_cost` 使用实测交互与准备成本；未知成本或超支会明确停止。

`task_context/` 保存公开初始场景及对象合同；`active_probe_n/` 保存 prompt、源码、预测和实测结果。
首次执行进入 `world_execution/result.json`，同源码继续执行进入 `world_execution/attempt_002/` 等目录。
`candidate_cap_1/`、`revised_cap_n/`、`models/` 分别保存任务候选、修订和模型版本。
`events.jsonl` 记录 API 请求响应，`progress.json` 与 `result.json` 保存进度及停止原因。
所有实验产物通过 `physrsr.paths.output_path()` 放入 `/data/wjt/PhyAgent-RSR/output/<实验名>/`。
