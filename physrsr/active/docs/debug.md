# 单次诊断工具

`run` 是唯一自动工作流。`debug` 每次只生成或执行一个冻结程序，复用当前 Agent 和 Target API。
实现位于 `tools/stages.py`，不做多实例管理、轮次调度、模型拟合、候选评估或停止认证。

| 操作 | 所需端点 | 输入 | 行为 |
| --- | --- | --- | --- |
| `design-probe` | `physic` | `--context` | 调用 `physic.design_probe`，只生成 |
| `repair-probe` | `physic` | `--context --bundle` | 同一 Probe 合同，保留实验 ID，只修源码 |
| `synthesize-policy` | `agent` | `--context`，可选旧 `--bundle` | 调用 `agent.synthesize_policy` 生成或修订 |
| `execute-probe` | `target` | `--bundle`，可选 `--belief --resume` | 显式执行一次 Probe，保存并更新实测记忆 |
| `execute-policy` | `target` | `--bundle`，可选 `--belief` | 显式执行一次候选，保留外部评分 |

配置仍使用主循环的任务格式，但只要求该操作所需端点，不要求 Simulator。
`--context` 必须是公开证据上下文，可参考主循环已归档的 `rounds/.../context.json` 与 [Agent 合同](agent_contracts.md)。
不要将 World 私有物理配置放进 Agent 上下文。目标语义取自 `--config`，机器人程序合同使用当前 v2 API。
设计阶段不调用 Target，不准备或执行机器人。

```bash
# 只编写一个辨识程序
python -m physrsr debug design-probe \
  --config physrsr/active/examples/three_rod_lift_return.json \
  --context /path/to/public_context.json --out diagnostics/probe_1

# 显式执行；不修改已冻结源码
python -m physrsr debug execute-probe \
  --config physrsr/active/examples/three_rod_lift_return.json \
  --bundle /data/wjt/PhyAgent-RSR/output/diagnostics/probe_1 \
  --out diagnostics/probe_1_execution

# 仅当上次结果有已确认安全断点时继续同一源码
python -m physrsr debug execute-probe \
  --config physrsr/active/examples/three_rod_lift_return.json \
  --bundle /data/wjt/PhyAgent-RSR/output/diagnostics/probe_1 \
  --belief /data/wjt/PhyAgent-RSR/output/diagnostics/probe_1_execution/belief.json \
  --resume --out diagnostics/probe_1_continuation

# 只修复源码；不会自动执行新版本
python -m physrsr debug repair-probe \
  --config physrsr/active/examples/three_rod_lift_return.json \
  --context /path/to/public_review_context.json \
  --bundle /data/wjt/PhyAgent-RSR/output/diagnostics/probe_1 \
  --out diagnostics/probe_2
```

生成目录包含 `context.json / schema.json / prompt.txt / response.json / analysis.json / program.py / manifest.json`。
只有 `debug` 生成的同类型 v2 bundle 可被这些执行命令读取，源码必须匹配冻结哈希。
修复后须使用新的 bundle；新源码不会继承旧源码的断点。
执行目录保存 `api.jsonl / request.json / result.json / summary.json`，Probe 还保存 `belief.json`。
所有输出目录必须是新的 `output_path()` 子目录，原始记录不覆盖。

未提供 `--belief` 时读取 Target 当前的已学记忆；提供时使用该测量文件。
场景对象 ID 必须匹配，每条估计必须有对象归属。恢复失败不删除此前获得的有效估计。
暂停程序必须显式加 `--resume` 才可继续，且断点需与源码和记忆范围一致。
程序执行使用 Target 的真实准备与 RPC 限制；诊断工具不提供主 Loop 的累计预算和收益判断。

`debug_only=true` 的产物不认证任务充分，也不会生成 `final/`。
`status=completed` 表示程序运行完成；若 Target 提供任务评分，另见 `task_result.success`。
主 Loop 的模型更新、独立验证及停止合同仍须通过 `run` 完成。
旧固定实验入口及其原产物格式不在这里做隐式兼容转换，已有归档保留原文。
