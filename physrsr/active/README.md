# PhysRSR Active Identification

一个 World、一套机器人、一份整场景 CaP，通过主动辨识更新预测模型，再评估、修订和独立验证。
多个物体以 `object_id` 区分物理估计，共用场景与机器人；不会创建多个自动闭环。

## 目录职责

```text
active/
  workflows/loop.py           # 唯一自动工作流 ActiveLoop
  tools/stages.py             # 单次生成/执行诊断，不管理实验轮次或任务停止
  agents/                    # Physic / Coding 合同及公共推理校验
  core/                      # 数据、源码、证据、物理记忆和确定性决策
  runtime/                   # 机器人执行、测量与预测 Simulator
  tasks/                     # 任务物理绑定、传感和整任务评分
  api/                       # HTTP 服务、进程与调用日志
  prompts/                   # 任务决策、Probe、模型修订与 CaP prompt
  examples/                  # 单杆及三杆任务配置
```

`workflows/` 只包含 `loop.py` 和包标记，不存放固定实验、报告生成、旧轨迹重放或手动阶段管理。
原 task-only、多 World 固定对照及共享修订工具已退出当前代码包；历史数据与冻结源码保留在输出目录。

## 自动运行

配置提供 `agent / physic / simulator / target` 四个端点。首次 Probe 前不生成 CaP，也不运行未辨识模型预测。
有实测证据后，主循环重放拟合数据、提交模型、预测任务候选，并决定继续辨识、修订或独立验证。
稳健候选、当前模型的独立验证和低剩余探索价值共同满足时才冻结 `final/`。
预算耗尽、程序自报成功和单次执行完成均不等同于任务充分。

```bash
src/cap-x/.venv/bin/python -m physrsr --help
src/cap-x/.venv/bin/python -m physrsr validate-config \
  physrsr/active/examples/three_rod_lift_return.json

# 先按场景文档启动服务，再显式开始主循环
src/cap-x/.venv/bin/python -m physrsr run \
  --config physrsr/active/examples/three_rod_lift_return.json --out three_rod_run

# 显式执行主循环冻结的最终程序
src/cap-x/.venv/bin/python -m physrsr execute \
  --bundle /data/wjt/PhyAgent-RSR/output/three_rod_run/final \
  --target-url http://127.0.0.1:8123
```

三杆任务在一个 MuJoCo 环境中构造全部长条，按逐根抓取、抬升、保持和放回评分。
World 私有物理配置不传给 Agent 或预测服务，预测服务使用模型假设与公开场景几何。
单杆 `rod_lift` 基线保留。真机通过 `InstrumentedTarget(MeasuredRobot(...))` 接入同一机器人 API。
感知、腕部传感、保持和设置成本必须由真实服务提供，不用模拟信号替代。

## 显式诊断

`debug` 提供 `design-probe / repair-probe / synthesize-policy / execute-probe / execute-policy`。
每次只执行指定操作，生成源码不会自动触碰机器人。它没有多实例、轮次调度、模型拟合或最终冻结逻辑。
程序和观测记录在新的输出目录；源码哈希、对象归属和安全断点仍按当前合同校验。

接口及示例见 [单次诊断工具](docs/debug.md)。旧 `identification`、`author-initial`、`smoke`、
`probe-world / revise-cap / compare-caps` 命令不再提供。

## 合同与产物

- [架构与服务启动](docs/architecture.md)
- [Agent 职责和 RPC](docs/agent_contracts.md)
- [测量与机器人 API](docs/identification.md)
- [探索、预算和断点](docs/exploration.md)
- [单 World 多对象与三杆物理任务](docs/single_world.md)

新实验产物使用 `physrsr.paths.output_path()`，统一位于 `/data/wjt/PhyAgent-RSR/output/<实验名>/`。
`--out` 接受实验名或该根目录内的绝对路径，不依赖工作目录。输入文件使用实际路径。
主循环按 `task_context/`、`active_probe_n/`、`candidate_cap_1/`、`revised_cap_n/`、`models/` 保存过程。
Schema、prompt、生成源码、API 请求响应、实测轨迹、验证和停止原因分别归档。

## 验证

```bash
MUJOCO_GL=egl src/cap-x/.venv/bin/python -m pytest tests/active -q
```

测试覆盖合同、隔离执行器、主循环、单次诊断和三杆真实编译/接触/步进，不调用真实 LLM、IK 服务或真机。
