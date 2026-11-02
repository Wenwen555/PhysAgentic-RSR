# PhysRSR

任务驱动的主动物理辨识、模型与程序修订模块，统一使用机器人 API，仿真和真机共用策略源码。
源码已从 `src/cap-x/experiments/physics_cap` 迁到仓库根目录的 `physrsr/`，Python 命名空间为 `physrsr`。

## 当前入口

- [主动辨识与共享策略](active/README.md)
- [通用实验设计和测量 API](active/docs/identification.md)
- [Coding Agent](coding/README.md)
- `worker.py`、`physics_spec.py`、`phase_tracker.py`：共用仿真、物理配置和执行观测。
- `m21r/`、`m21s/`：历史协议相关代码，部分依赖归档源码；复现说明见下方历史记录。

从 `/data/wjt/PhyAgent-RSR-v1` 运行，复用 CaP-X 环境：

```bash
uv pip install --python src/cap-x/.venv/bin/python 'jsonschema>=4.18,<5'
uv pip install --python src/cap-x/.venv/bin/python --no-deps -e .
src/cap-x/.venv/bin/python -m physrsr --help
src/cap-x/.venv/bin/python -m physrsr.active validate-config physrsr/active/examples/rod_lift.json
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m pytest tests/active -q
```

新实验的输出、分析、结果报告和日志统一保存在 `/data/wjt/PhyAgent-RSR/output/<实验名>/`。
命令中的 `--out <实验名>` 和 `--experiment <实验名>` 相对此目录解析，也可提供目录内的绝对路径。
命令会拒绝写到该目录之外；Python 调用使用 `physrsr.paths.output_path()`。
现有有效历史对照也已迁入上述 `output/`。

只有显式选择实验子命令才会执行交互。提示词和配置通过包内路径定位，控制器资源使用 CaP-X 的绝对路径。
有效实验输出及冻结哈希已统一迁入上述 `output/`，当前报告索引见 [项目记录](../docs/2026-11-01-task-driven-identification.md)。
自动 v2 闭环与服务调用链见 [架构说明](active/docs/architecture.md)。

## 历史入口说明

以下保留历史研究阶段说明。旧输出中的命令、源码和快照反映当时布局，不批量改写以免破坏其来源与哈希。

### Rod Lift 历史协议

新的任务驱动主动辨识闭环位于 [active/README.md](active/README.md)：Simulator 预测反过来选择 Probe，
持续修订物理模型和同一份 policy，独立新动作预测检查后才停止。仿真／真机共用源码和公开 API；
本次只搭建代码和离线合同测试，未启动实验。下述 M1/M2.1-R/M2.1-S 为历史入口。

当前保留 M1/M2.1-R 冻结模板基线，以及新增 Codex Coding Agent 的辨识→代码生成→仿真修订→World 评估。
M2.1-R 研究结果保持冻结。本次 v2 仅清理代码结构，不新增物理实验或改变选择规则。

新增独立版本 [M2.1-S](m21s/README.md)：共享静态审查、控制步轨迹反馈与独立验证门控；
四组严格对照 A/B/C/D，B/C 复用同一修订候选。既有冻结入口和结果不变。

## 代码入口

| 文件 | 用途 |
|---|---|
| `physics_spec.py` / `candidate_programs.py` | 物理配置与 M1 候选 |
| `worker.py` / `phase_tracker.py` | 共用控制、确定性 reset 与阶段观测 |
| `evaluate.py` / `run_experiment.py` | 共用统计及 M1 实验入口 |
| `m21r/core.py` | 传感响应、CoM/摩擦反演、程序模板与候选选择 |
| `m21r/identification.py` | 校准模型库构建／读取 |
| `m21r/runner.py` | World/Model 独立角色进程池与记录 |
| `m21r/experiment.py` | 核心配对实验入口 |
| `m21r/report.py` | 核心实验统计报告 |

`generate_candidates(task, physics_estimate, uncertainty, control_constraints)` 使用估计 CoM 生成数值抓取点、
速度与可重放 Python。所有组使用相同机器人控制 API；几何范围过滤不是完整碰撞／IK证书。
辨识输入仅包含白名单 pose/TCP 观测；不确定支持不是校准置信区间。该冻结入口为模板合成；新的真实 LLM 路径见下方 Coding Agent。
真机需要对应物体跟踪接口，尚未完成硬件验证。

## 运行

从仓库根目录运行，使用已有 `.venv`。物理 rollout 需要 PyRoKi 服务：

```bash
src/cap-x/.venv/bin/python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand
```

仅在明确需要新实验时使用新的输出目录；以下命令不会覆盖冻结结果：

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m physrsr.m21r.identification --out physics_cap_new_library
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m physrsr.m21r.experiment --library /data/wjt/PhyAgent-RSR/output/physics_cap_new_library --out physics_cap_new_run --deltas -0.01 -0.005 0 0.005 0.01 --test-seeds 8
```

校准库构建仅保留模型校准；历史四配置开发验证及一次性审计已退役。主实验输出配置、源码、
传感观测、生成程序、决策、完整 rollouts 和报告；基础设施 error 写入后中止。

## 保留的回归检查

```bash
MUJOCO_GL=egl src/cap-x/.venv/bin/python -m pytest src/cap-x/tests/test_physics_cap_coding.py src/cap-x/tests/test_physics_cap_metrics.py src/cap-x/tests/test_robosuite_rod_lift.py src/cap-x/tests/test_robosuite_physics_lift.py -q
```

保留执行隔离、传感边界、配对协议、错误中止、统计及物理回归；历史模板／阶段／audit 专用测试归档后，不再进入常规收集。

## 结果与归档

- [M2.1-R 研究结果](../docs/2026-11-01-m21r.md)
- [失败归因审计](../docs/2026-11-01-m21r-freeze.md)
- 冻结数据：`outputs/physics_cap_m21r_holdout/`；审计：`outputs/physics_cap_m21r_failure_audit_final/`。
- v1 完整源码快照：`outputs/physics_cap_m21r_freeze_v1/source.tar.gz`。
- 本次移出主目录的17个文件及旧文档：`outputs/physics_cap_m21r_cleanup_v2/legacy_sources.tar.gz`。
- 模板版本的冻结说明：[m21r/FREEZE.md](m21r/FREEZE.md)。

若需复现旧 audit／敏感性／预算分析，请在独立仓库副本中恢复 v1 源码快照，使用其 README 和依赖记录。
旧模块不再保留空壳兼容入口。不要把旧源码覆盖回当前工作树。


## LLM Coding Agent 扩展

新增 [coding/README.md](coding/README.md)：通过本机 Codex CLI 调用 `gpt-6-astra`、low effort，
运行 Vanilla、Physics-informed、Sim-CaP、Oracle Sim-CaP 四组。后两种普通辨识方法共享
完全相同的初始代码，静态检查与仿真修订使用相同推理调用预算；最终统一在独立 World 评分。
生成的任意 Python 在 OS 命名空间中隔离，只经受限机器人 RPC 操作，不复用不安全的进程内 exec。
此扩展不修改 M2.1-R 的模板冻结结果。
