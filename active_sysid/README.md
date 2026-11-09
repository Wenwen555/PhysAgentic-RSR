# Active SysID — Task-Driven Active Physical System Identification

> 让 Agent 根据下游 Code-as-Policy (CaP) 的执行需求, 主动设计能够区分物理
> 机制、辨识关键参数的交互实验 (Probe), 并利用机器人本体感受更新可交互
> 仿真, 使仿真能够预测真实任务中的动作结果。

研究问题: *How can an agent actively discover and identify task-critical
physical mechanisms missing from a reconstructed simulator, using minimal
robot interactions to build a simulation that reliably supports downstream
code-based manipulation?*

完整方案见 [docs/research_plan.md](docs/research_plan.md),
实验设计见 [docs/experiment_design.md](docs/experiment_design.md)。

## 目录结构

```
active_sysid/
├── src/active_sysid/
│   ├── types.py             # 共享数据类型 (ProbeSpec / Evidence / Hypothesis ...)
│   ├── pipeline.py          # 主链路闭环 + 停止条件
│   ├── planner/
│   │   ├── hypothesis.py    # 物理机构候选与可信度 (H1..H4)
│   │   └── probe_planner.py # 信息增益评分选择下一个 Probe
│   ├── probes/
│   │   ├── library.py       # 参数化 Probe 模板库 (读 configs/probes.yaml)
│   │   └── executor.py      # World 交互契约 + 数据记录 + 特征提取
│   ├── identification/
│   │   ├── mechanism.py     # 结构模型选择 (MAP 假设)
│   │   └── cma_es.py        # CMA-ES 连续参数拟合 (分阶段)
│   ├── backends/            # World 后端: base (统一执行模板) + mujoco + mock + 工厂
│   ├── simulator/
│   │   ├── model_editor.py  # MJCF 结构修复 (功能性闩锁优先) + 参数修复
│   │   └── validator.py     # 保留 Probe 重放 + 未见交互预测验证
│   └── cap/
│       └── evaluator.py     # CaP 的 sim 验证 / World 迁移评测
├── configs/
│   ├── probes.yaml          # 9 个 Probe 模板 (3.1 Physical Probe Library)
│   ├── mechanisms.yaml      # 机构库 (3.2 Mechanism Library)
│   ├── pipeline.yaml        # 评分权重 λ / 停止条件 / 特征阈值
│   └── eval_truth.yaml      # World 真值 (仅评测, Planner 禁读)
├── worlds/                  # reconstructed/ (初始重建) + oracle/ (评测真值)
├── scripts/run_probe_loop.py
├── tests/                   # mock 双门 World + 冒烟测试
├── docs/                    # 研究方案与实验设计
└── runs/                    # 实验产物 (gitignore)
```

## 快速开始

```bash
cd active_sysid
pip install -e ".[dev,opt,sim]"     # 至少 pip install -e . (numpy + pyyaml)

# mock 双门 World 上的冒烟闭环 (闩锁辨识 + 解锁阈值估计)
python scripts/run_probe_loop.py --backend mock

# MuJoCo 后端 (object-space 直驱; channel_map 见 configs/mujoco_world.example.yaml)
python scripts/run_probe_loop.py --backend mujoco --mjcf worlds/reconstructed/x.mjcf

# 测试
python -m pytest tests/ -q
```

## 与 ../agentic_rsr 的关系

上游 `../agentic_rsr` 负责视觉重建与场景导出 (Real2Sim 三阶段)。本包
消费其 stage3 导出的 MJCF (放入 `worlds/reconstructed/`), 专注其后的
**主动物理辨识闭环**:

```
无闩锁模型 → 拉门失败 → Agent 提出机构假设 → 选择把手实验
→ 检测解锁条件 → 修复仿真 → CaP 自动改变动作顺序 → World 成功
```

上游已有的 `real2sim-stage3-physics` 是固定轨迹的物理探针; 本包把它
升级为任务驱动、假设引导的主动 Probe 选择。

## 实施优先级 (研究方案第八节)

| 偏差 | 处理模块 | 优先级 |
| - | - | - |
| 全局坐标旋转错位 | SE(3) Registration (Probe 之前完成) | P0 |
| Panda 与 FR3 不一致 | 机器人/控制器统一 + 末端动作接口 | P0 |
| 闩锁结构缺失 | Mechanism Hypothesis + Active Probe + Structure Repair | P0 |
| 质量/惯量/摩擦/阻尼错误 | Proprioception + CMA-ES | P0 |
| SAM3 外观匹配失败 | 几何/功能部件定位与跨域视觉匹配 | P1 |
| 铰链轴、碰撞、限位错误 | 运动学 Probe + 几何接触修订 | P1 |

## 当前占位与待接入 (TODO 汇总)

- [x] MuJoCo 后端: `backends/mujoco_backend.py` (object-space 直驱, 见 configs/mujoco_world.example.yaml)
- [ ] 机器人臂/真机后端: 同一 BaseWorld 模板, 补本体感受通道 (q/tau/ee_wrench)
- [ ] `rollout_loss`: MJCF 重放 + 分段轨迹 Huber 损失 (cma_es.py)
- [ ] CMA-ES 真实优化循环 (`cma` 已声明为 optional dependency)
- [ ] 结构/参数修复的 XML 编辑 (model_editor.py, 当前写 sidecar 清单)
- [ ] 验证器真实重放 held-out Probe (validator.py)
- [ ] CaP sim/World rollout (cap/evaluator.py)
- [ ] IG_struct 的 p(o|a,H) 似然评估 + IG_param 的 Fisher 信息
- [ ] LLM 驱动的假设生成 (Full Agentic RSR)
