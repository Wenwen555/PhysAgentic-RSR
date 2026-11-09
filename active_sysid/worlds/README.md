# Worlds: MJCF 场景存放处

## reconstructed/ — 初始重建仿真 (Probe 的对象)

视觉重建导出的、物理上有缺陷的 MJCF。双门场景可从上游仓库导出, 例如:

```bash
# 从 ../agentic_rsr 的 stage3 导出结果复制/链接过来
ln -s ../agentic_rsr/runs/sim_processed_stage3_all/<scene>.mjcf reconstructed/unidoor_left.mjcf
```

当前已知六类偏差 (见 docs/research_plan.md 第十节):
闩锁结构缺失、质量/惯量/摩擦/阻尼错误、全局坐标旋转错位、SAM3 外观
匹配失败、Panda 与 FR3 不一致、铰链轴/碰撞/限位错误。

其中坐标系错位 (SE(3) 配准) 与机器人统一应在任何 Physical Probe 之前修复。

## oracle/ — 真值 World (仅评测)

带完整机构 (闩锁/弹簧/门栓/耦合) 与真值参数的 World, 只允许:

1. 评测脚本读取 (与 configs/eval_truth.yaml 配套);
2. 模拟"真实交互"的后端 (实现 WorldInterface, 但不得暴露隐藏关节)。

禁止被 planner / identification / simulator 模块直接加载真值。
