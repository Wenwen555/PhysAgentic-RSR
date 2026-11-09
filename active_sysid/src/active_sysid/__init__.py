"""Active SysID: Task-Driven Active Physical System Identification.

研究主线: 让 Agent 根据下游 Code-as-Policy (CaP) 的执行需求, 主动设计能够
区分物理机制、辨识关键参数的交互实验 (Probe), 并利用机器人本体感受更新
可交互仿真, 使仿真能够预测真实任务中的动作结果。

子模块:
- planner:        物理假设维护与下一个 Probe 的主动选择
- probes:         参数化 Probe 模板库与 World 交互执行
- identification: 结构 (机构) 模型选择 + CMA-ES 连续参数拟合
- simulator:      MJCF 结构修复 / 参数修复 + 预测验证
- cap:            CaP 在 sim / World 中的执行评测
"""

from __future__ import annotations

from active_sysid.types import (
    Evidence,
    MechanismHypothesis,
    ProbeSpec,
    ProbeTrajectory,
    TaskSpec,
)

__all__ = [
    "Evidence",
    "MechanismHypothesis",
    "ProbeSpec",
    "ProbeTrajectory",
    "TaskSpec",
]
