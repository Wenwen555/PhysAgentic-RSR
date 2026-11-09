"""共享数据类型: Planner / Probe / Identification / Simulator / CaP 之间的接口契约。

约定 (见 docs/research_plan.md 第 6 节):
- 一次 Probe 的记录 D_i = {u_t, q_t, qdot_t, tau_t, T_t^ee, T_t^obj, F_hat_t},
  全部来自机器人本体可获得的状态, 不直接读取 World 的隐藏关节、等式约束
  或物理参数真值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskSpec:
    """下游任务定义, Planner 的信息增益以它为条件。"""

    task_id: str
    description: str  # 例: "打开左侧房门, 使机器人能够进入下一个房间"
    success_criteria: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeSpec:
    """一次具体化的 Probe (参数化动作模板的实例, 而非固定轨迹)。

    例:
        ProbeSpec(
            name="hold_and_pull",
            target="door_0",
            params={"handle_angle": 0.4, "pull_direction": "door_normal",
                    "max_force": 10.0, "duration": 1.0},
            observations=["robot_joint_state", "joint_torque", "handle_pose",
                          "door_pose", "contact_state"],
        )
    """

    name: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)

    def key(self) -> str:
        """用于去重的稳定标识 (同名不同参数视为不同实验)。"""
        params = dict(sorted(self.params.items()))
        return f"{self.name}:{self.target}:{params}"


@dataclass
class ProbeTrajectory:
    """一次 Probe 执行记录的时间序列。

    channels 中每个条目是 shape (T,) 或 (T, k) 的数组, 至少应包含:
    q / qdot / tau (机器人关节), ee_wrench (末端六维力), est_ext_force
    (扣除机器人自身重力/惯性/摩擦后估计的外部接触力)。
    注意: 不要把 MuJoCo 的 qfrc_actuator 直接当作把手接触力。
    """

    t: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)  # "unlock" / "limit_hit" / "slip" ...

    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


@dataclass
class Evidence:
    """一条交互证据 = Probe 规格 + 轨迹 + 判别性特征摘要。

    summary 供假设更新与机构辨识使用, 例:
        {"door_immobile": True, "handle_rotates": True, "unlock_angle": 0.4}
    """

    probe: ProbeSpec
    trajectory: ProbeTrajectory
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": {
                "name": self.probe.name,
                "target": self.probe.target,
                "params": self.probe.params,
            },
            "summary": self.summary,
            "events": self.trajectory.events,
        }


@dataclass
class MechanismHypothesis:
    """一个物理机构假设 H_i: 结构模型 M_i + 连续参数 theta_i + 可信度。

    structure 示例 (双门场景 H3, 闩锁假设):
        {
            "door":   ["revolute_joint", "friction_damping"],
            "handle": ["revolute_joint", "torsion_spring", "latch_coupling"],
        }
    params 示例:
        {"I_d": ..., "b_d": ..., "mu_d": ..., "k_h": ..., "b_h": ...,
         "theta_h0": ..., "theta_unlock": ...}
    """

    hid: str
    structure: dict[str, list[str]] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)
    param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    log_belief: float = 0.0  # 未归一化的对数可信度
    notes: str = ""

    def belief(self) -> float:
        return float(np.exp(self.log_belief))


@dataclass
class LoopReport:
    """一次 probe loop 的总结, 写入 runs/<run_id>/report.json。"""

    n_probes: int = 0
    probe_keys: list[str] = field(default_factory=list)
    belief_history: list[dict[str, float]] = field(default_factory=list)
    best_hypothesis: dict[str, Any] = field(default_factory=dict)
    fitted_params: dict[str, float] = field(default_factory=dict)
    predictive: bool = False
    cap_result: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
