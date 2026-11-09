"""物理机构候选假设集合与可信度维护。

研究方案 5.1 节: Planner 维护 H_t = {H_1..H_n}, 每个假设有结构模型 M_i、
连续参数 theta_i 与可信度。双门场景的初始四假设:

- H1: 门未上锁, 但质量较大或摩擦较高
- H2: 门具有回位弹簧或阻尼机构
- H3: 门存在由把手控制的隐藏闩锁
- H4: 门被门框或其他几何结构卡住

当前 update() 使用规则式的启发更新 (对应方案第四节"第一轮"的判别逻辑),
后续替换为 LLM 驱动的假设生成 + p(o|a,H_i) 似然评估。
"""

from __future__ import annotations

import numpy as np

from active_sysid.types import Evidence, MechanismHypothesis

# 连续参数的物理先验: 有界、非真值中心 (真值只存在于 configs/eval_truth.yaml,
# 供评测使用; 辨识链路任何模块不得读取)。界要宽到覆盖真值但不以真值为中心。
PARAM_BOUNDS = {
    "I_d": (0.5, 40.0),        # 门叶绕铰链转动惯量; 真值 ~6.6 (12 kg 门叶) 在界内
    "b_d": (0.01, 2.0),        # 铰链阻尼
    "mu_d": (0.0, 3.0),        # 铰链干摩擦
    "k_h": (0.0, 2.0),         # 把手回位弹簧刚度
    "b_h": (0.0, 0.5),         # 把手阻尼
    "theta_h0": (-0.2, 0.2),   # 弹簧平衡角
    "theta_unlock": (0.1, 1.2),  # 解锁阈值
}

# 各假设的先验点: 粗略物理估计, 刻意偏离真值
_PRIOR = {"I_d": 3.0, "b_d": 0.12, "k_h": 0.15, "b_h": 0.03, "theta_h0": 0.0}

# 初始假设模板, 与 configs/mechanisms.yaml 的 hypothesis_templates 保持一致
_DOOR_SCENE_HYPOTHESES: list[dict] = [
    {
        "hid": "H1_heavy_friction",
        "structure": {"door": ["revolute_joint", "friction_damping"],
                      "handle": ["revolute_joint"]},
        "params": {"mu_d": 1.5},  # 偏向"高摩擦重门"解释
        "notes": "门未上锁, 但质量较大或摩擦较高",
    },
    {
        "hid": "H2_spring_damped",
        "structure": {"door": ["revolute_joint", "friction_damping"],
                      "handle": ["revolute_joint", "torsion_spring"]},
        "params": {},
        "notes": "门具有回位弹簧或阻尼机构",
    },
    {
        "hid": "H3_hidden_latch",
        "structure": {"door": ["revolute_joint", "friction_damping"],
                      "handle": ["revolute_joint", "torsion_spring", "latch_coupling"]},
        "params": {"theta_unlock": 0.85},  # 先验阈值刻意偏高, 由交互证据向下修正
        "notes": "门存在由把手控制的隐藏闩锁",
    },
    {
        "hid": "H4_geometric_jam",
        "structure": {"door": ["revolute_joint", "joint_stop"],
                      "handle": ["revolute_joint"]},
        "params": {"mu_d": 0.2},
        "notes": "门被门框或其他几何结构卡住",
    },
]


class HypothesisSet:
    """假设集合: 维护 log_belief, 提供归一化后验与最优假设。"""

    def __init__(self, hypotheses: list[MechanismHypothesis]):
        self.hypotheses = hypotheses

    # ------------------------------------------------------------------ 构建
    @classmethod
    def initial_door_hypotheses(cls) -> "HypothesisSet":
        hyps = []
        for h in _DOOR_SCENE_HYPOTHESES:
            # 公共先验 + 假设特定覆盖; 不含的键 (如 theta_unlock) 不预设
            params = dict(_PRIOR)
            params.update(h["params"])
            hyps.append(MechanismHypothesis(
                hid=h["hid"],
                structure=h["structure"],
                params=params,
                param_bounds=dict(PARAM_BOUNDS),
                log_belief=0.0,  # 均匀先验
                notes=h["notes"],
            ))
        return cls(hyps)

    # ------------------------------------------------------------------ 读取
    def posterior(self) -> dict[str, float]:
        logs = np.array([h.log_belief for h in self.hypotheses])
        w = np.exp(logs - logs.max())
        w = w / w.sum()
        return {h.hid: float(p) for h, p in zip(self.hypotheses, w)}

    def entropy(self) -> float:
        """归一化熵 (用于结构信息增益的快速代理)。"""
        p = np.array(list(self.posterior().values()))
        h = -(p * np.log(p + 1e-12)).sum()
        return float(h / np.log(len(p)))

    def best(self) -> MechanismHypothesis:
        return max(self.hypotheses, key=lambda h: h.log_belief)

    def get(self, hid: str) -> MechanismHypothesis:
        for h in self.hypotheses:
            if h.hid == hid:
                return h
        raise KeyError(hid)

    # ------------------------------------------------------------------ 更新
    def update(self, evidence: Evidence) -> None:
        """规则式启发更新 (占位实现, 语义见各规则注释)。

        TODO: 替换为基于 p(o | a, H_i) 的贝叶斯似然更新,
              以及 LLM 驱动的新假设生成 (Full Agentic RSR)。
        """
        s = evidence.summary
        if s.get("door_immobile") and s.get("handle_rotates"):
            # "门拉不动但把手能转" -> 闩锁假设的直接证据 (方案第四节 Probe 1+2)
            self.get("H3_hidden_latch").log_belief += 1.0
            self.get("H1_heavy_friction").log_belief -= 0.3
        if s.get("handle_returns"):
            # 释放后把手回弹 -> 回位弹簧假设 (方案第四节 Probe 2 验证)
            for hid in ("H2_spring_damped", "H3_hidden_latch"):
                self.get(hid).log_belief += 0.4
        if s.get("unlock_angle") is not None:
            # 条件可动性: 超过某把手角度门才能拉开 -> 因果证据 (Probe 3)
            self.get("H3_hidden_latch").log_belief += 1.5
            self.get("H4_geometric_jam").log_belief -= 1.0
            self.get("H1_heavy_friction").log_belief -= 0.5
        if s.get("door_opens") and s.get("unlock_angle") is None:
            # 施力即可开门 -> 更可能是普通的重门/高摩擦
            self.get("H1_heavy_friction").log_belief += 0.5
            self.get("H3_hidden_latch").log_belief -= 0.5
            self.get("H4_geometric_jam").log_belief -= 1.0
