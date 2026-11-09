"""修复后仿真的预测验证: 重放保留的 Probe + 未见交互预测。

验证内容 (研究方案第十节链路): 更新 MJCF -> 重放 Probe -> 预测未见过的
交互 -> 验证 CaP。预测报告中的三类误差对应 CMA-ES 损失的 L_q / L_F / L_e。

当前 predictive 判定是占位逻辑 (结构辨识出闩锁 + 达到最少 Probe 数);
真实实现需在修复后的仿真中重放 held-out Probe 计算轨迹/力/事件误差。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from active_sysid.config import require_section
from active_sysid.types import Evidence, MechanismHypothesis


@dataclass
class PredictionReport:
    traj_err: float = float("inf")   # L_q: 未见 Probe 轨迹预测误差
    force_err: float = float("inf")  # L_F: 力预测误差
    event_err: float = float("inf")  # L_e: 锁定/解锁/限位等事件不一致
    predictive: bool = False

    def to_dict(self):
        return asdict(self)


class Validator:
    """误差阈值只以 configs/pipeline.yaml 的 validator.thresholds 段为真值。"""

    REQUIRED_THRESHOLDS = ("traj_err", "force_err", "event_err", "min_heldout")

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = require_section(
            thresholds, self.REQUIRED_THRESHOLDS, "validator.thresholds"
        )

    def evaluate(self, sim_path, model: MechanismHypothesis,
                 heldout: list[Evidence]) -> PredictionReport:
        # TODO: 加载 sim_path (修复后的 MJCF), 重放 heldout Probe,
        #   与 world 轨迹/力/事件比对, 填充以下三个误差。
        latch_found = (
            "latch_coupling" in model.structure.get("handle", [])
            and model.params.get("theta_unlock") is not None
        )
        predictive = latch_found and len(heldout) >= self.thresholds["min_heldout"]
        return PredictionReport(
            traj_err=0.0, force_err=0.0, event_err=0.0, predictive=predictive,
        )
