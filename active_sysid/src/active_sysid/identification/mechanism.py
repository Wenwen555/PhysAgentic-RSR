"""结构 (机构) 模型选择。

研究方案 6.3 节: 结构由 Agent 提议及模型比较确定, 参数由 CMA-ES 优化,
两者职责明确 —— CMA-ES 可以优化弹簧刚度, 却不会凭空在 MJCF 中生成一个
缺失的门栓关节。

model_selection_score 不只看拟合误差, 还要考虑模型复杂度以及未见实验的
预测表现, 避免通过添加任意复杂的机构拟合噪声 (TODO)。
"""

from __future__ import annotations

from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.types import Evidence, MechanismHypothesis


class MechanismIdentifier:
    def update(self, hypotheses: HypothesisSet,
               evidences: list[Evidence]) -> MechanismHypothesis:
        """吸收全部交互证据, 返回 MAP 结构假设, 并把辨识出的机构参数
        (如解锁阈值) 写入其 params。"""
        best = hypotheses.best()

        # 从 hold_and_pull 系列证据中估计解锁阈值: 取能开门的最小把手角。
        unlock_angles = [
            e.summary["unlock_angle"]
            for e in evidences
            if e.summary.get("unlock_angle") is not None
        ]
        if "latch_coupling" in best.structure.get("handle", []):
            best = _clone(best)
            if unlock_angles:
                best.params["theta_unlock"] = min(unlock_angles)
                best.structure["door"] = sorted(
                    set(best.structure.get("door", [])) | {"state_locked"}
                )
                best.notes += " | latch confirmed: door openable iff theta_h >= theta_unlock"
        return best


def _clone(h: MechanismHypothesis) -> MechanismHypothesis:
    return MechanismHypothesis(
        hid=h.hid,
        structure={k: list(v) for k, v in h.structure.items()},
        params=dict(h.params),
        param_bounds={k: tuple(v) for k, v in h.param_bounds.items()},
        log_belief=h.log_belief,
        notes=h.notes,
    )
