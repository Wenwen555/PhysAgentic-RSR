
"""M1: 从条件可动性证据识别隐藏闩锁，并估计解锁阈值。"""

from __future__ import annotations

import numpy as np

from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.types import Evidence, MechanismHypothesis


class MechanismIdentifier:
    def update(
        self,
        hypotheses: HypothesisSet,
        evidences: list[Evidence],
    ) -> MechanismHypothesis:

        # 不操作把手时，门确实被阻挡。
        blocked_without_handle = any(
            e.probe.name == "gentle_pull"
            and e.summary["door_immobile"]
            for e in evidences
        )

        blocked_angles: list[float] = []
        open_angles: list[float] = []

        for e in evidences:
            if e.probe.name != "hold_and_pull":
                continue

            # 使用实际观测角度，而非 Probe 的目标角度。
            observed_angle = float(
                np.max(
                    e.trajectory.channels["handle_angle"]
                )
            )

            if e.summary["door_opens"]:
                open_angles.append(observed_angle)
            else:
                blocked_angles.append(observed_angle)

        if not (
            blocked_without_handle
            and blocked_angles
            and open_angles
        ):
            return hypotheses.best()

        lower = max(blocked_angles)
        upper = min(open_angles)

        if lower >= upper:
            raise ValueError(
                "条件可动性观测矛盾：无法形成有效阈值区间"
            )

        h = hypotheses.get("H3_hidden_latch")

        params = dict(h.params)
        params["theta_unlock"] = (
            lower + upper
        ) / 2.0

        bounds = dict(h.param_bounds)
        bounds["theta_unlock"] = (lower, upper)

        structure = {
            k: list(v)
            for k, v in h.structure.items()
        }
        structure["door"] = sorted(
            set(structure["door"]) | {"state_locked"}
        )

        return MechanismHypothesis(
            hid=h.hid,
            structure=structure,
            params=params,
            param_bounds=bounds,
            log_belief=h.log_belief,
            notes=(
                "conditional mobility: "
                f"theta_unlock in [{lower:.4f}, "
                f"{upper:.4f}] rad"
            ),
        )

