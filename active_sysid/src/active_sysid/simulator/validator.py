
"""M1: 使用独立 Probe 实测修复后的预测性能。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from active_sysid.backends import create_backend
from active_sysid.config import require_section
from active_sysid.types import Evidence, MechanismHypothesis


@dataclass
class PredictionReport:
    traj_err: float = float("inf")

    # 当前 object-space 后端没有真实接触外力观测。
    force_err: float | None = None

    # 可观测的开门状态分类错误率。
    event_err: float = float("inf")

    structure_predictive: bool = False
    predictive: bool = False
    n_heldout: int = 0

    cases: list[dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Validator:
    REQUIRED_THRESHOLDS = (
        "traj_err",
        "force_err",
        "event_err",
        "min_heldout",
    )

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
    ):
        self.thresholds = require_section(
            thresholds,
            self.REQUIRED_THRESHOLDS,
            "validator.thresholds",
        )

    def evaluate(
        self,
        sim_path: str | Path | None,
        model: MechanismHypothesis,
        heldout: list[Evidence],
        *,
        backend_spec: dict[str, Any] | None = None,
        door_opens_eps: float | None = None,
    ) -> PredictionReport:

        if (
            sim_path is None
            or backend_spec is None
            or door_opens_eps is None
        ):
            return PredictionReport(
                n_heldout=len(heldout)
            )

        sim_path = Path(sim_path)

        spec = dict(backend_spec)
        spec["mjcf"] = str(sim_path)

        # 加载本次修复生成的后端参数。
        sidecar = sim_path.with_suffix(
            sim_path.suffix + ".repair.json"
        )

        if sidecar.exists():
            repair = json.loads(
                sidecar.read_text(encoding="utf-8")
            )
            if "latch" in repair:
                spec["latch"] = repair["latch"]

        simulator = create_backend(spec)

        cases = []

        for evidence in heldout:
            reference = evidence.trajectory

            predicted = simulator.run_probe(
                evidence.probe
            )

            np.testing.assert_allclose(
                reference.t,
                predicted.t,
                atol=1e-9,
                rtol=0,
            )

            door_mae = float(
                np.mean(
                    np.abs(
                        predicted.channels["door_angle"]
                        - reference.channels["door_angle"]
                    )
                )
            )

            handle_mae = float(
                np.mean(
                    np.abs(
                        predicted.channels["handle_angle"]
                        - reference.channels["handle_angle"]
                    )
                )
            )

            actual_opens = bool(
                evidence.summary["door_opens"]
            )

            predicted_opens = bool(
                predicted.channels["door_angle"].max()
                > door_opens_eps
            )

            cases.append({
                "probe": evidence.probe.key(),
                "world_opens": actual_opens,
                "sim_opens": predicted_opens,
                "door_mae": door_mae,
                "handle_mae": handle_mae,
                "opens_mismatch": (
                    actual_opens != predicted_opens
                ),
            })

        if not cases:
            return PredictionReport()

        traj_err = float(
            np.mean([
                0.5 * (
                    c["door_mae"] + c["handle_mae"]
                )
                for c in cases
            ])
        )

        event_err = float(
            np.mean([
                c["opens_mismatch"]
                for c in cases
            ])
        )

        structure_predictive = (
            len(cases)
            >= self.thresholds["min_heldout"]
            and event_err
            <= self.thresholds["event_err"]
        )

        return PredictionReport(
            traj_err=traj_err,
            event_err=event_err,
            structure_predictive=structure_predictive,

            # M1 尚未验证接触力预测和完整动力学。
            predictive=False,

            n_heldout=len(cases),
            cases=cases,
        )

