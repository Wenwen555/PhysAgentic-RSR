
"""M2: CMA-ES 连续参数辨识。只使用 World Probe 观测与修复后的 Sim。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from active_sysid.backends import create_backend
from active_sysid.config import require_section
from active_sysid.types import Evidence, MechanismHypothesis


@dataclass
class ParameterFit:
    params: dict[str, float] = field(default_factory=dict)
    loss: float = float("inf")
    n_evals: int = 0
    backend: str = "unconfigured"
    stages: dict[str, dict[str, float]] = field(
        default_factory=dict
    )


class CMASysID:
    REQUIRED_KEYS = (
        "loss_weights", "sigma0", "max_iter", "seed"
    )
    REQUIRED_LOSS_WEIGHTS = (
        "w_q", "w_F", "w_e", "lambda_reg"
    )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        cfg = require_section(
            config, self.REQUIRED_KEYS, "cma_es"
        )
        self.weights = require_section(
            cfg["loss_weights"],
            self.REQUIRED_LOSS_WEIGHTS,
            "cma_es.loss_weights",
        )
        self.sigma0 = float(cfg["sigma0"])
        self.max_iter = int(cfg["max_iter"])
        self.seed = int(cfg["seed"])

    def _set_params(
        self, params: dict[str, float]
    ) -> None:
        sim = self._sim
        mj = sim._mj
        m = sim.model

        door = sim._joints["door_angle"]
        handle = sim._joints["handle_angle"]

        m.dof_damping[door["vadr"]] = params["b_d"]
        m.dof_frictionloss[door["vadr"]] = params["mu_d"]
        m.dof_damping[handle["vadr"]] = params["b_h"]
        m.jnt_stiffness[handle["jid"]] = params["k_h"]

        # 只缩放 door body 的质量与惯量；
        # 把手子 body 不变。
        ratio = params["I_d"] / self._base_inertia

        m.body_mass[self._door_body_id] = (
            self._base_mass * ratio
        )
        m.body_inertia[self._door_body_id] = (
            self._base_body_inertia * ratio
        )

        mj.mj_resetData(m, sim.data)
        mj.mj_setConst(m, sim.data)

    @staticmethod
    def _huber(
        delta: np.ndarray, scale: float
    ) -> float:
        e = np.abs(delta) / scale
        return float(
            np.mean(
                np.where(
                    e <= 1.0,
                    0.5 * e**2,
                    e - 0.5,
                )
            )
        )

    def _replay_fixed_input(
        self, evidence: Evidence
    ) -> dict[str, np.ndarray]:
        """重放 World 记录的实际广义力序列。"""
        sim = self._sim
        reference = evidence.trajectory

        sim._reset()

        channels = (
            "door_angle",
            "handle_angle",
        )

        recorded = {
            name: []
            for name in (
                "door_angle",
                "door_angle_vel",
                "handle_angle",
                "handle_angle_vel",
            )
        }

        for i in range(len(reference.t)):
            torques = {
                name: float(
                    reference.channels[
                        f"{name}_tau_applied"
                    ][i]
                )
                for name in channels
            }

            sim._apply(torques)
            sim._step()

            snapshot = sim._read_channels()
            for name in recorded:
                recorded[name].append(snapshot[name])

        return {
            name: np.asarray(values)
            for name, values in recorded.items()
        }

    def rollout_loss(
        self,
        model: MechanismHypothesis,
        params: dict[str, float],
        dataset: list[Evidence],
        channels: tuple[
            tuple[str, float], ...
        ] | None = None,
    ) -> float:
        """相同 World 力矩输入下的轨迹预测损失。"""
        if channels is None:
            channels = (
                ("door_angle", 0.2),
                ("handle_angle", 0.2),
            )

        self._set_params(params)
        errors = []

        for evidence in dataset:
            prediction = self._replay_fixed_input(
                evidence
            )
            reference = evidence.trajectory.channels

            loss = np.mean([
                self._huber(
                    prediction[name] - reference[name],
                    scale,
                )
                for name, scale in channels
            ])
            errors.append(float(loss))

        self._n_evals += 1

        return (
            self.weights["w_q"]
            * float(np.mean(errors))
        )

    @staticmethod
    def _encode(
        key: str,
        value: float,
        bounds: tuple[float, float],
    ) -> float:
        lo, hi = bounds

        if key in ("I_d", "b_d"):
            return float(
                (np.log(value) - np.log(lo))
                / (np.log(hi) - np.log(lo))
            )

        return (value - lo) / (hi - lo)

    @staticmethod
    def _decode(
        key: str,
        value: float,
        bounds: tuple[float, float],
    ) -> float:
        lo, hi = bounds

        if key in ("I_d", "b_d"):
            return float(
                np.exp(
                    np.log(lo)
                    + value * (
                        np.log(hi) - np.log(lo)
                    )
                )
            )

        return float(lo + value * (hi - lo))

    def fit(
        self,
        model: MechanismHypothesis,
        dataset: list[Evidence],
        *,
        source_mjcf: str | Path | None = None,
        backend_spec: dict[str, Any] | None = None,
    ) -> ParameterFit:
        # 原有 Mock Pipeline 无可交互 Sim：
        # 保持兼容，但不返回虚假的零损失。
        if source_mjcf is None or backend_spec is None:
            return ParameterFit(backend="unconfigured")

        import cma

        path = Path(source_mjcf)
        spec = dict(backend_spec)
        spec["mjcf"] = str(path)

        repair = json.loads(
            path.with_suffix(
                path.suffix + ".repair.json"
            ).read_text(encoding="utf-8")
        )
        spec["latch"] = repair["latch"]

        self._sim = create_backend(spec)

        m = self._sim.model
        mj = self._sim._mj

        door = self._sim._joints["door_angle"]
        handle = self._sim._joints["handle_angle"]

        self._door_body_id = int(
            mj.mj_name2id(
                m,
                mj.mjtObj.mjOBJ_BODY,
                "door",
            )
        )

        self._base_mass = float(
            m.body_mass[self._door_body_id]
        )
        self._base_body_inertia = (
            m.body_inertia[self._door_body_id].copy()
        )

        ipos = m.body_ipos[self._door_body_id]

        self._base_inertia = float(
            self._base_body_inertia[2]
            + self._base_mass * (
                ipos[0] ** 2 + ipos[1] ** 2
            )
        )

        self._n_evals = 0

        # 起始物理参数来自当前 Sim。
        current = {
            "I_d": self._base_inertia,
            "b_d": float(
                m.dof_damping[door["vadr"]]
            ),
            "mu_d": float(
                m.dof_frictionloss[door["vadr"]]
            ),
            "k_h": float(
                m.jnt_stiffness[handle["jid"]]
            ),
            "b_h": float(
                m.dof_damping[handle["vadr"]]
            ),
        }

        stages = (
            (
                "handle",
                ("k_h", "b_h"),
                [
                    e for e in dataset
                    if e.probe.name
                    == "release_and_observe"
                ],
                (
                    ("handle_angle", 0.15),
                    ("handle_angle_vel", 0.6),
                ),
            ),
            (
                "door",
                ("I_d", "b_d", "mu_d"),
                [
                    e for e in dataset
                    if e.probe.name == "hold_and_pull"
                ],
                (
                    ("door_angle", 0.2),
                    ("door_angle_vel", 0.8),
                ),
            ),
        )

        stage_reports: dict[
            str, dict[str, float]
        ] = {}

        for idx, (
            name, keys, episodes, channels
        ) in enumerate(stages):
            if not episodes:
                raise ValueError(
                    f"M2 训练集缺少 {name} 阶段 Probe"
                )

            x0 = np.array([
                self._encode(
                    k,
                    float(model.params.get(
                        k, current[k]
                    )),
                    model.param_bounds[k],
                )
                for k in keys
            ])
            x0 = np.clip(x0, 0.001, 0.999)

            start_loss = self.rollout_loss(
                model, current, episodes, channels
            )

            def objective(x: np.ndarray) -> float:
                proposal = dict(current)

                proposal.update({
                    k: self._decode(
                        k, float(v),
                        model.param_bounds[k],
                    )
                    for k, v in zip(keys, x)
                })

                fit_loss = self.rollout_loss(
                    model,
                    proposal,
                    episodes,
                    channels,
                )

                regularization = (
                    self.weights["lambda_reg"]
                    * float(np.mean(
                        (np.asarray(x) - x0) ** 2
                    ))
                )

                return fit_loss + regularization

            es = cma.CMAEvolutionStrategy(
                x0,
                self.sigma0,
                {
                    "bounds": [0.0, 1.0],
                    "maxiter": self.max_iter,
                    "seed": self.seed + idx,
                    "verbose": -9,
                },
            )

            while not es.stop():
                xs = es.ask()
                es.tell(
                    xs,
                    [objective(x) for x in xs],
                )

            candidate = dict(current)
            candidate.update({
                k: self._decode(
                    k,
                    float(v),
                    model.param_bounds[k],
                )
                for k, v in zip(
                    keys, es.result.xbest
                )
            })

            final_loss = self.rollout_loss(
                model,
                candidate,
                episodes,
                channels,
            )

            if final_loss < start_loss:
                current = candidate

            stage_reports[name] = {
                "before": start_loss,
                "after": min(
                    start_loss, final_loss
                ),
            }

        loss = self.rollout_loss(
            model, current, dataset
        )

        return ParameterFit(
            params=current,
            loss=loss,
            n_evals=self._n_evals,
            backend="cma-es",
            stages=stage_reports,
        )

