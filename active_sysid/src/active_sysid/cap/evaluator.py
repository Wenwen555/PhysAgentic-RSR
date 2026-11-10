
"""评估现有 CaP 的 Sim / World 执行结果，不负责 LLM 生成。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from active_sysid.backends import create_backend
from active_sysid.cap.runtime import (
    DoorRuntime,
    StepBudgetExceeded,
)
from active_sysid.types import TaskSpec


Policy = Callable[[DoorRuntime], None]


def scene_spec(
    base_spec: dict[str, Any],
    mjcf: str | Path,
) -> dict[str, Any]:
    """加载 M1/M2 MJCF 及对应的闩锁运行时配置。"""
    path = Path(mjcf)

    spec = {
        **base_spec,
        "mjcf": str(path),
    }
    spec.pop("latch", None)

    sidecar = path.with_suffix(
        path.suffix + ".repair.json"
    )

    if sidecar.exists():
        repair = json.loads(
            sidecar.read_text(encoding="utf-8")
        )

        if "latch" in repair:
            spec["latch"] = repair["latch"]

    return spec


class CaPEvaluator:
    def evaluate(
        self,
        policy: Policy,
        backend_spec: dict[str, Any],
        task: TaskSpec,
        *,
        n_rollouts: int = 1,
    ) -> dict[str, Any]:

        criteria = task.success_criteria
        goal = float(criteria["door_angle_min"])
        results = []

        for _ in range(n_rollouts):
            runtime = DoorRuntime(
                create_backend(backend_spec),
                max_steps=int(
                    criteria["max_steps"]
                ),
                max_pull_force_n=float(
                    criteria["max_pull_force_n"]
                ),
            )

            exhausted = False

            try:
                policy(runtime)
            except StepBudgetExceeded:
                exhausted = True

            final = runtime.observe()

            success = (
                not exhausted
                and final["door_angle"] >= goal
            )

            results.append({
                "success": bool(success),
                "budget_exceeded": exhausted,
                "elapsed_s": final["time_s"],
                "action_calls": runtime.calls,
                "final_observation": final,
                "trace": runtime.trace,
            })

        return {
            "task_id": task.task_id,
            "n_rollouts": n_rollouts,
            "success_rate": (
                sum(r["success"] for r in results)
                / n_rollouts
            ),
            "rollouts": results,
        }

    def run_in_sim(
        self,
        policy: Policy,
        sim_spec: dict[str, Any],
        task: TaskSpec,
    ) -> dict:
        return self.evaluate(policy, sim_spec, task)

    def run_in_world(
        self,
        policy: Policy,
        world_spec: dict[str, Any],
        task: TaskSpec,
    ) -> dict:
        return self.evaluate(policy, world_spec, task)

