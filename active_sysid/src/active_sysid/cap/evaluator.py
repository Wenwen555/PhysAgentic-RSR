"""CaP (Code-as-Policy) 评测: sim 内验证 -> World 迁移执行。

主实验 (Hidden Mechanism Recovery for CaP Transfer) 的最终指标:
在更新后的 sim 中验证成功的 CaP, 迁移至 World 的实际任务成功率。
最有价值的展示是完整因果链:

    无闩锁模型 -> 拉门失败 -> Agent 提出机构假设 -> 选择把手实验
    -> 检测解锁条件 -> 修复仿真 -> CaP 自动改变动作顺序 -> World 成功
"""

from __future__ import annotations

from typing import Any, Protocol


class Policy(Protocol):
    """CaP 策略接口: 一段可在 sim/World 中执行的程序。"""

    def code(self) -> str: ...


class CaPEvaluator:
    def run_in_sim(self, policy: Policy, sim_path: str,
                   n_rollouts: int = 10) -> dict[str, Any]:
        """在修复后的仿真中多次执行 policy, 返回成功率与失败模式。

        TODO: 接入 MuJoCo rollout (复用 ../agentic_rsr 的 stage3 agent_runner
        思路), 统计成功率、平均完成时间、失败阶段 (抓取/旋把/拉门/通过)。
        """
        return {"success_rate": None, "n_rollouts": n_rollouts,
                "status": "not_implemented"}

    def run_in_world(self, policy: Policy, world) -> dict[str, Any]:
        """在 World (真机或 oracle 仿真) 中执行并评测迁移成功率。"""
        return {"success_rate": None, "status": "not_implemented"}
