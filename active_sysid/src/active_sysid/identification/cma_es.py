"""CMA-ES 连续参数拟合。

研究方案第六节 (参考 SPI-Active): 固定真实交互采集到的控制输入, 让仿真
在不同参数下重复执行, 用真实与仿真的轨迹误差驱动 CMA-ES; 采用分段轨迹
预测损失与并行参数采样。

优化目标:

    theta* = argmin_theta  sum_i [ w_q L_q^i + w_F L_F^i + w_e L_e^i ] + lambda R(theta)

    L_q^i = sum_t rho( (q_sim - q_world) / sigma_q )   # rho 为 Huber 损失
    L_F^i = sum_t rho( (F_sim - F_world) / sigma_F )
    L_e   = 锁定/解锁/限位等离散事件的不一致损失

分阶段拟合: (1) 低速响应 -> 静摩擦/库仑摩擦/弹簧刚度; (2) 动态响应 ->
黏性阻尼与转动惯量; (3) 全量记录联合精修; (4) 未见轨迹验证。

注意: 对铰链门, 直接辨识得到的是绕门铰链的转动惯量 I_d, 只有结合可信的
门叶几何与质量分布假设才能进一步换算为质量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from active_sysid.config import require_section
from active_sysid.types import Evidence, MechanismHypothesis


@dataclass
class ParameterFit:
    params: dict[str, float] = field(default_factory=dict)
    loss: float = float("inf")
    n_evals: int = 0
    backend: str = "placeholder"  # "cma-es" | "placeholder"


class CMASysID:
    """优化器设置只以 configs/pipeline.yaml 的 cma_es 段为真值。"""

    REQUIRED_KEYS = ("loss_weights", "sigma0", "max_iter")
    REQUIRED_LOSS_WEIGHTS = ("w_q", "w_F", "w_e", "lambda_reg")

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = require_section(config, self.REQUIRED_KEYS, "cma_es")
        self.weights = require_section(
            cfg["loss_weights"], self.REQUIRED_LOSS_WEIGHTS, "cma_es.loss_weights"
        )
        self.sigma0 = float(cfg["sigma0"])
        self.max_iter = int(cfg["max_iter"])

    # ------------------------------------------------------------------
    def rollout_loss(self, model: MechanismHypothesis, params: dict[str, float],
                     dataset: list[Evidence]) -> float:
        """在给定结构模型与参数下, 重放 Probe 计算分段轨迹预测损失。

        TODO: 用 MuJoCo 加载 model_editor 修复后的 MJCF, 固定 dataset 中
        记录的控制输入 u_t 逐段 rollout, 与 world 轨迹计算 Huber 损失。
        """
        raise NotImplementedError("rollout_loss 依赖 MJCF rollout, 待接入 mujoco 后端")

    def fit(self, model: MechanismHypothesis,
            dataset: list[Evidence]) -> ParameterFit:
        """对已确认的机构模型拟合连续参数。

        依赖 `cma` 包 (pip install active-sysid[opt]); 未安装时退化为
        返回当前假设参数的占位结果, 保证主链路可先跑通。
        """
        theta0 = dict(model.params)
        try:
            import cma  # noqa: PIC001
        except ImportError:
            return ParameterFit(params=theta0, loss=0.0, n_evals=0,
                                backend="placeholder")

        # TODO: 定义参数向量与 rollout_loss 后启用:
        #   es = cma.CMAEvolutionStrategy(x0, self.sigma0, {"maxiter": self.max_iter})
        #   es.optimize(lambda x: self.rollout_loss(model, dict(zip(theta0, x)), dataset))
        return ParameterFit(params=theta0, loss=0.0, n_evals=0, backend="placeholder")
