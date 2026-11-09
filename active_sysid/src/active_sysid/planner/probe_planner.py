"""主动 Probe 选择: 任务条件下的信息增益评分。

研究方案 5.3 节的评分:

    a* = argmax_a [ lambda_s * IG_struct(a)
                  + lambda_p * IG_param(a)
                  + lambda_t * dV_CaP(a)
                  - lambda_c * C(a) ]

当前实现使用确定性代理:
- IG_struct  ~ 模板结构增益 x 当前假设集合归一化熵
              (熵越高, 越需要区分机构的实验)
- IG_param   ~ 模板参数增益 x 最优假设可信度
              (结构已定时, 才值得为参数辨识做实验)
- dV_CaP     ~ 0 (占位, 待实现 CaP 在候选模型中的重放估计)
- C(a)       ~ 模板代价 (执行时间 / 接触风险)

真实实现路线 (TODO):
- IG_struct: 在每个候选假设模型中预测 p(o|a,H_i), 计算期望熵降;
- IG_param:  Fisher 信息 I(a, theta) = sum_t J_t^T Sigma_o^-1 J_t,
             J_t = d o_hat_t / d theta (ASID / SPI-Active 的做法);
- dV_CaP:    在候选仿真模型中重放同一份 CaP 估计成功率改善。
"""

from __future__ import annotations

from typing import Any

from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.probes.library import ProbeLibrary
from active_sysid.config import require_section
from active_sysid.types import ProbeSpec, TaskSpec


def _default_target(task: TaskSpec) -> str:
    """从任务描述解析探测目标; 占位返回双门场景的左门。"""
    # TODO: 由 scene 解析目标物体 (视觉重建的部件列表)
    return "door_0"


class ProbePlanner:
    """λ 权重是研究超参, 只以 configs/pipeline.yaml 的 weights 段为真值:

    lambda_s 结构信息增益 / lambda_p 参数信息增益 /
    lambda_t CaP 价值改善 (占位) / lambda_c 执行代价
    """

    REQUIRED_WEIGHTS = ("lambda_s", "lambda_p", "lambda_t", "lambda_c")

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = require_section(weights, self.REQUIRED_WEIGHTS, "weights")
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def propose(self, task: TaskSpec, scene: dict[str, Any],
                history: list[Any] | None = None) -> HypothesisSet:
        """提出/初始化物理假设集合。

        TODO: 由 LLM 根据 scene 与 CaP 失败轨迹生成候选机构假设,
              而非固定返回双门场景的 H1..H4。
        """
        return HypothesisSet.initial_door_hypotheses()

    def select_probe(self, hypotheses: HypothesisSet, library: ProbeLibrary,
                     task: TaskSpec, executed_keys: set[str] | None = None) -> ProbeSpec:
        """按评分选择下一个 Probe; 已执行过的 (name,target,params) 不重复。"""
        executed_keys = executed_keys or set()
        entropy = hypotheses.entropy()
        posterior = hypotheses.posterior()
        top_belief = max(posterior.values())

        best_spec, best_score = None, -float("inf")
        for template in library.templates:
            spec = template.instantiate(target=_default_target(task))
            if spec.key() in executed_keys:
                continue
            ig_struct = template.info_gain["structural"] * entropy
            ig_param = template.info_gain["parametric"] * top_belief
            dv_cap = 0.0  # 占位: lambda_t 默认为 0
            score = (
                self.weights["lambda_s"] * ig_struct
                + self.weights["lambda_p"] * ig_param
                + self.weights["lambda_t"] * dv_cap
                - self.weights["lambda_c"] * template.cost
            )
            if score > best_score:
                best_spec, best_score = spec, score

        if best_spec is None:
            # 全部模板已执行: 以默认参数之外的一档重新实例化 (简单扰动重试)
            template = library.templates[0]
            best_spec = template.instantiate(
                target=_default_target(task), overrides={"duration": template.default_params.get("duration", 1.0) * 1.5}
            )
        self.history.append({"selected": best_spec.name, "score": best_score})
        return best_spec

    def update(self, result: Any) -> None:
        """吸收验证结果, 调整内部状态 (占位)。"""
        self.history.append({"validator": getattr(result, "to_dict", lambda: result)()})

    def should_stop(self, hypotheses: HypothesisSet, validator_report: Any,
                    n_probes: int, min_probes: int, max_probes: int) -> tuple[bool, str]:
        """停止条件与下游任务绑定 (方案第十节):

        当当前仿真能够预测保留的 Probe 结果, 且进一步物理辨识对 CaP 的
        成功预测改善有限时, 停止探索并执行 CaP。
        """
        if n_probes >= max_probes:
            return True, "max_probes"
        if n_probes >= min_probes and getattr(validator_report, "predictive", False):
            return True, "predictive"
        return False, ""
