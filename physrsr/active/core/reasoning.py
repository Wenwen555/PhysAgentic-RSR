"""Finite-model Bayesian identification and task decision value of probes.

Weights and success rates are empirical model beliefs, not calibrated real-world
reliability guarantees. No function here receives true target parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .contracts import Hypothesis, LoopConfig, finite, normalize
from .policy import Policy


@dataclass(frozen=True)
class Probe:
    id: str
    policy: Policy
    cost: float
    factors: tuple[str, ...]
    family: str
    purpose: str = "fit"

    def __post_init__(self):
        if not self.id or not self.family or finite(self.cost) <= 0 or not self.factors:
            raise ValueError("Probe needs id, positive cost, factors and action family")
        if self.purpose not in ("fit", "validation"):
            raise ValueError("Unknown probe purpose")

    def wire(self):
        return dict(id=self.id, policy=self.policy.wire(), cost=self.cost,
                    factors=list(self.factors), family=self.family, purpose=self.purpose)


def support(models, mass):
    selected, total = [], 0.0
    for model in sorted(models, key=lambda m: (-m.weight, m.id)):
        if model.weight <= 0:
            continue
        selected.append(model)
        total += model.weight
        if total >= mass - 1e-12:
            break
    return selected


def validate_evaluations(rows, models, policies):
    expected = {(m.id, p.sha256) for m in models for p in policies}
    keys = [(r["model_id"], r["policy_sha256"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Simulator returned an incomplete/duplicate policy-model matrix")
    for row in rows:
        for field in ("success_rate", "risk"):
            if not 0 <= finite(row[field], field) <= 1:
                raise ValueError(f"Invalid simulator {field}")
        if finite(row["duration_s"]) < 0:
            raise ValueError("Negative predicted execution cost")
        finite(row["margin"])


def utilities(rows, cfg):
    return {(r["model_id"], r["policy_sha256"]):
            r["success_rate"] - cfg.time_weight * r["duration_s"] - cfg.risk_weight * r["risk"]
            for r in rows}


def assess(models, policies, rows, cfg: LoopConfig):
    validate_evaluations(rows, models, policies)
    plausible = support(models, cfg.support_mass)
    table = {(r["model_id"], r["policy_sha256"]): r for r in rows}
    scores = []
    utility = utilities(rows, cfg)
    for policy in policies:
        rr = [table[m.id, policy.sha256] for m in plausible]
        scores.append(dict(policy_sha256=policy.sha256,
            worst_success=min(r["success_rate"] for r in rr),
            worst_risk=max(r["risk"] for r in rr),
            worst_margin=min(r["margin"] for r in rr),
            expected_duration_s=sum(m.weight * table[m.id, policy.sha256]["duration_s"] for m in models),
            expected_utility=sum(m.weight * utility[m.id, policy.sha256] for m in models)))
    robust = [s for s in scores if s["worst_success"] >= cfg.reliability and
              s["worst_risk"] <= cfg.max_risk and s["worst_margin"] >= cfg.min_margin]
    best = max(robust or scores, key=lambda s: s["expected_utility"])
    perfect = sum(m.weight * max(utility[m.id, p.sha256] for p in policies) for m in models)
    current = best["expected_utility"]
    return dict(selected=best["policy_sha256"], robust=bool(robust), scores=scores,
                support=[m.id for m in plausible], support_mass=sum(m.weight for m in plausible),
                decision_regret=max(0.0, perfect - current),
                interpretation="Empirical model-set sufficiency, not a World success certificate")


def feature_error(actual, predicted, scales):
    if len(actual) != len(scales) or len(predicted) != len(scales):
        raise ValueError("Probe response feature shape mismatch")
    return sum(((finite(a) - finite(p)) / s) ** 2 for a, p, s in zip(actual, predicted, scales)) / len(scales)


def validate_predictions(predictions, models, scales):
    if set(predictions) != {m.id for m in models}:
        raise ValueError("Missing or unknown probe-model prediction")
    for prediction in predictions.values():
        feature_error(prediction["features"], prediction["features"], scales)
        if not 0 <= finite(prediction["risk"]) <= 1:
            raise ValueError("Invalid predicted probe risk")


def posterior(models, actual, predictions, scales):
    validate_predictions(predictions, models, scales)
    log_weights = [math.log(max(m.weight, 1e-300)) -
                   0.5 * len(scales) * feature_error(actual, predictions[m.id]["features"], scales)
                   for m in models]
    peak = max(log_weights)
    return normalize([replace(m, weight=math.exp(v - peak)) for m, v in zip(models, log_weights)])


def probe_value(probe, predictions, models, policies, rows, cfg):
    """Predictive-mean quadrature approximation to expected decision improvement.

    Branches are sensor predictions from the evolving simulator, and the
    posterior changes *policy choice*. Parameter entropy is not the objective.
    Observation noise is represented by feature scales in the likelihood;
    finite branches are an approximation, not an exact continuous EVSI integral.
    """
    validate_predictions(predictions, models, cfg.feature_scales)
    utility = utilities(rows, cfg)

    def best_value(belief):
        decision = assess(belief, policies, rows, cfg)
        return next(s["expected_utility"] for s in decision["scores"] if s["policy_sha256"] == decision["selected"])

    expected = sum(m.weight * best_value(posterior(
        models, predictions[m.id]["features"], predictions, cfg.feature_scales)) for m in models)
    value = max(0.0, expected - best_value(models))
    return dict(probe_id=probe.id, expected_decision_value=value,
                net_value=value - cfg.probe_cost_weight * probe.cost,
                predicted_risk=max(predictions[m.id]["risk"] for m in support(models, cfg.support_mass)))


def predictive_error(models, actual, predictions, cfg):
    plausible = normalize(support(models, cfg.support_mass))
    return math.sqrt(sum(m.weight * feature_error(actual, predictions[m.id]["features"], cfg.feature_scales)
                         for m in plausible))
