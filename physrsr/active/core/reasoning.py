"""Finite-model Bayesian identification and task decision value of probes.

Weights and success rates are empirical model beliefs, not calibrated real-world
reliability guarantees. No function here receives true target parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    validation: dict | None = None
    estimator_outputs: tuple = ()

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
    signed=[(finite(a)-finite(p))/s for a,p,s in zip(actual,predicted,scales)]
    residuals=[x*x for x in signed]
    # Gate each object and the shared robot block separately. Adding quiet
    # tracks must not dilute a wrist/model contradiction through a larger N.
    from .measurements import OBJECT_CHANNELS, SHARED_CHANNELS
    if len(scales)==16:
        groups=[tuple(i for i in OBJECT_CHANNELS if i!=4),SHARED_CHANNELS]
        relative=[signed[4]]
    elif len(scales)>16 and (len(scales)-10)%6==0:
        groups=[tuple(i+j for j in (0,1,2,3,5)) for i in range(0,len(scales)-10,6)]+[tuple(range(len(scales)-10,len(scales)))]
        relative=[signed[i+4] for i in range(0,len(scales)-10,6)]
    else:groups=[tuple(range(len(scales)))];relative=[]
    errors=[math.fsum(residuals[i] for i in group)/len(group) for group in groups]
    if relative:
        common=math.fsum(relative)/len(relative)
        errors.extend([common*common,max((r-common)**2 for r in relative)])
    return max(errors)


def residual_energy(actual,predicted,scales):
    """Engineering precision: one common TCP-relative response plus contrasts.

    Relative-object observables share TCP tracking. Treating identical residuals
    on three stationary objects as three independent signals would repeat the
    same robot information even after deduplicating wrist channels.
    """
    if len(actual)!=len(scales) or len(predicted)!=len(scales):raise ValueError("Probe response feature shape mismatch")
    residuals=[(finite(a)-finite(p))/s for a,p,s in zip(actual,predicted,scales)]
    indices=([4] if len(scales)==16 else list(range(4,len(scales)-10,6))
        if len(scales)>16 and (len(scales)-10)%6==0 else [])
    if not indices:return math.fsum(r*r for r in residuals)
    common=math.fsum(residuals[i] for i in indices)/len(indices)
    return math.fsum(r*r for i,r in enumerate(residuals) if i not in indices)+common*common+math.fsum(
        (residuals[i]-common)**2 for i in indices)


def validate_predictions(predictions, models, scales):
    if set(predictions) != {m.id for m in models}:
        raise ValueError("Missing or unknown probe-model prediction")
    for prediction in predictions.values():
        feature_error(prediction["features"], prediction["features"], scales)
        if not 0 <= finite(prediction["risk"]) <= 1:
            raise ValueError("Invalid predicted probe risk")


def posterior(models, actual, predictions, scales):
    validate_predictions(predictions, models, scales)
    log_weights = [(math.log(m.weight) if m.weight>0 else -math.inf) -
                   0.5 * residual_energy(actual,predictions[m.id]["features"],scales)
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
    return max(math.sqrt(feature_error(actual,predictions[m.id]["features"],cfg.feature_scales)) for m in plausible)


def joint_fit(models, data, scales, threshold):
    """Recompute from a uniform physical-support prior over ALL unique evidence.

    A surviving hypothesis must explain every segment. Agent weights are proposal
    annotations, never an already conditioned prior. Replaying the same evidence
    therefore cannot condition twice.
    """
    if not data or len({row["evidence_id"] for row in data})!=len(data):
        raise ValueError("Joint fitting requires unique, nonempty measured evidence")
    signatures=[(m.mechanism,tuple(sorted(m.parameters.items()))) for m in models]
    if len(set(signatures))!=len(signatures):raise ValueError("Duplicate physical hypotheses")
    prior=normalize([replace(m,weight=1.) for m in models])
    errors={m.id:{row["evidence_id"]:feature_error(row["actual"],row["predictions"][m.id]["features"],scales)**.5
        for row in data} for m in prior}
    accepted={key for key,values in errors.items() if max(values.values())<=threshold}
    report=dict(prior="uniform_over_proposed_physical_support",criterion="every_segment_and_object_shared_group_rms_below_threshold",
        threshold=threshold,errors=errors,accepted_model_ids=sorted(accepted),
        proposed_support=[asdict(m) for m in prior],
        evidence_ids=[row["evidence_id"] for row in data])
    if not accepted:return None,report
    # Accumulate the complete likelihood in log space once. Intermediate
    # normalization must not underflow a hypothesis that later evidence revives.
    logs={m.id:-.5*math.fsum(residual_energy(row["actual"],row["predictions"][m.id]["features"],scales) for row in data)
        for m in prior if m.id in accepted}
    peak=max(logs.values())
    fitted=[replace(m,weight=math.exp(logs[m.id]-peak)) for m in prior if m.id in accepted]
    return normalize([m for m in fitted if m.weight>0]),report
