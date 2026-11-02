"""Versioned predictive models with pluggable task adapters, using robot API v2.

Adapters own task geometry, physics bindings and the unchanged success predicate.
The controller and physical scientist do not dispatch on benchmark names.
"""
from __future__ import annotations

from dataclasses import asdict
import copy

from ..api.client import APIError
from ..core.contracts import IDENTIFICATION_API_VERSION, Hypothesis, normalize
from ..core.measurements import FEATURE_SCHEMA, FEATURE_NAMES, FEATURE_SCALES, response_features
from ..core.policy import Policy
from ..tasks.base import TaskAdapter


class PredictiveSimulator:
    def __init__(self, adapters: list[TaskAdapter]):
        adapters = list(adapters)
        self.adapters = {adapter.task_id: adapter for adapter in adapters}
        if len(self.adapters) != len(adapters) or not adapters or any(not key for key in self.adapters):
            raise ValueError("Task adapter IDs must be nonempty and unique")
        self.sessions = {}

    def descriptor(self):
        return dict(api_version=IDENTIFICATION_API_VERSION, feature_schema=FEATURE_SCHEMA,
            feature_names=list(FEATURE_NAMES), feature_scales=list(FEATURE_SCALES),
            tasks=list(self.adapters), mechanisms=sorted({m for a in self.adapters.values() for m in a.mechanisms}),
            adapters={key: dict(mechanisms=list(a.mechanisms), parameter_contracts=a.parameter_contracts)
                for key,a in self.adapters.items()})

    def adapter(self, task):
        try:
            return self.adapters[task["id"]]
        except KeyError:
            raise APIError("No predictive task adapter registered for "+task["id"]) from None

    def models(self, task, values):
        models = normalize([Hypothesis(**m) for m in values])
        if len(models) > 128:
            raise APIError("Too many model hypotheses")
        adapter = self.adapter(task)
        for model in models:
            adapter.validate(task, model)
        return models

    def call(self, method, params):
        if method == "sim.describe" and not params:
            return self.descriptor()
        if method == "sim.configure":
            session_id = params["session_id"]
            if not session_id or session_id in self.sessions:
                raise APIError("Predictive session already exists or is unnamed")
            models = self.models(params["task"], params["models"])
            self.sessions[session_id] = dict(task=copy.deepcopy(params["task"]), version=0,
                models=models, belief=dict(estimates=[]), evidence_ids=[])
            return dict(model_version=0, models=[asdict(m) for m in models])
        try:
            session = self.sessions[params["session_id"]]
        except KeyError:
            raise APIError("Configure a predictive model session first") from None
        if params["model_version"] != session["version"]:
            raise APIError("Stale predictive model version")
        if method == "sim.update_model":
            models = self.models(session["task"], params["models"])
            evidence = params["fit_evidence_ids"]
            if not evidence or len(set(evidence)) != len(evidence):
                raise APIError("Model update requires uniquely identified measured fit evidence")
            if not isinstance(params["belief"].get("estimates"), list):
                raise APIError("Model update requires the measured identification belief")
            # One commit updates both physics and policy-visible identification.
            session.update(models=models, belief=copy.deepcopy(params["belief"]),
                version=session["version"]+1, evidence_ids=list(evidence))
            return dict(model_version=session["version"], models=[asdict(m) for m in models],
                applied_parameters={m.id: m.parameters for m in models}, fit_evidence_ids=evidence)
        if method not in ("sim.evaluate", "sim.predict_probe"):
            raise APIError("Unknown predictive simulator method: "+method)
        # Prospective revisions can be replayed before committing the session.
        models = self.models(session["task"], params.get("models",[asdict(m) for m in session["models"]]))
        belief = params.get("belief", session["belief"])
        seeds = params["seeds"]
        if not seeds or any(isinstance(s,bool) or not isinstance(s,int) or s < 0 for s in seeds):
            raise APIError("Explicit nonnegative model seeds are required")
        policies = ([Policy.from_wire(params["probe"]["policy"])] if method == "sim.predict_probe"
            else [Policy.from_wire(p) for p in params["policies"]])
        if not policies or any(p.api_version != IDENTIFICATION_API_VERSION for p in policies):
            raise APIError("Predictive simulator requires v2 programs")
        adapter = self.adapter(session["task"])
        rows, predictions = [], {}
        for model in models:
            for policy in policies:
                outcomes = [adapter.execute(session["task"],model,policy,seed,belief,
                    "policy" if method == "sim.evaluate" else "probe") for seed in seeds]
                if method == "sim.evaluate":
                    rows.append(dict(model_id=model.id, policy_sha256=policy.sha256,
                        success_rate=sum(o["success"] for o in outcomes)/len(outcomes),
                        risk=max(o["risk"] for o in outcomes), margin=min(o["margin"] for o in outcomes),
                        duration_s=sum(o["duration_s"] for o in outcomes)/len(outcomes), feedback=outcomes))
                else:
                    features = [response_features(o["observations"]) for o in outcomes]
                    predictions[model.id] = dict(features=[sum(x)/len(x) for x in zip(*features)],
                        risk=max(o["risk"] for o in outcomes))
        return dict(model_version=session["version"],rows=rows) if method == "sim.evaluate" else dict(
            model_version=session["version"],predictions=predictions,policy_sha256=policies[0].sha256,
            feature_schema=FEATURE_SCHEMA)
