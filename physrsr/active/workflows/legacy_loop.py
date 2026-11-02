"""Task -> policy -> predictive model -> useful probe -> model/policy revision."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from ..core.contracts import API_VERSION, Hypothesis, finite, normalize
from ..core.policy import Policy, PolicyStore
from ..core.reasoning import (Probe, assess, feature_error, posterior, predictive_error,
                        probe_value, validate_predictions)
from ..runtime.robot import public_observation


class LegacyActiveLoop:
    def __init__(self, task, config, models, *, agent, simulator, target, output):
        self.task, self.cfg, self.models = task, config, normalize(models)
        self.agent, self.simulator, self.target = agent, simulator, target
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.store = PolicyStore(self.output / "policies")
        self.fit_data, self.validation_data, self.history = [], [], []
        self.prediction_cache = {}
        self.model_version = self.probe_count = self.model_revisions = 0
        self.spent = 0.0
        self.last_assessment = None
        self.used_evidence_ids = set()
        self.write("config.json", dict(task=task.public(), loop=asdict(config), models=self.model_json()))

    def model_json(self):
        return [asdict(m) for m in self.models]

    def write(self, name, value):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

    def event(self, kind, **data):
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event=kind, **data), ensure_ascii=False, allow_nan=False) + "\n")

    def call(self, endpoint, method, params):
        self.event("api_request", method=method, params=params)
        result = endpoint.call(method, params)
        self.event("api_response", method=method, result=result)
        return result

    def simulate_params(self):
        return dict(task=self.task.public(), models=self.model_json(), seeds=list(self.cfg.model_seeds))

    def generate_initial_cap(self):
        from .initial_cap import generate_initial_cap
        owner = self

        class LoggedEndpoint:
            def __init__(self, endpoint):
                self.endpoint = endpoint

            def call(self, method, params):
                return owner.call(self.endpoint, method, params)

        policy = generate_initial_cap(self.task, target=LoggedEndpoint(self.target),
            agent=LoggedEndpoint(self.agent), output=self.output / "initial_cap")
        self.store.add(policy, round_index=-1)
        return policy

    def predict(self, probe):
        # Belief weights can change without changing any forward prediction.
        key = json.dumps(dict(policy=probe.policy.sha256, models=[dict(id=m.id, parameters=m.parameters,
            mechanism=m.mechanism) for m in self.models]), sort_keys=True)
        if key not in self.prediction_cache:
            result = self.call(self.simulator, "sim.predict_probe", dict(**self.simulate_params(), probe=probe.wire()))
            if result.get("policy_sha256") != probe.policy.sha256 or result.get("feature_schema") != "rod_response_v1":
                raise ValueError("Simulator probe hash/feature contract mismatch")
            validate_predictions(result["predictions"], self.models, self.cfg.feature_scales)
            self.prediction_cache[key] = result["predictions"]
        return self.prediction_cache[key]

    def context(self, round_index, observation):
        previous = None
        if self.last_assessment:
            policy = self.store.policies[self.last_assessment["selected"]]
            previous = dict(**policy.wire(), assessment=self.last_assessment)
        return dict(task=self.task.public(), round=round_index, observation=observation,
            models=self.model_json(), model_version=self.model_version, previous_policy=previous,
            prediction_feedback=self.agent_feedback(),
            fit_evidence=[dict(probe=d["probe"].wire(), observation=d["observation"]) for d in self.fit_data],
            used_fit_families=sorted({d["probe"].family for d in self.fit_data}),
            used_validation_families=sorted({v["family"] for v in self.validation_data}),
            validation_reports=self.validation_data,
            remaining_probe_count=self.cfg.max_probes-self.probe_count,
            remaining_probe_cost=self.cfg.max_probe_cost-self.spent)

    def agent_feedback(self):
        if not self.history:
            return None
        history = self.history[-1]
        # All matrix scores, plus bounded public failure trajectories. Complete
        # service responses remain in events.jsonl and each assessment artifact.
        result = {k: v for k, v in history.items() if k != "simulator_feedback"}
        rows = history.get("simulator_feedback", [])
        result["model_policy_scores"] = [{k: v for k, v in r.items() if k != "feedback"} for r in rows]
        failed = sorted(rows, key=lambda r: (r["success_rate"], r["margin"]))[:6]
        result["failure_trajectories"] = []
        for row in failed:
            for outcome in row.get("feedback", [])[:2]:
                item = {k: v for k, v in outcome.items() if k != "observations"}
                samples = outcome.get("observations", [])
                if samples:
                    first_bad = next((i for i, s in enumerate(samples)
                        if abs(s["observation"]["rod_axis"][2]) > .25), len(samples)-1)
                    indices = sorted({0, len(samples)-1, *range(max(0, first_bad-2), min(len(samples), first_bad+3))})
                    item["observations"] = [samples[i] for i in indices]
                result["failure_trajectories"].append(dict(model_id=row["model_id"],
                    policy_sha256=row["policy_sha256"], outcome=item))
        return result

    def revise_model(self, mismatch):
        if self.model_revisions >= self.cfg.max_model_revisions:
            return False
        answer = self.call(self.agent, "agent.revise_model", dict(task=self.task.public(),
            supported_mechanisms=self.simulator_info["mechanisms"], models=self.model_json(), mismatch=mismatch,
            fit_evidence=[dict(probe=d["probe"].wire(), observation=d["observation"]) for d in self.fit_data]))
        proposed = normalize([Hypothesis(**m) for m in answer["models"]])
        if len(proposed) > 128 or any(m.mechanism not in self.simulator_info["mechanisms"] for m in proposed):
            raise ValueError("Agent proposed an unsupported model ensemble")
        old = self.models
        self.models = proposed
        self.model_revisions += 1
        # Refit all fit interactions with the new forward model. Validation
        # observations never enter this likelihood, including mismatch cases.
        for datum in self.fit_data:
            predictions = self.predict(datum["probe"])
            observed = datum["observation"]["features"]
            minimum = min(feature_error(observed, p["features"], self.cfg.feature_scales) for p in predictions.values())
            if minimum ** .5 > self.cfg.max_prediction_error:
                self.models = old
                self.event("model_revision_rejected", reason="Cannot explain accumulated fit interactions", answer=answer)
                return False
            self.models = posterior(self.models, observed, predictions, self.cfg.feature_scales)
        self.model_version += 1
        self.event("model_revised", version=self.model_version, rationale=answer["rationale"], models=self.model_json())
        return True

    def finish(self, status, reason):
        result = dict(status=status, reason=reason, model_version=self.model_version,
            models=self.model_json(), probe_count=self.probe_count, probe_cost=self.spent,
            fit_probe_count=len(self.fit_data), validation_probe_count=len(self.validation_data),
            assessment=self.last_assessment, final_policy=None,
            target_task_executed=False,
            rounds=sum(h.get("phase") != "initial_simulation" for h in self.history))
        if status == "task_sufficient":
            policy = self.store.policies[self.last_assessment["selected"]]
            self.store.freeze(policy, self.output / "final", dict(
                model_version=self.model_version, assessment=self.last_assessment,
                validation=[v for v in self.validation_data if v["model_version"] == self.model_version]))
            result["final_policy"] = dict(path=str(self.output / "final/policy.py"), sha256=policy.sha256)
        self.write("result.json", result)
        progress = [dict(round=h["round"], model_version=h.get("model_version"),
            probe_cost=h.get("probe_cost"), assessment=h.get("assessment"),
            probe_values=h.get("probe_values"), current_validation_count=h.get("current_validation_count"))
            for h in self.history]
        progress.append(dict(event="stop", status=status, model_version=self.model_version,
            probe_count=self.probe_count, probe_cost=self.spent, assessment=self.last_assessment,
            current_validation_count=sum(v["passed"] and v["model_version"] == self.model_version for v in self.validation_data)))
        self.write("progress.json", progress)
        self.event("loop_stopped", **result)
        return result

    def run(self):
        try:
            return self._run()
        except Exception as exc:
            self.finish("infrastructure_error", f"{type(exc).__name__}: {exc}")
            raise

    def _run(self):
        self.simulator_info = self.call(self.simulator, "sim.describe", {})
        target_info = self.call(self.target, "target.describe", {})
        if (self.simulator_info.get("api_version") != API_VERSION or target_info.get("api_version") != API_VERSION
                or self.simulator_info.get("feature_schema") != "rod_response_v1"
                or target_info.get("feature_schema") != "rod_response_v1"):
            raise ValueError("Incompatible simulator/target API")
        if self.task.id not in self.simulator_info["tasks"] or any(
                m.mechanism not in self.simulator_info["mechanisms"] for m in self.models):
            raise ValueError("Simulator does not support task/model mechanism")
        # The initial author sees only task + public API + initial observation.
        # First execute that one policy, then let the agent analyze uncertainty
        # and author probes using actual prediction evidence.
        initial = self.generate_initial_cap()
        rows = self.call(self.simulator, "sim.evaluate", dict(**self.simulate_params(),
            policies=[initial.wire()]))["rows"]
        self.last_assessment = assess(self.models, [initial], rows, self.cfg)
        initial_report = dict(round=-1, phase="initial_simulation", model_version=self.model_version,
            assessment=self.last_assessment, simulator_feedback=rows,
            probe_cost=self.spent, current_validation_count=0)
        self.history.append(initial_report)
        self.write("initial_cap/simulation.json", initial_report)
        for round_index in range(self.cfg.max_rounds):
            observation = public_observation(self.call(self.target, "target.observe", {}))
            context = self.context(round_index, observation)
            self.write(f"rounds/{round_index:03d}/context.json", context)
            proposal = self.call(self.agent, "agent.propose", context)
            self.write(f"rounds/{round_index:03d}/proposal.json", proposal)
            parent = self.last_assessment["selected"] if self.last_assessment else None
            errors = []
            for item in proposal["policies"][:3]:
                try:
                    policy = Policy(item["code"], item["rationale"], parent)
                except (ValueError, SyntaxError) as exc:
                    errors.append(str(exc))
                    continue
                self.store.add(policy, round_index)
            if not self.store.policies:
                self.history.append(dict(round=round_index, program_errors=errors))
                continue
            # Keep the previously selected source as a candidate for every
            # revision. Old files remain immutable even when the pool is bounded.
            candidates = list(self.store.policies.values())[-self.cfg.max_candidates:]
            if parent and all(p.sha256 != parent for p in candidates):
                candidates = [self.store.policies[parent], *candidates[:self.cfg.max_candidates-1]]
            evaluated = self.call(self.simulator, "sim.evaluate", dict(**self.simulate_params(),
                policies=[p.wire() for p in candidates]))["rows"]
            assessment = assess(self.models, candidates, evaluated, self.cfg)
            self.last_assessment = assessment
            probes, plans, predictions = {}, [], {}
            for item in proposal["probes"][:6]:
                try:
                    probe = Probe(item["id"], Policy(item["code"], item["rationale"]), item["cost"],
                                  tuple(item["factors"]), item["family"], item["purpose"])
                    if probe.id in probes:
                        raise ValueError("Duplicate probe id")
                except (ValueError, SyntaxError) as exc:
                    errors.append(str(exc))
                    continue
                probes[probe.id] = probe
                predictions[probe.id] = self.predict(probe)
                plans.append(probe_value(probe, predictions[probe.id], self.models, candidates, evaluated, self.cfg))
            valid = [v for v in self.validation_data if v["model_version"] == self.model_version and v["passed"]]
            history = dict(round=round_index, model_version=self.model_version, assessment=assessment,
                uncertainties=proposal["uncertainties"], probe_values=plans,
                simulator_feedback=evaluated, program_errors=errors,
                current_validation_count=len(valid), probe_cost=self.spent)
            self.history.append(history)
            self.write(f"rounds/{round_index:03d}/assessment.json", history)
            # Sufficiency depends on all proposed fit probes, even those outside
            # the remaining budget. Budget exhaustion cannot manufacture low VOI.
            fit_plans = [p for p in plans if probes[p["probe_id"]].purpose == "fit" and p["predicted_risk"] <= self.cfg.max_risk]
            worthwhile = [p for p in fit_plans if p["net_value"] > self.cfg.min_net_value]
            if assessment["robust"] and len(valid) >= self.cfg.min_validation_probes and not worthwhile:
                return self.finish("task_sufficient", "Robust model-set policy, independent prediction evidence, and low proposed-probe decision value")
            if self.probe_count >= self.cfg.max_probes:
                return self.finish("probe_budget_exhausted", "Exploration budget reached before task sufficiency")
            affordable = [p for p in plans if probes[p["probe_id"]].cost + self.spent <= self.cfg.max_probe_cost
                          and p["predicted_risk"] <= self.cfg.max_risk]
            fit_families = {d["probe"].family for d in self.fit_data}
            fit_sources = {d["probe"].policy.behavior_sha256 for d in self.fit_data}
            validation_sources = {v.get("behavior_sha256", v["policy_sha256"]) for v in self.validation_data}
            validation_families = {v["family"] for v in self.validation_data}
            if assessment["robust"] and not worthwhile and len(valid) < self.cfg.min_validation_probes:
                options = [p for p in affordable if probes[p["probe_id"]].purpose == "validation"
                           and probes[p["probe_id"]].family not in fit_families
                           and probes[p["probe_id"]].family not in validation_families
                           and probes[p["probe_id"]].policy.behavior_sha256 not in fit_sources | validation_sources]
                chosen = min(options, key=lambda p: probes[p["probe_id"]].cost, default=None)
            else:
                options = [p for p in affordable if probes[p["probe_id"]].purpose == "fit" and p["net_value"] > self.cfg.min_net_value]
                chosen = max(options, key=lambda p: p["net_value"], default=None)
            if chosen is None:
                safe = [p for p in plans if p["predicted_risk"] <= self.cfg.max_risk]
                if self.spent >= self.cfg.max_probe_cost or (safe and not affordable):
                    return self.finish("cost_budget_exhausted", "No safe proposed probe fits the remaining physical cost budget")
                # Code-only revision is useful when all hypotheses agree that
                # the current code is poor. Re-enter coding with actual feedback.
                history["next_action"] = "revise_policy"
                self.write(f"rounds/{round_index:03d}/assessment.json", history)
                continue
            probe = probes[chosen["probe_id"]]
            result = self.call(self.target, "target.execute_probe", dict(
                policy=probe.policy.wire(), purpose=probe.purpose, probe_id=probe.id, estimated_cost=probe.cost))
            self.probe_count += 1
            cost = finite(result["cost"], "actual probe cost")
            if cost < 0 or result.get("policy_sha256") != probe.policy.sha256 or result.get("feature_schema") != "rod_response_v1":
                raise ValueError("Target probe response contract mismatch")
            self.spent += cost
            evidence_id = result["evidence_id"]
            if not evidence_id or evidence_id in self.used_evidence_ids:
                raise ValueError("Repeated target evidence cannot count as a new interaction")
            self.used_evidence_ids.add(evidence_id)
            self.write(f"probes/{self.probe_count:03d}.json", dict(probe=probe.wire(), result=result,
                       predictions=predictions[probe.id], model_version=self.model_version))
            if result["status"] != "completed":
                return self.finish("probe_execution_error", result.get("error") or result["status"])
            actual = result["features"]
            predicted = predictions[probe.id]
            if probe.purpose == "fit":
                self.fit_data.append(dict(probe=probe, observation=result))
                minimum = min(feature_error(actual, p["features"], self.cfg.feature_scales) for p in predicted.values()) ** .5
                mismatch = minimum > self.cfg.max_prediction_error
                if not mismatch:
                    self.models = posterior(self.models, actual, predicted, self.cfg.feature_scales)
                    self.model_version += 1
                    self.event("model_updated", version=self.model_version, models=self.model_json(), fit_evidence=evidence_id)
            else:
                error = predictive_error(self.models, actual, predicted, self.cfg)
                mismatch = error > self.cfg.max_prediction_error
                self.validation_data.append(dict(evidence_id=evidence_id, policy_sha256=probe.policy.sha256,
                    behavior_sha256=probe.policy.behavior_sha256, family=probe.family,
                    model_version=self.model_version, prediction_error=error, passed=not mismatch))
                self.event("independent_prediction_check", **self.validation_data[-1])
            if self.spent > self.cfg.max_probe_cost:
                return self.finish("cost_budget_exhausted", "Measured physical probe cost exceeded the budget")
            if mismatch and not self.revise_model(dict(probe=probe.wire(), observed=result,
                    predictions=predicted, purpose=probe.purpose)):
                return self.finish("model_inadequate", "Model set cannot explain interaction; do not collapse uncertainty or deliver a final policy")
            if (probe.purpose == "validation" and not mismatch and assessment["robust"] and not worthwhile
                    and sum(v["passed"] and v["model_version"] == self.model_version for v in self.validation_data)
                    >= self.cfg.min_validation_probes):
                return self.finish("task_sufficient", "Robust model-set policy, fresh independent prediction evidence, and low proposed-probe decision value")
        return self.finish("round_budget_exhausted", "Policy/model revision budget reached before task sufficiency")
