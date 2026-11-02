"""Physic Agent: evidence-backed uncertainty analysis and probe authoring only."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from ..api.client import APIError
from .coding import obj, STRING
from ..core.contracts import finite
from ..core.policy import Policy
from ..core.paths import PROMPTS_ROOT


def strings():
    return dict(type="array", items=STRING, minItems=1)


PHYSIC_SCHEMA = obj(dict(
    observed_facts=dict(type="array", minItems=1, items=obj(dict(statement=STRING, evidence_ids=strings()))),
    task_relevant_uncertainties=dict(type="array", minItems=1, items=obj(dict(
        factor=STRING, task_decision_affected=STRING, supporting_evidence_ids=strings(),
        alternative_explanations=strings(), identifiable_with_current_api=STRING))),
    candidate_probes=dict(type="array", minItems=2, maxItems=4, items=obj(dict(
        id=STRING, description=STRING, target_factors=strings(), expected_discriminating_observations=strings(),
        policy_decision_if_outcomes_differ=STRING, expected_cost_seconds=dict(type="number", exclusiveMinimum=0),
        estimated_rpc_calls=dict(type="integer", minimum=1, maximum=96),
        qualitative_information_value=dict(type="string", enum=["high", "medium", "low"]),
        risks=strings(), selection_tradeoff=STRING))),
    selected_probe_id=STRING,
    selection_reason=STRING,
    code=STRING,
    observation_plan=dict(type="array", minItems=1, items=obj(dict(
        label=STRING, observable_fields=strings(), meaning=STRING))),
    interpretation_rules=dict(type="array", minItems=1, items=obj(dict(
        condition=STRING, hypotheses_supported=STRING, hypotheses_not_resolved=STRING, policy_change=STRING))),
    stop_and_recovery=strings(),
    limitations=strings(),
    status=dict(type="string", enum=["authored_not_executed"]),
    information_value_kind=dict(type="string", enum=["qualitative_not_validated_evsi"]),
))

DEFAULT_PROBE_BUDGET = dict(max_rpc_calls=64, max_expected_seconds=90.,
    max_commanded_lift_above_initial_rod_m=.02, max_observed_abs_axis_z=.18,
    max_grasp_attempts=3, speed_min=.1, speed_max=.35,
    release_requires_supported_rest=True)


def prompt_for(context):
    return (PROMPTS_ROOT / "physic_probe.txt").read_text().format(
        context=json.dumps(context, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def evidence_ids(context):
    ids = {"initial_policy"}
    for episode in context["episodes"]:
        ids.update((episode["episode_id"], episode["summary"]["evidence_id"]))
        ids.update(sample["evidence_id"] for sample in episode["trajectory"])
    return ids


def author_messages(context, previous=None, review=None):
    messages = [dict(role="user", content=prompt_for(context))]
    if previous is not None:
        messages += [dict(role="assistant", content=json.dumps(previous, ensure_ascii=False)),
            dict(role="user", content="Contract/source review only; no new physical or prospective simulation interaction occurred. Preserve the physical analysis and selected primitive unless this review requires a change. Return the complete corrected JSON.\n" +
                json.dumps(review, ensure_ascii=False))]
    return messages


def validate_proposal(payload, context):
    if not isinstance(payload, dict) or set(payload) != set(PHYSIC_SCHEMA["required"]):
        raise ValueError("Physic Agent response fields do not match schema")
    if payload["status"] != "authored_not_executed" or payload["information_value_kind"] != "qualitative_not_validated_evsi":
        raise ValueError("Probe authoring must not claim execution or measured information value")
    candidates = payload["candidate_probes"]
    if not 2 <= len(candidates) <= 4 or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Compare two to four uniquely named candidate probes")
    chosen = next((c for c in candidates if c["id"] == payload["selected_probe_id"]), None)
    if chosen is None:
        raise ValueError("Selected probe must be in the candidate comparison")
    budget = context["probe_budget"]
    if not 0 < finite(chosen["expected_cost_seconds"]) <= budget["max_expected_seconds"]:
        raise ValueError("Selected probe exceeds its declared time budget")
    rpc = chosen["estimated_rpc_calls"]
    if isinstance(rpc, bool) or not isinstance(rpc, int) or not 0 < rpc <= budget["max_rpc_calls"]:
        raise ValueError("Selected probe exceeds its declared RPC budget")
    valid = evidence_ids(context)
    for fact in payload["observed_facts"]:
        if not fact["evidence_ids"] or not set(fact["evidence_ids"]) <= valid:
            raise ValueError("Observed fact cites unavailable evidence")
    for factor in payload["task_relevant_uncertainties"]:
        if not factor["supporting_evidence_ids"] or not set(factor["supporting_evidence_ids"]) <= valid:
            raise ValueError("Uncertainty cites unavailable evidence")
    policy = Policy(payload["code"])
    for node in ast.walk(ast.parse(policy.source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "robot"
                and node.func.attr not in {"observe", "goto_pose", "open_gripper", "close_gripper"}):
            raise ValueError("Probe uses an unsupported robot API method")
    return policy


class PhysicService:
    def __init__(self, output, agent=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.turn = 0
        if agent is None:
            from physrsr.coding.agent import CodexAgent
            agent = CodexAgent()
        self.agent = agent

    def call(self, method, params):
        if method == "physic.revise_predictive_model":
            from .identification import revise_predictive_model
            self.turn += 1
            return revise_predictive_model(self.agent,params,self.output / f"model_{self.turn:04d}")
        if method == "physic.assess_task":
            from .identification import assess_task
            self.turn += 1
            return assess_task(self.agent,params,self.output / f"assessment_{self.turn:04d}")
        if method == "physic.design_identification":
            from .identification import generate_design
            self.turn += 1
            return generate_design(self.agent, params, self.output / f"design_{self.turn:04d}")
        previous, review = None, None
        if method == "physic.revise_probe":
            if set(params) != {"context", "previous_proposal", "review_findings"}:
                raise APIError("Probe review requires context, previous_proposal and review_findings")
            previous, review = params["previous_proposal"], params["review_findings"]
            params = params["context"]
        elif method != "physic.propose_probe":
            raise APIError(f"Unknown Physic Agent operation: {method}")
        expected = {"task", "initial_policy", "evidence_contract", "episodes", "probe_budget"}
        if set(params) != expected:
            raise APIError("Physic Agent accepts only the organized public evidence and probe budget")
        messages = author_messages(params, previous, review)
        for _ in range(3):
            self.turn += 1
            payload = self.agent.generate(messages, self.output / f"turn_{self.turn:04d}", schema=PHYSIC_SCHEMA)
            try:
                validate_proposal(payload, params)
            except (KeyError, ValueError, SyntaxError, TypeError) as exc:
                messages += [dict(role="assistant", content=json.dumps(payload, ensure_ascii=False)),
                    dict(role="user", content=f"Source/schema/evidence validation only: {exc}. Return the complete corrected JSON. No probe was executed; do not invent execution feedback.")]
                continue
            return payload
        raise APIError("Physic Agent authoring failed validation after three attempts")
