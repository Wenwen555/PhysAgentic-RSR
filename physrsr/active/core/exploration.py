"""Task-independent decisions and continuation of measured experiments.

Factor names, settings and inference equations belong to generated programs.
The host only checks scope, evidence provenance, decision alternatives and cost.
"""
from __future__ import annotations

import json
import math

CONTRACT = "scoped_exploration_v1"
PHYSICAL_ACTIONS = {"probe", "execute_probe", "resume_probe"}
ACTIONS = PHYSICAL_ACTIONS | {"repair_probe", "revise_policy", "stop"}
PAUSED_STATUSES = {"paused", "budget_exhausted"}
CHECKPOINT_BYTES = 4096


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def validate_plan(answer, context):
    """Validate a scientist decision without choosing any physical factor."""
    action = answer["action"]
    physical=action in PHYSICAL_ACTIONS or action=="validate"
    allowed=ACTIONS|({"validate"} if context.get("predictive_validation") else set())
    if action not in allowed:
        raise ValueError("Unknown scoped exploration action")
    instances = {i["episode_id"]: i for i in context["instances"]}
    rows = answer["instance_assessments"]
    ids = [r["episode_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(instances):
        raise ValueError("Assess every instance exactly once")
    for row in rows:
        available = set(instances[row["episode_id"]]["available_evidence_ids"])
        if not row["reason"].strip() or not set(row["evidence_ids"]) <= available:
            raise ValueError("Instance assessment cites unavailable evidence")
        if not isinstance(row["sufficient"], bool):
            raise ValueError("Instance sufficiency must be explicit")
        requirements = row["requirements"]
        if not requirements:
            raise ValueError("State the operation's evidence requirements")
        for requirement in requirements:
            if not requirement["quantity_or_condition"].strip() or not requirement["operation_decision"].strip():
                raise ValueError("Evidence must address an operation decision")
            if not isinstance(requirement["met"], bool) or not set(requirement["evidence_ids"]) <= available:
                raise ValueError("Requirement evidence is unavailable")
            if requirement["met"] and not requirement["evidence_ids"]:
                raise ValueError("A met requirement needs recorded evidence")
        if row["sufficient"] != all(r["met"] for r in requirements):
            raise ValueError("Sufficiency disagrees with unmet evidence requirements")
    selected = answer["target_instance_ids"]
    if len(selected) != len(set(selected)) or not set(selected) <= set(instances):
        raise ValueError("Invalid interaction scope")
    interactions = answer["interactions"]
    if not physical:
        if selected or interactions:
            raise ValueError("Code repair/policy revision/stop cannot authorize interaction")
        if answer["estimated_probe_cost_s"] != 0:
            raise ValueError("A code-only action has no new Probe cost")
        return answer
    if not selected or {i["episode_id"] for i in interactions} != set(selected) or len(interactions) != len(selected):
        raise ValueError("Every selected interaction needs its own decision argument")
    assessments = {r["episode_id"]: r for r in rows}
    total = 0.
    for item in interactions:
        eid = item["episode_id"]
        if assessments[eid]["sufficient"]:
            raise ValueError("Do not repeat Probe on an instance already sufficient for the stated decision")
        if not item["uncertainty"].strip() or not item["why_existing_evidence_insufficient"].strip():
            raise ValueError("A new interaction must explain the unresolved ambiguity")
        evidence = item["evidence_ids"]
        if not evidence or not set(evidence) <= set(instances[eid]["available_evidence_ids"]):
            raise ValueError("Interaction cites unavailable or cross-instance evidence")
        outcomes = item["outcomes"]
        if len(outcomes) < 2 or any(not o["condition"].strip() or not o["operation_decision"].strip() for o in outcomes):
            raise ValueError("Describe at least two observable outcomes and their operation decisions")
        if len({o["operation_decision"].strip() for o in outcomes}) < 2:
            raise ValueError("Indistinguishable operation decisions do not justify another interaction")
        total += positive(item["estimated_cost_s"], "Per-instance Probe cost")
    positive(answer["expected_decision_benefit"], "Expected decision benefit")
    cost = positive(answer["estimated_probe_cost_s"], "Probe cost")
    if not math.isclose(cost, total, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Total Probe cost must equal the selected instance costs")
    return answer


def checked_checkpoint(state):
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state must be a JSON object")
    encoded = json.dumps(state, allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > CHECKPOINT_BYTES:
        raise ValueError("Checkpoint state exceeds 4 KiB")
    return json.loads(encoded)


def resumable(memory, program_sha256):
    checkpoint = memory.get("probe_progress", {}).get(program_sha256)
    return bool(checkpoint and checkpoint.get("resumable") is True
        and checkpoint.get("program_sha256") == program_sha256
        and memory.get("instance_id") is not None
        and checkpoint.get("instance_id")==memory["instance_id"])
