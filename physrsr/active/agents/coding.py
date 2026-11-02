"""Structured coding/model-revision service; never called by offline checks."""
from __future__ import annotations

import json
from pathlib import Path

from ..api.client import APIError
from ..workflows.initial_cap import INITIAL_CAP_SCHEMA, initial_cap_prompt, policy_source
from ..core.paths import PROMPTS_ROOT
from ..core.policy import Policy


def obj(properties):
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)


STRING = dict(type="string")
POLICY_SCHEMA = obj(dict(code=STRING, rationale=STRING))
REVISION_SCHEMA = obj(dict(code=STRING, rationale=STRING,
    changes=dict(type="array", minItems=1, items=obj(dict(change=STRING,
        evidence_ids=dict(type="array", items=STRING, minItems=1), expected_effect=STRING))),
    remaining_uncertainties=dict(type="array", minItems=1, items=STRING)))


def revision_prompt(params):
    return (PROMPTS_ROOT / "revise_from_probe.txt").read_text().format(
        context=json.dumps(params, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
PROPOSAL_SCHEMA = obj(dict(
    uncertainties=dict(type="array", items=obj(dict(factor=STRING, task_effect=STRING, evidence=STRING))),
    policies=dict(type="array", minItems=1, maxItems=3, items=POLICY_SCHEMA),
    probes=dict(type="array", maxItems=6, items=obj(dict(
        id=STRING, code=STRING, rationale=STRING, cost=dict(type="number", exclusiveMinimum=0),
        factors=dict(type="array", minItems=1, items=STRING), family=STRING,
        purpose=dict(type="string", enum=["fit", "validation"]))))))
PARAMETER_SCHEMA = dict(anyOf=[
    obj(dict(com_offset=dict(type="number"), friction=dict(type="number"))),
    obj(dict(com_offset=dict(type="number"), friction=dict(type="number"), mass_kg=dict(type="number", exclusiveMinimum=0))),
])
MODEL_SCHEMA = obj(dict(rationale=STRING, models=dict(type="array", minItems=1, maxItems=128,
    items=obj(dict(id=STRING, mechanism=STRING, weight=dict(type="number", minimum=0),
                   parameters=PARAMETER_SCHEMA)))))

INSTRUCTIONS = """You are the PhysRSR coding and physical hypothesis agent.
Use only the supplied public task, observations, fit evidence and model predictions.
First state task-relevant unknown factors: explain how each could change success,
action structure, risk or time, and separate observations from causal hypotheses.
Generate complete executable candidate policies; revise the previous selected
policy using its prediction failures across the ensemble. Include a conservative
alternative when useful. Never choose from a fixed template/candidate pool.
Propose finite, modest robot probes that could change policy choice. A probe's
cost is a conservative estimate of physical seconds including setup/recovery.
Include a validation probe from an action family NOT used for fitting. It must
test new dynamics (different grasp/trajectory), not merely rename a fitted action.
Fit and validation are separate purposes; validation data never fit this model.
Probe programs must restore the object to supported rest before release.
All programs use exactly the supplied robot API. The same source is executed
on both backends. Never branch on simulation/hardware or request true physics.
Use Python math only. Return complete sources, bounded loops and small motions.
There is no research Gate and no automatic World task rollout in this loop.
"""


class CodingService:
    def __init__(self, output, agent=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.turn = 0
        if agent is None:
            from physrsr.coding.agent import CodexAgent
            agent = CodexAgent()
        self.agent = agent

    def call(self, method, params):
        if method == "agent.synthesize_identified_policy":
            from .identification import synthesize
            self.turn += 1
            return synthesize(self.agent, params, self.output / f"identified_{self.turn:04d}")
        if method == "agent.revise_cap_from_probe":
            return self.revise_from_probe(params)
        if method == "agent.generate_initial_cap":
            return self.generate_initial_cap(params)
        if method not in ("agent.propose", "agent.revise_model"):
            raise APIError(f"Unknown coding operation: {method}")
        self.turn += 1
        instruction = INSTRUCTIONS
        if method == "agent.revise_model":
            instruction += """\nThe model set could not predict a target interaction.
Propose a revised physical ensemble, preserving plausible alternatives. Reconsider
parameter support and, only with evidence and a supported simulator mechanism,
the mechanism. Use fit data and the mismatch report; no target truth is available.
All previous fit evidence will be replayed against your proposed models. Previous
validation becomes stale; new independent validation will be required. Never
invent support for an unimplemented mechanism or claim it has been validated.
"""
        messages = [dict(role="user", content=instruction + "\nContext:\n" +
                         json.dumps(params, ensure_ascii=False, allow_nan=False))]
        return self.agent.generate(messages, self.output / f"turn_{self.turn:04d}",
                                   schema=MODEL_SCHEMA if method == "agent.revise_model" else PROPOSAL_SCHEMA)

    def generate_initial_cap(self, params):
        prompt = initial_cap_prompt(params)
        messages = [dict(role="user", content=prompt)]
        # Like Stage 3 author_policy(), repair invalid source using the same
        # logical conversation. These turns have no simulator or robot tools.
        for _ in range(3):
            self.turn += 1
            folder = self.output / f"turn_{self.turn:04d}"
            payload = self.agent.generate(messages, folder, schema=INITIAL_CAP_SCHEMA)
            try:
                source = policy_source(payload, params["task"]["api_version"])
            except (ValueError, SyntaxError) as exc:
                messages += [dict(role="assistant", content=json.dumps(payload)),
                    dict(role="user", content=f"Invalid policy.py: {exc}. Repair the complete source using the same output format. This is source validation only; no execution has occurred.")]
                continue
            # Stage 3 keeps prompt, response and a runnable policy.py separately.
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "prompt.txt").write_text(prompt)
            (folder / "policy.py").write_bytes(source.encode())
            return payload
        raise APIError("Initial policy authoring failed source validation after three attempts")

    def revise_from_probe(self, params):
        expected = {"task", "initial_policy", "initial_evidence", "probe", "world_probe_evidence", "comparison_protocol"}
        if set(params) != expected:
            raise APIError("CaP revision accepts only task, frozen sources and public evidence")
        self.turn += 1
        folder = self.output / f"turn_{self.turn:04d}"
        answer = self.agent.generate([dict(role="user", content=revision_prompt(params))], folder, schema=REVISION_SCHEMA)
        Policy(answer["code"])
        valid = {"initial_policy", "probe"}
        for e in params["initial_evidence"]:
            valid.update((e["episode_id"], e["summary"]["evidence_id"]))
            valid.update(s["evidence_id"] for s in e["trajectory"])
        for e in params["world_probe_evidence"]:
            valid.add(e["summary"]["evidence_id"])
            valid.update(s["evidence_id"] for s in e["observations"])
            valid.update(s["evidence_id"] for s in e["labeled_observations"])
        for change in answer["changes"]:
            if not change["evidence_ids"] or not set(change["evidence_ids"]) <= valid:
                raise APIError("Revision cited unavailable evidence")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "policy.py").write_bytes(answer["code"].encode())
        return answer
