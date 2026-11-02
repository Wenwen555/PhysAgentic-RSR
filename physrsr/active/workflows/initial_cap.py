"""Stage-3-style policy.py authoring, before simulation or probe analysis."""
from __future__ import annotations

import json
from pathlib import Path

from ..core.contracts import API_VERSION, IDENTIFICATION_API_VERSION
from ..core.policy import Policy
from ..runtime.robot import public_observation
from ..core.paths import PROMPTS_ROOT

INITIAL_CAP_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["policy"]},
        "command": {"type": "string", "enum": [""]},
        "files": {"type": "array", "minItems": 1, "maxItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "path": {"type": "string", "enum": ["policy.py"]},
                "source": {"type": "string"},
            },
            "required": ["path", "source"],
        }},
    },
    "required": ["mode", "command", "files"],
}


def initial_context(task, observation):
    """Allowlist the initial author's inputs; reject later-loop context."""
    version = task.get("api_version")
    if version not in (API_VERSION, IDENTIFICATION_API_VERSION):
        raise ValueError("Initial authoring API mismatch")
    public_task = {key: task[key] for key in ("id", "instruction", "success", "constraints", "api_version", "robot_api")}
    if version == IDENTIFICATION_API_VERSION:
        from ..runtime.instrumented import observation as measured_observation
        return dict(task=public_task, observation=measured_observation(observation))
    return dict(task=public_task, observation=public_observation(observation))


def initial_cap_prompt(params):
    if set(params) != {"task", "observation"}:
        raise ValueError("Initial authoring accepts only task and public observation")
    context = initial_context(params["task"], params["observation"])
    task = {key: context["task"][key] for key in ("id", "instruction", "success", "constraints")}
    return (PROMPTS_ROOT / "initial_cap.txt").read_text().format(
        task=json.dumps(task, ensure_ascii=False, indent=2, allow_nan=False),
        observation=json.dumps(context["observation"], ensure_ascii=False, indent=2, allow_nan=False),
        robot_api=context["task"]["robot_api"],
    )


def policy_source(payload, api_version=API_VERSION):
    """Extract one policy.py, matching Stage 3's mode/command/files contract."""
    if not isinstance(payload, dict) or set(payload) != {"mode", "command", "files"}:
        raise ValueError("Initial authoring requires mode, command and files")
    if payload["mode"] != "policy" or payload["command"] != "":
        raise ValueError("Initial authoring may only submit a policy, not practice or execute")
    files = payload["files"]
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise ValueError("Exactly one policy.py is required")
    item = files[0]
    if set(item) != {"path", "source"} or item["path"] != "policy.py" or not isinstance(item["source"], str):
        raise ValueError("Initial policy file must be policy.py")
    Policy(item["source"], api_version=api_version)
    return item["source"]


def generate_initial_cap(task, *, target, agent, output, api_version=API_VERSION):
    """Only two APIs are available here; this function cannot run simulation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    observation = target.call("target.observe", {})
    context = initial_context(task.public(api_version), observation)
    prompt = initial_cap_prompt(context)
    for name, value in (("context.json", context), ("schema.json", INITIAL_CAP_SCHEMA)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    (output / "prompt.txt").write_text(prompt)
    payload = agent.call("agent.generate_initial_cap", context)
    (output / "response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    policy = Policy(policy_source(payload, api_version), rationale="Initial task-only CaP; no simulator or probe feedback", api_version=api_version)
    bundle = output / "bundle"
    bundle.mkdir()
    (bundle / "policy.py").write_bytes(policy.source.encode())
    (output / "manifest.json").write_text(json.dumps(dict(
        stage="initial_cap", status="authored", entrypoint="policy.py:run(robot)",
        api_version=api_version, sha256=policy.sha256,
        simulator_executed=False, target_task_executed=False), indent=2))
    return policy
