"""Execute a frozen probe in private Worlds, then make one shared CaP revision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import difflib

from ..api.process import LoggedAPI, service
from ..core.artifacts import save_json
from ..core.policy import Policy
from ..runtime.robot import public_observation
from ..agents.coding import REVISION_SCHEMA, revision_prompt
from .evidence import prepare_initial_evidence, episode_evidence


FAILED_CASES = (("episode_002", "com_minus4cm", -.04), ("episode_003", "com_plus8cm", .08))


def private_world(config, offset):
    return dict(physics=dict(com_offset=offset, friction=config["friction"], grip_force=2.,
        inertial_mode="fixed_mass_com", mass_kg=config["mass_kg"]),
        seeds=[config["seed"]], evaluation_seed=config["seed"])


def probe_summary(episode_id, result):
    first = result["observations"][0]["observation"]
    samples = result["observations"]
    returned = result.get("program_return") or {}
    return dict(episode_id=episode_id, execution_status=result["status"], error=result["error"],
        program_reason=returned.get("reason"), rpc_calls=result["rpc_calls"],
        physical_cost_seconds=result["cost"], wall_seconds=result["wall_s"],
        max_observed_rise_m=max(s["observation"]["rod_position"][2]-first["rod_position"][2] for s in samples),
        max_abs_axis_z=max(abs(s["observation"]["rod_axis"][2]) for s in samples),
        labeled_sample_count=len(returned.get("records", [])),
        timed_hold_verified=returned.get("timed_hold_verified", False))


def execute_world_probes(experiment, *, round_index=1, port=18123):
    experiment = Path(experiment).resolve()
    stage = experiment / f"active_probe_{round_index}"
    execution = stage / "world_execution"
    execution.mkdir(exist_ok=False)
    config = json.loads((experiment / "config.json").read_text())
    manifest = json.loads((stage / "manifest.json").read_text())
    source = (stage / "probe.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != manifest["probe_sha256"]:
        raise ValueError("Probe differs from the frozen authored source")
    policy = Policy(source.decode())
    shutil.copy2(stage / "manifest.json", execution / "authoring_manifest.json")
    save_json(execution / "protocol.json", dict(probe_sha256=policy.sha256, episodes=[x[0] for x in FAILED_CASES],
        policy_shared=True, mass_kg=config["mass_kg"], seed=config["seed"],
        physics_only_in_private_world_service=True, new_probe_sources=0))
    summaries = []
    for episode_id, _, offset in FAILED_CASES:
        folder = execution / episode_id
        folder.mkdir()
        save_json(folder / "evaluator_only_world.json", private_world(config, offset))
        print(f"Executing frozen World probe in {episode_id}", flush=True)
        with service(["serve-world", "--private-config", str(folder / "evaluator_only_world.json")],
                port, folder / "service.log") as endpoint:
            result = LoggedAPI(endpoint, folder / "api.jsonl").call("target.execute_probe", dict(policy=policy.wire()))
        save_json(folder / "result.json", result)
        summary = probe_summary(episode_id, result)
        summaries.append(summary)
        save_json(folder / "summary.json", summary)
        save_json(execution / "summary.json", summaries)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    manifest.update(status="executed_in_world", probe_executed=True, world_probe_executions=len(summaries),
        world_probe_evidence="world_execution/summary.json", real_robot_connected=False)
    save_json(stage / "manifest.json", manifest)
    return summaries


def revision_context(experiment, round_index=1):
    experiment = Path(experiment)
    public = prepare_initial_evidence(experiment)
    stage = experiment / f"active_probe_{round_index}"
    manifest = json.loads((stage / "manifest.json").read_text())
    if not manifest["probe_executed"]:
        raise ValueError("Revision requires actual World probe evidence")
    source = (stage / "probe.py").read_text()
    if Policy(source).sha256 != manifest["probe_sha256"]:
        raise ValueError("Frozen probe hash mismatch")
    episodes = []
    for eid, _, _ in FAILED_CASES:
        result = json.loads((stage / "world_execution" / eid / "result.json").read_text())
        if result["policy_sha256"] != manifest["probe_sha256"]:
            raise ValueError("World probe source mismatch")
        summary = probe_summary(eid, result)
        summary["evidence_id"] = f"{eid}:probe_{round_index}:summary"
        samples = [dict(evidence_id=f"{eid}:probe_{round_index}:sample_{i:03d}", event=s["event"],
            observation=public_observation(s["observation"])) for i, s in enumerate(result["observations"])]
        authoritative = [s["observation"] for s in samples if s["event"] == "observe"]
        labels = []
        for i, record in enumerate((result.get("program_return") or {}).get("records", [])):
            observation = public_observation(record["observation"])
            if observation not in authoritative:
                raise ValueError("Labeled probe return disagrees with host sensor logs")
            labels.append(dict(evidence_id=f"{eid}:probe_{round_index}:label_{i:03d}",
                label=record["label"], observation=observation))
        episodes.append(dict(episode_id=eid, summary=summary, observations=samples, labeled_observations=labels))
    return dict(task=public["task"], initial_policy=public["initial_policy"], initial_evidence=public["episodes"],
        probe=dict(evidence_id="probe", source=source, sha256=manifest["probe_sha256"]),
        world_probe_evidence=episodes, comparison_protocol=dict(one_shared_revised_policy=True,
            fresh_reset_for_each_policy=True, same_initialization_for_paired_comparison=True,
            diagnostic_task_outcomes_not_runtime_sensors=True, total_mass_and_com_frame_inertia_fixed=True,
            future_evaluation_results_unavailable=True))


def revise_cap_once(experiment, *, round_index=1, agent=None, port=18122):
    experiment = Path(experiment).resolve()
    revision = experiment / f"revised_cap_{round_index}"
    revision.mkdir(exist_ok=False)
    context = revision_context(experiment, round_index)
    save_json(revision / "context.json", context)
    save_json(revision / "schema.json", REVISION_SCHEMA)
    (revision / "prompt.txt").write_text(revision_prompt(context))
    if agent is None:
        with service(["serve-agent", "--out", str(revision / "inference")], port, revision / "service.log") as endpoint:
            answer = LoggedAPI(endpoint, revision / "api.jsonl").call("agent.revise_cap_from_probe", context)
    else:
        answer = LoggedAPI(agent, revision / "api.jsonl").call("agent.revise_cap_from_probe", context)
    save_json(revision / "response.json", answer)
    old = Policy.from_wire(context["initial_policy"])
    policy = Policy(answer["code"], answer["rationale"], old.sha256)
    bundle = revision / "bundle"
    bundle.mkdir()
    (bundle / "policy.py").write_bytes(policy.source.encode())
    (revision / "revision.diff").write_text("".join(difflib.unified_diff(old.source.splitlines(True),
        policy.source.splitlines(True), fromfile="initial_cap/policy.py", tofile=f"revised_cap_{round_index}/policy.py")))
    save_json(revision / "analysis.json", {k: v for k, v in answer.items() if k != "code"})
    manifest = dict(status="frozen_before_evaluation", round=round_index, parent_sha256=old.sha256,
        policy_sha256=policy.sha256, logical_revisions=1, new_coding_calls=1,
        one_shared_policy=True, evaluated=False, input_sha256=hashlib.sha256((revision / "context.json").read_bytes()).hexdigest())
    save_json(revision / "manifest.json", manifest)
    print(f"One shared revised policy frozen: {policy.sha256}", flush=True)
    return manifest


def summarize_task(result):
    samples = result["observations"]
    z0 = samples[0]["observation"]["rod_position"][2]
    last = samples[-1]["observation"]
    record = result["record"]
    return dict(status=record["status"], reason=record["reason"], error_kind=result["error_kind"],
        final_rise_m=last["rod_position"][2]-z0,
        max_observed_rise_m=max(s["observation"]["rod_position"][2]-z0 for s in samples),
        max_abs_axis_z=max(abs(s["observation"]["rod_axis"][2]) for s in samples),
        execution_seconds=record["sim_duration_s"],
        policy_rpc_calls=max(0, len(samples)-2), grasp_established=record["grasp_established"])


def evaluate_revision_pair(experiment, *, round_index=1, port=18123):
    experiment = Path(experiment).resolve()
    config = json.loads((experiment / "config.json").read_text())
    revision = experiment / f"revised_cap_{round_index}"
    manifest = json.loads((revision / "manifest.json").read_text())
    initial = Policy((experiment / "initial_cap/bundle/policy.py").read_text())
    revised = Policy((revision / "bundle/policy.py").read_text())
    if initial.sha256 != manifest["parent_sha256"] or revised.sha256 != manifest["policy_sha256"]:
        raise ValueError("Paired evaluation source mismatch")
    destination = revision / "evaluation"
    destination.mkdir(exist_ok=False)
    save_json(destination / "frozen_before_evaluation.json", dict(initial=initial.wire(), revised=revised.wire()))
    comparison = []
    for eid, case, offset in FAILED_CASES:
        folder = destination / eid
        folder.mkdir()
        save_json(folder / "evaluator_only_world.json", private_world(config, offset))
        row = dict(episode_id=eid, evaluator_only_case=case, com_offset_cm=offset*100)
        with service(["serve-world", "--private-config", str(folder / "evaluator_only_world.json")],
                port, folder / "service.log") as endpoint:
            for label, policy in (("initial", initial), ("revised", revised)):
                print(f"World evaluation {eid}/{label}", flush=True)
                result = LoggedAPI(endpoint, folder / "api.jsonl").call("world.evaluate_policy", dict(policy=policy.wire()))
                save_json(folder / f"{label}.json", result)
                if result["error_kind"] == "infrastructure":
                    save_json(revision / "evaluation_error.json", dict(episode_id=eid, policy=label, result=result))
                    raise RuntimeError("World infrastructure failed; do not count as a physical task failure")
                row[label] = summarize_task(result)
                # Full public observations and replay-derived commands, including
                # partial traces of program errors when replay is unavailable.
                if result["record"]["status"] != "error":
                    organized = episode_evidence(f"{eid}:{label}", policy, dict(
                        observations=result["observations"], status=result["record"]["status"], reason=result["record"]["reason"]))
                    save_json(folder / f"{label}_trajectory.json", organized)
                print(json.dumps(dict(episode=eid, policy=label, **row[label]), ensure_ascii=False), flush=True)
        comparison.append(row)
        save_json(revision / "comparison.json", comparison)
    manifest.update(status="evaluated_in_world", evaluated=True, paired_world_task_executions=4)
    save_json(revision / "manifest.json", manifest)
    return comparison
