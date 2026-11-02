"""Task-independent identification workflow with a separately owned World fixture.

The semantic goal and archived public evidence are inputs. No task name, unknown
parameter value, preset primitive or numerical estimator is sent to the scientist.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager

from ..api.process import LoggedAPI, service
from ..api.client import HTTPClient
from ..agents.identification import design_prompt, source_policy
from ..core.artifacts import save_json
from ..core.identification import DEFAULT_BUDGET, ROBOT_API
from ..core.exploration import resumable


def load(path):
    return json.loads(Path(path).read_text())


def merge_identification(previous,current,*,source_probe,evidence_id,checkpoint=None,program_sha256=None):
    """Retain valid per-instance measurements independently of recovery status."""
    estimates=list(previous.get("estimates",[]))
    known={json.dumps(e,sort_keys=True) for e in estimates}
    for estimate in current.get("estimates",[]):
        if estimate.get("identifiable") is not True:
            continue
        # A generated program may return the estimates it read from describe().
        # Those are retained evidence, not a second measurement of the object.
        if json.dumps(estimate,sort_keys=True) in known:
            continue
        item=dict(estimate,source_probe=source_probe,source_evidence_id=evidence_id)
        key=json.dumps(item,sort_keys=True)
        if key not in known:
            estimates.append(item);known.add(key)
    history=list(previous.get("trial_history",[]))
    if not any(h["source_evidence_id"]==evidence_id for h in history):
        history.append(dict(source_probe=source_probe,source_evidence_id=evidence_id,
            status=current.get("status"),trials=current.get("trials",[])))
    progress=dict(previous.get("probe_progress",{}))
    if program_sha256 in progress:
        progress[program_sha256]=dict(progress[program_sha256],resumable=False)
    if checkpoint is not None:
        if checkpoint.get("program_sha256")!=program_sha256 or checkpoint.get("instance_id")!=previous.get("instance_id"):
            raise ValueError("Checkpoint belongs to a different instance or frozen source")
        progress[program_sha256]=dict(checkpoint,source_evidence_id=evidence_id)
    return dict(current,estimates=estimates,current_estimates=current.get("estimates",[]),trial_history=history,
        probe_progress=progress,
        instance_id=previous.get("instance_id"),
        evidence_source="per_instance_measured_history",previous_evidence_retained=bool(previous.get("estimates")))


def identification_results(root,episode_id,before_round):
    """Ordered actual executions, including continuations of the same source."""
    for index in range(1,before_round):
        execution=Path(root)/f"active_probe_{index}/world_execution"
        paths=[execution/episode_id/"result.json"]
        paths+=sorted(execution.glob(f"attempts/*/{episode_id}/result.json"))
        for path in paths:
            if path.exists():yield index,path,load(path)


def instance_memory(root,episode_id,before_round):
    memory=dict(estimates=[],instance_id=episode_id)
    for index,_,result in identification_results(root,episode_id,before_round):
        returned=result.get("program_return")
        if not isinstance(returned,dict):returned=dict(status=result.get("status"),estimates=[])
        memory=merge_identification(memory,returned,source_probe=f"active_probe_{index}",
            evidence_id=result.get("evidence_id",f"{episode_id}:probe:{index}"),
            checkpoint=result.get("experiment_checkpoint"),program_sha256=result.get("policy_sha256"))
    return memory


def latest_identification(root,episode_id,before_round):
    records=list(identification_results(root,episode_id,before_round))
    return records[-1] if records else None


@contextmanager
def world_endpoint(case, folder, port):
    """Use any task's measured World API; private local fixtures are optional."""
    if "endpoint" in case:
        yield HTTPClient(case["endpoint"])
        return
    config = Path(folder) / "evaluator_only_world.json"
    save_json(config, case["world"])
    with service(["serve-measured-world", "--private-config", str(config)], port,
            Path(folder) / "service.log") as endpoint:
        yield endpoint


def current_capabilities(root, stage, port):
    """Refresh measured capabilities after explicit episode setup, before codegen."""
    case = load(Path(root) / "evaluator_only_cases.json")[0]
    with world_endpoint(case, stage, port) as endpoint:
        api = LoggedAPI(endpoint, Path(stage) / "capabilities_api.jsonl")
        api.call("target.prepare", {})
        return api.call("target.describe", {})


def contact_settings(samples):
    """Geometric contact proxies; exclude empty-hand tare closures.

    This is not a contact oracle: pose proximity and opening are measured API
    channels. The task evaluator separately checks actual maintained grasp.
    """
    result = []
    for index, sample in enumerate(samples):
        if sample["event"] != "close_gripper":
            continue
        obs = sample["observation"]
        obj = obs["object"]
        dims = obj["dimensions_m"]
        major = max(range(3), key=lambda j: dims[j])
        relative = [obs["tcp"]["position"][j]-obj["position"][j] for j in range(3)]
        body = [sum(x*y for x, y in zip(relative, a)) for a in obj["axes"]]
        if (all(abs(body[j]) <= dims[j]/2+.008 for j in range(3))
                and obs["gripper"]["opening_m"] > min(dims)*.4):
            result.append(dict(sample_index=index, body_coordinate_m=body,
                major_axis_index=major, offset_m=body[major], opening_m=obs["gripper"]["opening_m"]))
    return result


def author_identification(output, *, public_context, private_cases, initial_source=None, port=18124):
    """Generic API entry: fixtures/benchmark construction stays with the caller."""
    root = Path(output).resolve()
    root.mkdir(exist_ok=False)
    initial = root / ("initial_cap" if initial_source is not None else "task_context")
    initial.mkdir()
    if initial_source is not None:
        shutil.copy2(initial_source, initial / "policy.py")
    save_json(initial / "public_evidence.json", public_context["initial_evidence"])
    save_json(root / "evaluator_only_cases.json", private_cases)
    stage = root / "active_probe_1"
    stage.mkdir()
    context = dict(public_context, robot_api=ROBOT_API, experiment_budget=dict(DEFAULT_BUDGET))
    save_json(stage / "context.json", context)
    (stage / "prompt.txt").write_text(design_prompt(context))
    with service(["serve-physic", "--out", str(stage / "inference")], port, stage / "service.log") as endpoint:
        answer = LoggedAPI(endpoint, stage / "api.jsonl").call("physic.design_identification", context)
    save_json(stage / "response.json", answer)
    policy = source_policy(answer["code"])
    (stage / "probe.py").write_text(policy.source)
    save_json(stage / "analysis.json", {k: v for k, v in answer.items() if k != "code"})
    save_json(stage / "manifest.json", dict(status="authored_not_executed", probe_sha256=policy.sha256,
        api_version=policy.api_version, context_sha256=hashlib.sha256((stage / "context.json").read_bytes()).hexdigest(),
        change_kind="new_design",experiment_id=uuid.uuid4().hex,
        task_specific_prompt_or_primitive=False))
    return dict(root=str(root), sha256=policy.sha256)


def execute_identification(root, *, round_index=1, port=18123, instance_ids=None, resume=False):
    root = Path(root).resolve()
    stage = root / f"active_probe_{round_index}"
    policy = source_policy((stage / "probe.py").read_text())
    if policy.sha256 != load(stage / "manifest.json")["probe_sha256"]:
        raise ValueError("Probe differs from frozen source")
    execution = stage / "world_execution"
    cases=load(root/"evaluator_only_cases.json")
    ids=[c["episode_id"] for c in cases] if instance_ids is None else list(instance_ids)
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(c["episode_id"] for c in cases):
        raise ValueError("Select known distinct instances for Probe execution")
    for eid in ids:
        if resume and not resumable(instance_memory(root,eid,round_index+1),policy.sha256):
            raise ValueError("Resume requires an acknowledged safe checkpoint for this instance and source")
    execution.mkdir(exist_ok=True)
    ledger_path=execution/"execution_index.json"
    ledger=load(ledger_path) if ledger_path.exists() else []
    if not ledger and any(execution.glob("*/result.json")):
        ledger=[dict(attempt=1,status="legacy_completed",instance_ids=[p.parent.name for p in execution.glob("*/result.json")])]
    attempt=len(ledger)+1
    location=execution if attempt==1 else execution/"attempts"/f"{attempt:03d}"
    ledger.append(dict(attempt=attempt,status="running",instance_ids=ids,resumed=resume,
        program_sha256=policy.sha256,results=[]))
    save_json(ledger_path,ledger)
    summaries=load(execution/"summary.json") if (execution/"summary.json").exists() else []
    for case in cases:
        eid = case["episode_id"]
        if eid not in ids:continue
        folder = location / eid
        folder.mkdir(parents=True,exist_ok=False)
        print("Identification World execution: "+eid, flush=True)
        memory=instance_memory(root,eid,round_index+1)
        with world_endpoint(case, folder, port) as endpoint:
            api=LoggedAPI(endpoint,folder/"api.jsonl")
            api.call("target.update_identification",dict(belief=memory))
            result = api.call("target.execute_probe", dict(policy=policy.wire()))
        if result.get("policy_sha256")!=policy.sha256:
            raise ValueError("Measured Probe response belongs to a different source")
        save_json(folder / "result.json", result)
        returned = result["program_return"]
        if not isinstance(returned, dict) or not isinstance(returned.get("estimates"), list):
            returned = dict(status="execution_error",reason=result.get("error") or
                "No structured identification output",estimates=[])
        belief = dict(evidence_source="executed_physical_identification", probe_sha256=policy.sha256,
            status=returned.get("status"), reason=returned.get("reason"), estimates=returned["estimates"],
            trials=returned.get("trials",[]),
            recommended_action_parameters=returned.get("recommended_action_parameters", {}),
            embodiment_assessment=returned.get("embodiment_assessment", {}))
        belief=merge_identification(memory,belief,source_probe=stage.name,
            evidence_id=result.get("evidence_id",f"{eid}:probe:{round_index}"),
            checkpoint=result.get("experiment_checkpoint"),program_sha256=policy.sha256)
        save_json(folder / "belief.json", belief)
        save_json(root/"instance_memory"/(eid+".json"),belief)
        summary = dict(episode_id=eid, rpc_calls=result["rpc_calls"], cost_s=result["cost_s"],
            setup_cost_s=result.get("setup_cost_s"),attempt=attempt,policy_sha256=policy.sha256,
            result_path=str((folder/"result.json").relative_to(root)),
            resumable=resumable(belief,policy.sha256),
            status=returned.get("status"), reason=returned.get("reason"),
            gripper_close_commands=sum(s["event"] == "close_gripper" for s in result["observations"]),
            geometric_contact_proxy_count=len(contact_settings(result["observations"])),
            contact_settings=contact_settings(result["observations"]),
            estimates=belief["estimates"],current_estimates=returned["estimates"],
            retained_previous_evidence=belief["previous_evidence_retained"])
        summaries=[r for r in summaries if r["episode_id"]!=eid]+[summary]
        save_json(execution / "summary.json", summaries)
        ledger[-1]["results"].append(summary["result_path"]);save_json(ledger_path,ledger)
        print(json.dumps({k:summary[k] for k in ("episode_id","status","rpc_calls","attempt","resumable")}),flush=True)
    ledger[-1]["status"]="completed";save_json(ledger_path,ledger)
    manifest = load(stage / "manifest.json")
    manifest.update(status="executed_in_world", executions=sum(len(l.get("results",l["instance_ids"])) for l in ledger),
        execution_attempts=len(ledger))
    save_json(stage / "manifest.json", manifest)
    return summaries


def redesign_identification(root, review, *, round_index=2, previous_round=None, port=18124):
    """Replan using public experiment evidence, with no benchmark-specific repair."""
    from .loop import ActiveLoop
    if round_index < 2:
        raise ValueError("Redesign requires a preceding identification round")
    root = Path(root).resolve()
    previous = root / f"active_probe_{previous_round or round_index-1}"
    stage = root / f"active_probe_{round_index}"
    stage.mkdir(exist_ok=False)
    context = load(previous / "context.json")
    context["previous_probe_source"]=(previous/"probe.py").read_text()
    context["prior_identification_evidence"] = []
    scoped=review.get("public_measured_review")
    for case in ([] if scoped else load(root / "evaluator_only_cases.json")):
        latest=latest_identification(root,case["episode_id"],round_index)
        if latest is None:continue
        result=latest[2]
        context["prior_identification_evidence"].append(dict(episode_id=case["episode_id"],
            program_return=result["program_return"],
            execution_status=result["status"],execution_error=result.get("error"),
            observations=ActiveLoop.evidence_samples(result["observations"],limit=18),
            retained_identification=instance_memory(root,case["episode_id"],round_index),
            note="Actual public measurements; independent instances, no parameter truth or future evaluation"))
    context["design_review"] = {k:v for k,v in review.items() if k!="public_measured_review"} if scoped else review
    if scoped:
        context["instances"]=scoped["instances"]
        context.pop("current_task_policy",None)
        if "current_task_policy" in scoped:context["current_task_policy"]=scoped["current_task_policy"]
    repairing=review.get("mode")=="repair_probe"
    context["authoring_mode"]="repair_probe" if repairing else "new_experiment"
    save_json(stage / "context.json", context)
    (stage / "prompt.txt").write_text(design_prompt(context))
    with service(["serve-physic", "--out", str(stage / "inference")], port, stage / "service.log") as endpoint:
        answer = LoggedAPI(endpoint, stage / "api.jsonl").call("physic.design_identification", context)
    save_json(stage / "response.json", answer)
    policy = source_policy(answer["code"])
    (stage / "probe.py").write_text(policy.source)
    save_json(stage / "analysis.json", {k: v for k, v in answer.items() if k != "code"})
    save_json(stage / "manifest.json", dict(status="authored_not_executed", round=round_index,
        probe_sha256=policy.sha256, api_version=policy.api_version, previous_round_public_feedback=True,
        change_kind="repair" if repairing else "new_design",
        experiment_id=load(previous/"manifest.json").get("experiment_id",previous.name) if repairing else uuid.uuid4().hex,
        parent_probe_sha256=load(previous/"manifest.json")["probe_sha256"],
        context_sha256=hashlib.sha256((stage / "context.json").read_bytes()).hexdigest(),
        task_specific_prompt_or_primitive=False))
    return dict(sha256=policy.sha256)


def measured_instance_context(root,eid,before_round):
    from .loop import ActiveLoop
    latest=latest_identification(root,eid,before_round)
    result=latest[2] if latest else {}
    return dict(episode_id=eid,belief=instance_memory(root,eid,before_round),
        program_return=result.get("program_return"),latest_probe_round=latest[0] if latest else None,
        measured_action_boundaries=ActiveLoop.evidence_samples(result.get("observations",[]),limit=18))


def synthesize_identified_policy(root, *, round_index=1, probe_round=None, port=18122, world_port=18123,
        instance_assessments=None):
    root = Path(root).resolve()
    probe_round=round_index if probe_round is None else probe_round
    probe = root / f"active_probe_{probe_round}"
    stage = root / f"identified_cap_{round_index}"
    stage.mkdir(exist_ok=False)
    design = load(probe / "context.json")
    context = dict(goal=design["goal"], robot_api=ROBOT_API, capabilities=current_capabilities(root, stage, world_port),
        initial_evidence=design["initial_evidence"], physical_design=load(probe / "analysis.json"),
        probe_source=(probe / "probe.py").read_text(), instances=[],
        execution_protocol=dict(fresh_supported_setup_before_each_task=True,
            probe_recovery_status_is_retained=True, no_task_outcomes_available=True,
            same_source_uses_current_instance_estimates_from_api=True))
    for case in load(root / "evaluator_only_cases.json"):
        context["instances"].append(measured_instance_context(root,case["episode_id"],probe_round+1))
    context["instance_assessments"]=instance_assessments or []
    save_json(stage / "context.json", context)
    from ..core.paths import PROMPTS_ROOT
    (stage / "prompt.txt").write_text((PROMPTS_ROOT / "identified_policy.txt").read_text().format(
        context=json.dumps(context, ensure_ascii=False, allow_nan=False, separators=(",", ":"))))
    with service(["serve-agent", "--out", str(stage / "inference")], port, stage / "service.log") as endpoint:
        answer = LoggedAPI(endpoint, stage / "api.jsonl").call("agent.synthesize_identified_policy", context)
    save_json(stage / "response.json", answer)
    policy = source_policy(answer["code"])
    (stage / "policy.py").write_text(policy.source)
    save_json(stage / "analysis.json", {k: v for k, v in answer.items() if k != "code"})
    save_json(stage / "manifest.json", dict(status="frozen_before_evaluation", sha256=policy.sha256,
        api_version=policy.api_version, one_shared_policy=True,probe_round=probe_round))
    return dict(sha256=policy.sha256)


def evaluate_identified_policy(root, *, round_index=1, port=18123, resume=False, instance_ids=None):
    root = Path(root).resolve()
    stage = root / f"identified_cap_{round_index}"
    policy = source_policy((stage / "policy.py").read_text())
    if policy.sha256 != load(stage / "manifest.json")["sha256"]:
        raise ValueError("Shared policy differs from the frozen source")
    evaluation = stage / "evaluation"
    evaluation.mkdir(exist_ok=resume or instance_ids is not None)
    cases=load(root/"evaluator_only_cases.json")
    selected=[c["episode_id"] for c in cases] if instance_ids is None else list(instance_ids)
    if not selected or len(selected)!=len(set(selected)) or not set(selected)<=set(c["episode_id"] for c in cases):
        raise ValueError("Select known distinct instances for task evaluation")
    rows=load(stage/"summary.json") if (stage/"summary.json").exists() else []
    for case in cases:
        if case["episode_id"] not in selected:continue
        folder = evaluation / case["episode_id"]
        folder.mkdir(exist_ok=resume)
        probe_round = load(stage / "manifest.json").get("probe_round", round_index)
        belief=instance_memory(root,case["episode_id"],probe_round+1)
        print("Identified task World execution: "+case["episode_id"], flush=True)
        if resume and (folder/"result.json").exists():
            result=load(folder/"result.json")
            if result.get("policy_sha256")!=policy.sha256:
                raise ValueError("Saved evaluation belongs to a different policy")
        else:
            with world_endpoint(case, folder, port) as endpoint:
                api = LoggedAPI(endpoint, folder / "api.jsonl")
                api.call("target.update_identification", dict(belief=belief))
                result = api.call("world.evaluate_policy", dict(policy=policy.wire()))
                if case.get("inspect_compilation", "world" in case):
                    save_json(folder / "evaluator_only_compiled.json", api.call("world.inspect_compilation", {}))
            save_json(folder / "result.json", result)
        record = result["record"]
        if record["status"] == "error" and result.get("error_kind")!="program":
            raise RuntimeError("Task infrastructure error; inspect result rather than counting it as task failure")
        rows=[r for r in rows if r["episode_id"]!=case["episode_id"]]+[dict(episode_id=case["episode_id"],
            status=record["status"],reason=record["reason"],error_kind=result.get("error_kind"),
            policy_sha256=policy.sha256,program_return=result["program_return"])]
        save_json(stage / "summary.json", rows)
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    manifest = load(stage / "manifest.json")
    manifest.update(status="evaluated_in_world", executions=len(rows))
    save_json(stage / "manifest.json", manifest)
    return rows


def revise_identified_task(root, *, previous_round=2, round_index=3, probe_round=None, port=18122, world_port=18123,
        instance_assessments=None):
    """Retain the task candidate and feedback while incorporating newer probes."""
    root = Path(root).resolve()
    previous = root / f"identified_cap_{previous_round}"
    stage = root / f"identified_cap_{round_index}"
    stage.mkdir(exist_ok=False)
    context = load(previous / "context.json")
    cases = load(root / "evaluator_only_cases.json")
    # Read the current instrument contract rather than reusing an old archived
    # descriptor after a controller capability update. No parameter truth exits.
    context["capabilities"] = current_capabilities(root, stage, world_port)
    previous_probe=load(previous/"manifest.json").get("probe_round",previous_round)
    probe_round=previous_probe if probe_round is None else probe_round
    probe=root/f"active_probe_{probe_round}"
    context["physical_design"]=load(probe/"analysis.json")
    context["probe_source"]=(probe/"probe.py").read_text()
    context["instances"] = []
    for case in cases:
        context["instances"].append(measured_instance_context(root,case["episode_id"],probe_round+1))
    context["instance_assessments"]=instance_assessments or []
    context["previous_task_candidate"] = (previous / "policy.py").read_text()
    context["actual_task_feedback"] = []
    for case in cases:
        result_path=previous/"evaluation"/case["episode_id"]/"result.json"
        if not result_path.exists():continue
        raw=load(result_path)
        boundaries = []
        for i, sample in enumerate(raw["observations"]):
            if sample["event"] not in ("goto_pose", "terminal"):
                continue
            obs = sample["observation"]
            boundaries.append(dict(evidence_id=f"{case['episode_id']}:task:sample_{i:03d}",
                event=sample["event"], params=sample.get("params"), time_s=obs["time_s"],
                object=obs["object"], tcp=obs["tcp"], wrench=obs["wrench"],
                motion_feedback=sample.get("motion_feedback", [])))
        context["actual_task_feedback"].append(dict(episode_id=case["episode_id"],
            external_task_status=raw["record"]["status"], external_reason=raw["record"]["reason"],
            program_return=raw["program_return"], boundaries=boundaries))
    context["review_scope"] = dict(public_control_and_goal_feedback=True,
        controller_precision_is_not_sensor_resolution=True, no_new_probe_requested=probe_round==previous_probe,
        no_hidden_physical_parameters=True, same_shared_program_required=True,
        new_candidate_has_not_been_evaluated=True)
    context["execution_protocol"]["no_task_outcomes_available"]=False
    save_json(stage / "context.json", context)
    from ..core.paths import PROMPTS_ROOT
    (stage / "prompt.txt").write_text((PROMPTS_ROOT / "identified_policy.txt").read_text().format(
        context=json.dumps(context, ensure_ascii=False, allow_nan=False, separators=(",", ":"))))
    with service(["serve-agent", "--out", str(stage / "inference")], port, stage / "service.log") as endpoint:
        answer = LoggedAPI(endpoint, stage / "api.jsonl").call("agent.synthesize_identified_policy", context)
    save_json(stage / "response.json", answer)
    from ..agents.identification import task_candidate
    policy = task_candidate(answer["code"])
    (stage / "policy.py").write_text(policy.source)
    save_json(stage / "analysis.json", {k:v for k,v in answer.items() if k != "code"})
    save_json(stage / "manifest.json", dict(status="frozen_before_evaluation", sha256=policy.sha256,
        api_version=policy.api_version, one_shared_policy=True, parent_policy=load(previous / "manifest.json")["sha256"],
        probe_round=probe_round,
        probe_program_revisions=probe_round-previous_probe,
        new_physical_identification_rounds=len({index for case in cases
            for index,_,_ in identification_results(root,case["episode_id"],probe_round+1) if index>previous_probe}),
        task_executions_before_freeze=0))
    return dict(sha256=policy.sha256)
