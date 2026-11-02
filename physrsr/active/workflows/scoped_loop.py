"""Instance-scoped scientific decisions, source repair and measured execution."""
from dataclasses import dataclass,asdict
from pathlib import Path
import math
import shutil

from .identification import (load,world_endpoint,author_identification,execute_identification,
    redesign_identification,synthesize_identified_policy,evaluate_identified_policy,revise_identified_task,
    instance_memory,identification_results,latest_identification)
from .loop import ActiveLoop
from ..api.process import service,LoggedAPI
from ..agents.identification import source_policy
from ..core.artifacts import save_json
from ..core.identification import public_goal,ROBOT_API
from ..core.exploration import CONTRACT,PHYSICAL_ACTIONS,validate_plan,resumable


@dataclass(frozen=True)
class ExplorationBudget:
    max_probe_cost_s:float=1800.
    max_probe_executions:int=24
    probe_cost_weight:float=.0005
    min_net_value:float=0.

    def __post_init__(self):
        if (not math.isfinite(self.max_probe_cost_s) or self.max_probe_cost_s<0
                or isinstance(self.max_probe_executions,bool) or not isinstance(self.max_probe_executions,int)
                or self.max_probe_executions<0
                or not math.isfinite(self.probe_cost_weight) or self.probe_cost_weight<0
                or not math.isfinite(self.min_net_value) or self.min_net_value<0):
            raise ValueError("Invalid exploration budget")


def complete_cases(rows,cases):
    expected=[c["episode_id"] for c in cases];observed=[r["episode_id"] for r in rows]
    return len(expected)==len(set(expected)) and sorted(observed)==sorted(expected)


def execution_cost(root,cases,probe_round):
    records=[(eid,result) for case in cases for eid in [case["episode_id"]]
        for _,_,result in identification_results(root,eid,probe_round+1)]
    total=0.;complete=True;unresolved=[]
    for index in range(1,probe_round+1):
        ledger=Path(root)/f"active_probe_{index}/world_execution/execution_index.json"
        if ledger.exists():
            for attempt in load(ledger):
                if attempt.get("status")=="running":
                    complete=False;unresolved.append(dict(probe_round=index,attempt=attempt["attempt"]))
    for _,result in records:
        for field in ("cost_s","setup_cost_s"):
            value=result.get(field)
            if value is None:complete=False;continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError("Invalid measured interaction cost")
            total+=value
    return dict(executions=len(records),cost_s=total,cost_complete=complete,unresolved_attempts=unresolved)


def authoring_counts(root):
    manifests=[load(p) for p in Path(root).glob("active_probe_*/manifest.json")]
    return dict(program_versions=len(manifests),source_repairs=sum(m.get("change_kind")=="repair" for m in manifests),
        scientific_designs=sum(m.get("change_kind","new_design")=="new_design" for m in manifests))


def review_context(root,probe_round,policy_round=None):
    root=Path(root);stage=root/f"active_probe_{probe_round}";design=load(stage/"context.json")
    cases=load(root/"evaluator_only_cases.json");instances=[]
    for case in cases:
        eid=case["episode_id"];memory=instance_memory(root,eid,probe_round+1)
        evidence={"task_goal","robot_capabilities"}
        initial=design.get("initial_visual_observations",{}).get(eid)
        if initial is not None:evidence.add(f"{eid}:initial_visual")
        latest=latest_identification(root,eid,probe_round+1)
        for _,_,result in identification_results(root,eid,probe_round+1):
            if result.get("evidence_id"):evidence.add(result["evidence_id"])
        item=dict(episode_id=eid,initial_visual=initial,retained_identification=memory,
            available_evidence_ids=sorted(evidence),latest_execution=None)
        if latest:
            result=latest[2]
            item["latest_execution"]=dict(probe_round=latest[0],evidence_id=result.get("evidence_id"),
                execution_status=result.get("status"),error=result.get("error"),
                program_return=result.get("program_return"),
                observations=ActiveLoop.evidence_samples(result.get("observations",[]),limit=18))
        if policy_round is not None:
            path=root/f"identified_cap_{policy_round}/evaluation"/eid/"result.json"
            if path.exists():
                result=load(path);key=f"{eid}:task:{result['policy_sha256']}"
                item["available_evidence_ids"].append(key)
                item["task_feedback"]=dict(evidence_id=key,status=result["record"]["status"],
                    reason=result["record"]["reason"],error_kind=result.get("error_kind"),
                    program_return=result.get("program_return"),
                    observations=ActiveLoop.evidence_samples(result.get("observations",[]),limit=18))
        instances.append(item)
    manifest=load(stage/"manifest.json")
    context=dict(decision_contract=CONTRACT,goal=design["goal"],capabilities=design["capabilities"],
        robot_api=ROBOT_API,instances=instances,initial_evidence=design["initial_evidence"],
        phase="probe_first" if policy_round is None else "candidate_review",
        current_probe=dict(manifest=manifest,source=(stage/"probe.py").read_text()),
        experiment_budget=design["experiment_budget"],
        measured_interactions=execution_cost(root,cases,probe_round),
        review_contract=dict(no_parameter_truth=True,independent_instances=True,
            shared_sources=True,code_repair_never_executes=True))
    if policy_round is not None:context["current_task_policy"]=(root/f"identified_cap_{policy_round}/policy.py").read_text()
    return context


def decide(root,context,index,port):
    folder=Path(root)/"decisions"/f"{index:03d}";folder.mkdir(parents=True,exist_ok=False)
    save_json(folder/"context.json",context)
    with service(["serve-physic","--out",str(folder/"inference")],port,folder/"service.log") as endpoint:
        answer=LoggedAPI(endpoint,folder/"api.jsonl").call("physic.assess_task",context)
    validate_plan(answer,context)
    save_json(folder/"decision.json",answer)
    return answer


def freeze(root,policy_round,probe_round,rows,cases):
    if not complete_cases(rows,cases) or not all(r["status"]=="success" for r in rows):
        raise ValueError("Cannot freeze a partial or unsuccessful evaluation")
    stage=root/f"identified_cap_{policy_round}";policy=source_policy((stage/"policy.py").read_text())
    if any(r.get("policy_sha256")!=policy.sha256 for r in rows):
        raise ValueError("World records do not match the shared frozen source")
    final=root/"final_cap";final.mkdir(exist_ok=False);shutil.copy2(stage/"policy.py",final/"policy.py")
    save_json(final/"manifest.json",dict(status="frozen",api_version=policy.api_version,sha256=policy.sha256,
        entrypoint="run(robot)",one_shared_policy=True,recorded_world_successes=len(rows),
        recorded_world_cases=len(rows),source_stage=stage.name,
        interpretation="Functional acceptance on these instances; not held-out generalization"))
    return dict(status="accepted",probe_round=probe_round,policy_round=policy_round,successes=len(rows),
        cases=len(rows),policy_sha256=policy.sha256,initial_policy_generated=False,
        initial_simulation_executed=False,hardware=False)


def continue_probe_first(root,*,probe_round=1,policy_round=None,max_iterations=8,port=18324,budget=None):
    root=Path(root).resolve();cases=load(root/"evaluator_only_cases.json")
    if len({c["episode_id"] for c in cases})!=len(cases):raise ValueError("Duplicate instance IDs")
    state_path=root/"exploration_state.json"
    saved=load(state_path) if state_path.exists() else {}
    limits=budget or ExplorationBudget(**saved.get("budget",{}))
    if saved:
        probe_round=saved["probe_round"];policy_round=saved["policy_round"]
    history=load(root/"progress.json") if (root/"progress.json").exists() else []
    completion=load(root/"completion.json") if (root/"completion.json").exists() else {}
    if completion.get("status")=="accepted":
        final=load(root/"final_cap/manifest.json")
        rows=load(root/f"identified_cap_{completion['policy_round']}/summary.json")
        policy=source_policy((root/"final_cap/policy.py").read_text())
        if (not complete_cases(rows,cases) or not all(r["status"]=="success" and r["policy_sha256"]==policy.sha256 for r in rows)
                or final["sha256"]!=policy.sha256 or completion["policy_sha256"]!=policy.sha256):
            raise ValueError("Previously accepted artifacts no longer match")
        return completion
    def finish(status,reason,**extra):
        result=dict(status=status,reason=reason,probe_round=probe_round,policy_round=policy_round,
            measured_interactions=execution_cost(root,cases,probe_round),authoring=authoring_counts(root),**extra)
        save_json(root/"completion.json",result);return result
    def persist():
        save_json(root/"progress.json",history)
        save_json(state_path,dict(probe_round=probe_round,policy_round=policy_round,budget=asdict(limits)))
    def accepted(rows):
        result=freeze(root,policy_round,probe_round,rows,cases)
        result.update(measured_interactions=execution_cost(root,cases,probe_round),authoring=authoring_counts(root))
        save_json(root/"completion.json",result);return result
    for _ in range(max_iterations):
        persist()
        if policy_round is not None:
            path=root/f"identified_cap_{policy_round}/summary.json"
            rows=load(path) if path.exists() else []
            if complete_cases(rows,cases) and all(r["status"]=="success" for r in rows):
                return accepted(rows)
        context=review_context(root,probe_round,policy_round)
        measured=context["measured_interactions"]
        if not measured["cost_complete"]:return finish("cost_unknown","A measured setup/interaction cost is missing")
        if measured["cost_s"]>limits.max_probe_cost_s:return finish("cost_budget_exhausted","Measured interaction budget exceeded")
        context["remaining_budget"]=dict(cost_s=max(0.,limits.max_probe_cost_s-measured["cost_s"]),
            executions=max(0,limits.max_probe_executions-measured["executions"]))
        existing=[int(p.name) for p in (root/"decisions").glob("*") if p.is_dir() and p.name.isdigit()]
        answer=decide(root,context,max(existing+[0])+1,port);validate_plan(answer,context)
        action=answer["action"]
        entry=dict(action=action,probe_round=probe_round,policy_round=policy_round,
            decision=answer,measured_before=measured)
        history.append(entry);persist()
        if action=="stop":return finish("exploration_stopped",answer["rationale"])
        if action in PHYSICAL_ACTIONS:
            net=answer["expected_decision_benefit"]-limits.probe_cost_weight*answer["estimated_probe_cost_s"]
            if net<=limits.min_net_value:return finish("low_decision_value","Predicted operation benefit does not cover Probe cost")
            if (len(answer["target_instance_ids"])>context["remaining_budget"]["executions"]
                    or answer["estimated_probe_cost_s"]>context["remaining_budget"]["cost_s"]):
                return finish("probe_budget_exhausted","Requested interactions exceed the remaining measured budget")
        if action in ("repair_probe","probe"):
            next_probe=max([int(p.name.rsplit('_',1)[1]) for p in root.glob("active_probe_*")]+[0])+1
            redesign_identification(root,dict(mode=action,public_measured_review=context,scientist_decision=answer),
                round_index=next_probe,previous_round=probe_round,port=port)
            probe_round=next_probe
            entry["authored_probe_round"]=probe_round
            if action=="repair_probe":
                entry["physical_interactions"]=0;persist();continue
        if action in PHYSICAL_ACTIONS:
            source=source_policy((root/f"active_probe_{probe_round}/probe.py").read_text())
            if action=="execute_probe" and any(resumable(instance_memory(root,eid,probe_round+1),source.sha256)
                    for eid in answer["target_instance_ids"]):
                raise ValueError("Use resume_probe for an unfinished same-source experiment")
            if action=="resume_probe" and any(not resumable(instance_memory(root,eid,probe_round+1),source.sha256)
                    for eid in answer["target_instance_ids"]):
                raise ValueError("Resume requires same-source safe progress for every selected instance")
            execute_identification(root,round_index=probe_round,port=18323,
                instance_ids=answer["target_instance_ids"],resume=action=="resume_probe")
            entry["physical_interactions"]=len(answer["target_instance_ids"])
            entry["executed_sha256"]=source.sha256
            persist()
            after=execution_cost(root,cases,probe_round)
            if not after["cost_complete"]:return finish("cost_unknown","Measured execution cost is incomplete")
            if after["cost_s"]>limits.max_probe_cost_s:
                return finish("cost_budget_exhausted","Measured execution cost exceeded its budget")
            continue
        if action=="revise_policy":
            ready=[r["episode_id"] for r in answer["instance_assessments"] if r["sufficient"]]
            if not ready:raise ValueError("Task synthesis needs at least one instance with sufficient decision evidence")
            next_policy=max([int(p.name.rsplit('_',1)[1]) for p in root.glob("identified_cap_*")]+[0])+1
            kwargs=dict(round_index=next_policy,probe_round=probe_round,port=18322,world_port=18323,
                instance_assessments=answer["instance_assessments"])
            if policy_round is None:synthesize_identified_policy(root,**kwargs)
            else:revise_identified_task(root,previous_round=policy_round,**kwargs)
            policy_round=next_policy
            rows=evaluate_identified_policy(root,round_index=policy_round,port=18323,instance_ids=ready)
            entry["evaluated_instances"]=ready;persist()
            if complete_cases(rows,cases) and all(r["status"]=="success" for r in rows):
                return accepted(rows)
    persist();return finish("iteration_budget_exhausted","Decision/authoring budget reached; frozen progress is resumable")


def run_probe_first_batch(output,task,cases,*,max_iterations=8,budget=None):
    if not cases or len({c["episode_id"] for c in cases})!=len(cases):
        raise ValueError("Supply nonempty independent instances with unique IDs")
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    setup=root/"task_context";setup.mkdir();save_json(root/"evaluator_only_cases.json",cases)
    visual={};capabilities=None
    for case in cases:
        folder=setup/case["episode_id"];folder.mkdir()
        with world_endpoint(case,folder,18323) as endpoint:
            api=LoggedAPI(endpoint,folder/"api.jsonl")
            api.call("target.update_identification",dict(belief=dict(estimates=[],instance_id=case["episode_id"])))
            observed=api.call("target.prepare",{})
            descriptor=api.call("target.describe",{})
        if capabilities is None:capabilities=descriptor
        visual[case["episode_id"]]={k:observed[k] for k in ("time_s","coordinate_frame","object","tcp")}
    context=dict(goal=public_goal(task.public()),capabilities=capabilities,
        initial_visual_observations=visual,current_observation=next(iter(visual.values())),
        initial_evidence=[dict(evidence_id=f"{eid}:initial_visual",source="initial_visual_observation",observation=v)
            for eid,v in visual.items()],phase="bootstrap_identification",purpose="fit",
        evidence_contract=dict(no_prior_policy=True,no_prior_predictions=True,no_parameter_truth=True,
            instances_are_independent=True))
    save_json(setup/"public_context.json",context)
    author_identification(root/"run",public_context=context,private_cases=cases,port=18324)
    # Authored source is not permission to touch every instance. The scoped
    # scientist decision authorizes first execution, repair or early synthesis.
    return continue_probe_first(root/"run",max_iterations=max_iterations,budget=budget)
