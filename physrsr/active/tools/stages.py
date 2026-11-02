"""Author or execute one frozen program; no rounds, cases or certification."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

from ..agents.base import validate_schema
from ..agents.coding import SYNTHESIS_SCHEMA, synthesis_prompt
from ..agents.physic import DESIGN_SCHEMA, probe_prompt, validate_probe_design
from ..api.process import LoggedAPI
from ..core.artifacts import save_json
from ..core.belief import merge_identification
from ..core.contracts import IDENTIFICATION_API_VERSION
from ..core.exploration import resumable
from ..core.identification import ROBOT_API, public_goal
from ..core.programs import source_policy, task_candidate
from ..core.scene import object_ids, check_objects
from ..runtime.instrumented import observation


def read_program(bundle, kind):
    bundle=Path(bundle)
    manifest=json.loads((bundle/"manifest.json").read_text())
    if manifest.get("kind")!=kind or manifest.get("api_version")!=IDENTIFICATION_API_VERSION:
        raise ValueError("Expected a matching v2 debug program bundle")
    policy=source_policy((bundle/"program.py").read_text())
    if manifest.get("sha256")!=policy.sha256:
        raise ValueError("Debug source differs from its frozen hash")
    return policy,manifest


def author_program(output, *, kind, task, context, agent, previous=None):
    if kind not in ("probe","policy"):
        raise ValueError("Choose probe or policy authoring")
    prior,parent=read_program(previous,kind) if previous else (None,None)
    request=dict(context,goal=public_goal(task.public(IDENTIFICATION_API_VERSION)),robot_api=ROBOT_API)
    if kind=="probe":
        request.update(purpose="fit",authoring_mode="repair_probe" if prior else "new_experiment")
        if prior:request["previous_probe_source"]=prior.source
        schema,prompt,method=DESIGN_SCHEMA,probe_prompt(request),"physic.design_probe"
    else:
        if prior:request["previous_task_candidate"]=prior.source
        schema,prompt,method=SYNTHESIS_SCHEMA,synthesis_prompt(request),"agent.synthesize_policy"
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/"context.json",request)
    save_json(output/"schema.json",schema)
    (output/"prompt.txt").write_text(prompt)
    answer=LoggedAPI(agent,output/"api.jsonl").call(method,request)
    save_json(output/"response.json",answer)
    validate_schema(answer,schema)
    policy=validate_probe_design(answer,request) if kind=="probe" else task_candidate(answer["code"])
    (output/"program.py").write_bytes(policy.source.encode())
    save_json(output/"analysis.json",{key:value for key,value in answer.items() if key!="code"})
    manifest=dict(kind=kind,status="authored_not_executed",sha256=policy.sha256,api_version=policy.api_version,
        debug_only=True,physical_executions=0,parent_sha256=prior.sha256 if prior else None,
        experiment_id=parent["experiment_id"] if parent else uuid.uuid4().hex)
    save_json(output/"manifest.json",manifest)
    return manifest


def execute_program(output, *, kind, bundle, target, belief=None, resume=False):
    policy,_=read_program(bundle,kind)
    if kind not in ("probe","policy") or resume and kind!="probe":
        raise ValueError("Only a Probe can resume a measured checkpoint")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    api=LoggedAPI(target,output/"api.jsonl")
    memory=deepcopy(belief if belief is not None else api.call("target.describe",{})["identification"])
    memory.setdefault("estimates",[])
    memory.setdefault("instance_id","current")
    observed=observation(api.call("target.observe",{}))
    ids=object_ids(observed)
    if "objects" in observed:
        if memory.get("object_ids",ids)!=ids:
            raise ValueError("Belief belongs to different scene objects")
        if any(e.get("object_id") not in ids for e in memory["estimates"]):
            raise ValueError("Scene estimates require known object IDs")
        memory.update(object_ids=ids,objects={key:dict(estimates=[
            e for e in memory["estimates"] if e["object_id"]==key]) for key in ids})
    if kind=="probe" and resume!=resumable(memory,policy.sha256):
        raise ValueError("Use --resume only for an acknowledged same-source checkpoint; unfinished progress requires --resume")
    save_json(output/"request.json",dict(kind=kind,policy=policy.wire(),belief=memory,resume=resume))
    api.call("target.update_identification",dict(belief=memory))
    result=api.call("target.execute_probe" if kind=="probe" else "target.execute_policy",dict(policy=policy.wire()))
    save_json(output/"result.json",result)
    if result.get("policy_sha256")!=policy.sha256:
        raise ValueError("Target result belongs to a different program")
    for sample in result["observations"]:check_objects(sample["observation"],ids)
    if kind=="probe":
        evidence=result.get("evidence_id")
        if (not isinstance(evidence,str) or not evidence or
                evidence in {row["source_evidence_id"] for row in memory.get("trial_history",[])}):
            raise ValueError("Missing or repeated measured evidence ID")
        returned=result.get("program_return")
        if not isinstance(returned,dict) or not isinstance(returned.get("estimates"),list):
            returned=dict(status="program_error",estimates=[],reason=result.get("error"))
        memory=merge_identification(memory,returned,source_probe=str(Path(bundle).resolve()),evidence_id=evidence,
            checkpoint=result.get("experiment_checkpoint"),program_sha256=policy.sha256)
        save_json(output/"belief.json",memory)
        api.call("target.update_identification",dict(belief=memory))
    summary=dict(kind=kind,status=result["status"],debug_only=True,policy_sha256=policy.sha256,
        program_status=result.get("program_return",{}).get("status") if isinstance(result.get("program_return"),dict) else None,
        evidence_id=result.get("evidence_id"),resumable=resumable(memory,policy.sha256) if kind=="probe" else False,
        task_result=result.get("task_result"),cost_s=result.get("cost_s"),setup_cost_s=result.get("setup_cost_s"))
    save_json(output/"summary.json",summary)
    return summary
