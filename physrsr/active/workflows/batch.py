"""Probe-first development loop for independent instances and one shared source.

Private cases stay with the World service. The scientist and coder see semantic
goals, visual inputs and actual public observations, never evaluator parameters.
No initial policy or ungrounded predictive rollout is required.
"""
from pathlib import Path
import hashlib

from .identification import load
from ..core.artifacts import save_json
from ..agents.identification import source_policy


def restore_authored_probe(stage):
    """Recover the exact completed API answer after an orchestrator interruption."""
    stage=Path(stage)
    if (stage/"manifest.json").exists():
        return load(stage/"manifest.json")
    answers=sorted(stage.glob("inference/design_*/turn_*/answer.json"))
    if not answers:
        raise ValueError("No completed model answer to restore")
    answer=load(answers[-1]);policy=source_policy(answer["code"])
    if answer["experimental_design"]["minimum_distinct_settings"]<2:
        raise ValueError("Completed source fails the identification contract")
    save_json(stage/"response.json",answer)
    save_json(stage/"analysis.json",{k:v for k,v in answer.items() if k!="code"})
    (stage/"probe.py").write_bytes(policy.source.encode())
    manifest=dict(status="authored_not_executed",probe_sha256=policy.sha256,api_version=policy.api_version,
        restored_from=str(answers[-1].relative_to(stage)),new_model_call=False,
        context_sha256=hashlib.sha256((stage/"context.json").read_bytes()).hexdigest())
    save_json(stage/"manifest.json",manifest)
    return manifest


# Public batch entry points; scheduling is separated from source recovery.
from .scoped_loop import (complete_cases,review_context,decide,continue_probe_first,
    run_probe_first_batch,ExplorationBudget)
