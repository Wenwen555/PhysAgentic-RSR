"""Frozen execution inputs and identity checks for a single measured World."""
import hashlib
import json


BUNDLE_VERSION = "physrsr.bundle.v2"


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def execution_identity(description, observed):
    from .scene import object_ids
    identity=description.get("world_identity")
    if not isinstance(identity,dict) or not isinstance(identity.get("world_id"),str) or not identity["world_id"].strip():
        raise ValueError("Target must publish a controller-owned World identity")
    return dict(world_id=identity["world_id"],object_ids=object_ids(observed))


def check_identity(expected, description, observed):
    actual=execution_identity(description,observed)
    if actual!=expected:
        raise ValueError("Frozen identification belongs to a different World/object identity; reidentify this scene")
    return actual


def execute_bundle(bundle, target):
    """Only a matching World may receive frozen memory; task outcome is separate."""
    from pathlib import Path
    from .policy import Policy
    from ..api.client import APIError
    root=Path(bundle)
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest.get("bundle_version")!=BUNDLE_VERSION or manifest.get("status")!="task_sufficient":
        raise ValueError("Execution requires an identity-bound v2 bundle; reidentify historical/other scenes")
    policy=Policy.from_wire(dict(source=(root/"policy.py").read_text(),sha256=manifest["sha256"],api_version=manifest["api_version"]))
    memory=json.loads((root/"identification.json").read_text())
    task=json.loads((root/"task.json").read_text())
    models=json.loads((root/"models.json").read_text())
    config=json.loads((root/"config.json").read_text())
    provenance=json.loads((root/"run_manifest.json").read_text())
    if content_hash(memory)!=manifest["identification_sha256"] or content_hash(task)!=manifest["task_sha256"]:
        raise ValueError("Frozen memory/task hash mismatch")
    if content_hash(models)!=manifest["models_sha256"]:raise ValueError("Frozen model hash mismatch")
    if content_hash(config)!=manifest["config_sha256"] or content_hash(provenance)!=manifest["run_manifest_sha256"]:
        raise ValueError("Frozen execution/provenance hash mismatch")
    from .provenance import verify_snapshot
    verify_snapshot(provenance)
    description=target.call("target.describe",{})
    observed=target.call("target.observe",{})
    check_identity(manifest["world_identity"],description,observed)
    if description.get("runtime_contract")!=manifest["runtime_contract"]:
        raise ValueError("Target execution runtime changed; revalidate the frozen operation")
    contract={key:task[key] for key in ("id","success","constraints")}
    if description.get("task_contract")!=contract:
        raise ValueError("Target task contract differs from the frozen bundle")
    acknowledgement=target.call("target.restore_identification",dict(belief=memory,expected_identity=manifest["world_identity"]))
    if acknowledgement.get("identification_sha256")!=manifest["identification_sha256"]:
        raise APIError("Target did not acknowledge the frozen identification snapshot")
    expected=dict(world_identity=manifest["world_identity"],identification_sha256=manifest["identification_sha256"])
    result=target.call("target.execute_policy",dict(policy=policy.wire(),expected=expected))
    if result.get("policy_sha256")!=policy.sha256:raise APIError("Executed source hash differs from frozen policy")
    if result.get("status")!="completed":return result,2
    score=result.get("task_result")
    if not isinstance(score,dict) or not isinstance(score.get("success"),bool):return result,3
    return result,0 if score["success"] else 2
