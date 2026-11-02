"""Measured identification memory shared by the loop and replay tools."""
import json
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Estimate(BaseModel):
    """Generated numerical evidence: values are finite and use explicit units/frames."""
    model_config=ConfigDict(extra="allow",strict=True,allow_inf_nan=False)
    name: Annotated[str,Field(min_length=1)]
    value: float
    interval: Annotated[list[float],Field(min_length=2,max_length=2)]
    unit: Annotated[str,Field(min_length=1)]
    frame: Annotated[str,Field(min_length=1)]
    identifiable: bool
    assumptions: Annotated[list[str],Field(min_length=1)]

    @model_validator(mode="after")
    def ordered_interval(self):
        if not self.interval[0]<=self.value<=self.interval[1]:
            raise ValueError("Estimate interval must contain its value in increasing order")
        return self


def merge_identification(previous,current,*,source_probe,evidence_id,checkpoint=None,program_sha256=None,contracts=None):
    """Retain valid per-instance measurements independently of recovery status."""
    object_ids = previous.get("object_ids")
    if object_ids is not None:
        for estimate in current.get("estimates", []):
            if estimate.get("identifiable") is True and estimate.get("object_id") not in object_ids:
                raise ValueError("Scene estimates must cite a known object_id")
    estimates=list(previous.get("estimates",[]))
    known={json.dumps(e,sort_keys=True) for e in estimates}
    for estimate in current.get("estimates",[]):
        if estimate.get("identifiable") is not True:
            continue
        json.dumps(estimate,allow_nan=False)
        Estimate.model_validate(estimate)
        if contracts is not None and (estimate["name"],estimate["unit"],estimate["frame"]) not in {
                (item["name"],item["unit"],item["frame"]) for item in contracts}:
            raise ValueError("Returned estimate quantity/unit/frame differs from the declared measurement contract")
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
    result = dict(current,estimates=estimates,current_estimates=current.get("estimates",[]),trial_history=history,
        probe_progress=progress,
        instance_id=previous.get("instance_id"),
        evidence_source="per_instance_measured_history",previous_evidence_retained=bool(previous.get("estimates")))
    if object_ids is not None:
        result.update(object_ids=list(object_ids), evidence_source="single_world_object_history",
            objects={key: dict(estimates=[e for e in estimates if e.get("object_id") == key]) for key in object_ids})
    return result
