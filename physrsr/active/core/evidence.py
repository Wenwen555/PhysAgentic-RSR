"""Compact views of archived measured trajectories."""


def available_evidence_ids(context):
    """Only supplied records establish evidence; reviews/generated claims do not."""
    ids=set()
    if context.get("goal") or context.get("task"): ids.add("task_goal")
    if context.get("capabilities"): ids.add("robot_capabilities")
    for name in ("initial_policy","probe"):
        if context.get(name): ids.add(context[name].get("evidence_id",name))
    for instance in context.get("instances",[]):
        ids.update(instance.get("available_evidence_ids",[]))
    for name in ("initial_evidence","prior_identification_evidence","fit_evidence","world_probe_evidence"):
        for record in context.get(name,[]):
            if record.get("evidence_id"): ids.add(record["evidence_id"])
            if record.get("summary",{}).get("evidence_id"): ids.add(record["summary"]["evidence_id"])
            for samples in ("trajectory","observations","labeled_observations"):
                ids.update(sample["evidence_id"] for sample in record.get(samples,[]) if sample.get("evidence_id"))
    return ids


def require_evidence(citations, available, *, exact=False):
    if (not citations or len(citations)!=len(set(citations)) or not set(citations)<=set(available)
            or exact and set(citations)!=set(available)):
        raise ValueError("Response cites missing, repeated or unavailable evidence IDs")

def evidence_samples(samples,limit=10):
    """Keep endpoints, elapsed phases and load extrema; archive stays complete."""
    if len(samples)<=limit:
        return samples
    indices={0,len(samples)-1,*[i*(len(samples)-1)//(limit-3) for i in range(limit-2)]}
    for channel in ("force_n","torque_nm"):
        indices.add(max(range(len(samples)),key=lambda i:sum(
            x*x for x in samples[i]["observation"]["wrench"][channel])))
    return [samples[i] for i in sorted(indices)]


def prediction_feedback(rows):
    """Pure compact view of ensemble scores and measured failure trajectories."""
    scores=[{k:v for k,v in row.items() if k!="feedback"} for row in rows]
    trajectories=[]
    for row in sorted(rows,key=lambda r:(r["success_rate"],r["margin"]))[:4]:
        for outcome in row.get("feedback",[])[:2]:
            trajectories.append(dict(model_id=row["model_id"],policy_sha256=row["policy_sha256"],
                status=outcome.get("status"),reason=outcome.get("reason"),
                observations=evidence_samples(outcome.get("observations",[]))))
    return dict(model_policy_scores=scores,failure_trajectories=trajectories,
        sampling="Selected API boundaries and load extrema; complete responses remain in simulation artifacts")
