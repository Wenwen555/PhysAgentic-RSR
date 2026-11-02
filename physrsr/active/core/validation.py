"""Measured intervention coverage, bound to a candidate and physical support.

Coverage establishes executed inputs and sensor excitation, not causal truth or
calibrated confidence. The predictive discrepancy test remains independently required.
"""
import math

from .bundle import content_hash
from .scene import objects


def local_offset(observed,key):
    obj=objects(observed)[key]
    delta=[a-b for a,b in zip(observed["tcp"]["position"],obj["position"])]
    return [sum(a*b for a,b in zip(delta,axis)) for axis in obj["axes"]]


def closest_object(observed):
    def distance(key):
        obj=objects(observed)[key]
        return sum(max(0.,abs(x)-size/2)**2 for x,size in zip(local_offset(observed,key),obj["dimensions_m"]))
    return min(sorted(objects(observed)),key=distance)


def intervention_trace(samples):
    if not samples:raise ValueError("Intervention coverage requires actual observations")
    first=samples[0]["observation"]
    def metrics():return dict(loaded_duration_s=0.,tcp_translation_m=0.,object_rise_m=0.,
        object_translation_m=0.,object_rotation_rad=0.,force_change_n=0.)
    rows={key:dict(closes=0,settings=[],executed_settings=[],**metrics()) for key in objects(first)}
    active=None;setting=None;anchor=None;previous=first["time_s"];previous_observation=first;actions=[]
    for sample in samples:
        obs=sample["observation"];now=obs["time_s"]
        if now<previous:raise ValueError("Intervention timestamps are not monotonic")
        if active is not None:
            initial_obj=setting["reference_object"];current_obj=objects(obs)[active]
            qa,qb=initial_obj["quaternion_wxyz"],current_obj["quaternion_wxyz"]
            cosine=abs(sum(a*b for a,b in zip(qa,qb)))/math.sqrt(sum(a*a for a in qa)*sum(b*b for b in qb))
            values=dict(tcp_translation_m=math.dist(obs["tcp"]["position"],anchor),
                object_rise_m=current_obj["position"][2]-initial_obj["position"][2],
                object_translation_m=math.dist(current_obj["position"],initial_obj["position"]),
                object_rotation_rad=2*math.acos(min(1.,cosine)),
                force_change_n=math.dist(obs["wrench"]["force_n"],setting["reference_force_n"]))
            for dest in (rows[active],setting):
                dest["loaded_duration_s"]+=now-previous
                for field,value in values.items():dest[field]=max(dest[field],value)
        event=sample.get("event")
        if event in ("goto_pose","close_gripper","open_gripper","hold"):
            actions.append(dict(event=event,params=sample.get("params",{})))
        if event=="close_gripper":
            active=closest_object(obs);anchor=obs["tcp"]["position"]
            offset=local_offset(obs,active);obj=objects(obs)[active]
            gap=math.sqrt(sum(max(0.,abs(x)-size/2)**2 for x,size in zip(offset,obj["dimensions_m"])))
            if gap>.02:
                active=None
                previous=now
                previous_observation=obs
                continue
            rows[active]["closes"]+=1
            rows[active]["settings"].append(offset)
            baseline=sample.get("action_initial",previous_observation)
            setting=dict(tcp_offset_body_m=offset,reference_object=objects(baseline)[active],
                reference_force_n=baseline["wrench"]["force_n"],**metrics())
            rows[active]["executed_settings"].append(setting)
        elif event=="open_gripper":active=None
        previous=now
        previous_observation=obs
    return dict(objects=rows,action_fingerprint=content_hash(actions))


def task_coverage(rows, candidate_sha, ids):
    required={key:dict(min_duration_s=.05,min_translation_m=0.,min_object_translation_m=0.,
        min_rotation_rad=0.,min_rise_m=0.,min_force_change_n=0.) for key in ids}
    for row in rows:
        if row["policy_sha256"]!=candidate_sha:continue
        for outcome in row.get("feedback",[]):
            observed=intervention_trace(outcome["observations"])["objects"]
            for key,values in observed.items():
                if key not in required:continue
                dest=required[key]
                for field,measured in (("min_duration_s","loaded_duration_s"),("min_translation_m","tcp_translation_m"),
                        ("min_rise_m","object_rise_m"),("min_force_change_n","force_change_n"),
                        ("min_object_translation_m","object_translation_m"),("min_rotation_rad","object_rotation_rad")):
                    dest[field]=max(dest[field],values[measured])
    return required


def validate_contract(contract, context):
    if contract is None:raise ValueError("Independent validation requires an intervention contract")
    required=context.get("validation_requirements")
    if not required or contract["candidate_sha256"]!=context.get("candidate_sha256"):
        raise ValueError("Validation must bind the current candidate and its operation envelope")
    if sorted(contract["object_ids"])!=sorted(required) or len(set(contract["object_ids"]))!=len(required):
        raise ValueError("Validation must cover every target object")
    if set(contract["hypothesis_ids"])!={m["id"] for m in context["models"] if m["weight"]>0}:
        raise ValueError("Validation must identify the effective physical support")
    if {item["object_id"] for item in contract["settings"]}!=set(required):
        raise ValueError("Declare executed settings for every task object")
    if not contract.get("alternative_explanation","").strip() or not contract.get("task_operation","").strip():
        raise ValueError("Validation requires a competing explanation and task operation")
    for index,setting in enumerate(contract["settings"]):
        for other in contract["settings"][:index]:
            if setting["object_id"]==other["object_id"] and math.dist(setting["tcp_offset_body_m"],other["tcp_offset_body_m"])<=setting["setting_tolerance_m"]+other["setting_tolerance_m"]:
                raise ValueError("Declared validation settings must be distinguishable")
    for item in contract["settings"]:
        if any(item[field]<value for field,value in required[item["object_id"]].items()):
            raise ValueError("Validation excitation cannot be weaker than the candidate operation requirements")
    if contract["max_prediction_error"]>context["max_prediction_error"]:
        raise ValueError("Validation cannot weaken the host prediction acceptance threshold")


def check_executed(contract, samples, *, earlier=()):
    trace=intervention_trace(samples);errors=[]
    for segment in earlier:
        for key,values in segment["objects"].items():
            dest=trace["objects"][key]
            for field in ("closes","loaded_duration_s"):dest[field]+=values[field]
            dest["settings"]+=values["settings"]
            dest["executed_settings"]+=values["executed_settings"]
            for field in ("tcp_translation_m","object_rise_m","force_change_n","object_translation_m","object_rotation_rad"):dest[field]=max(dest[field],values[field])
    for setting in contract["settings"]:
        actual=trace["objects"].get(setting["object_id"])
        if actual is None or not actual["closes"]:
            errors.append(f"{setting['object_id']}: no measured contact intervention")
            continue
        matching=[observed for observed in actual["executed_settings"]
            if math.dist(observed["tcp_offset_body_m"],setting["tcp_offset_body_m"])<=setting["setting_tolerance_m"]]
        if not matching:
            errors.append(f"{setting['object_id']}: declared contact setting was not executed")
        thresholds=(("min_duration_s","loaded_duration_s"),("min_translation_m","tcp_translation_m"),
            ("min_rise_m","object_rise_m"),("min_force_change_n","force_change_n"),
            ("min_object_translation_m","object_translation_m"),("min_rotation_rad","object_rotation_rad"))
        if matching and not any(all(item[measured]+1e-9>=setting.get(field,0.) for field,measured in thresholds) for item in matching):
            errors.append(f"{setting['object_id']}: insufficient excitation at the declared setting")
    return dict(passed=not errors,errors=errors,trace=trace)
