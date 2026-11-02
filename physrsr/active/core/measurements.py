"""Task-independent projections of measured pose and wrist trajectories."""
import math

from .contracts import finite
from .scene import object_ids, check_objects

FEATURE_SCHEMA = "measured_response_v3"
SCENE_FEATURE_SCHEMA = "measured_scene_response_v2"
FEATURE_NAMES = ("object_dx_m", "object_dy_m", "object_dz_m", "max_rotation_rad",
    "max_relative_translation_m", "max_force_change_n", "max_torque_change_nm",
    "max_rise_m", "peak_force_n", "peak_torque_nm", "mean_force_dx_n", "mean_force_dy_n",
    "mean_force_dz_n", "mean_torque_dx_nm", "mean_torque_dy_nm", "mean_torque_dz_nm")
FEATURE_SCALES = (.005, .005, .005, .05, .005, .05, .005, .005, .05, .005, .05, .05, .05, .005, .005, .005)


def response_schema(observed):
    return SCENE_FEATURE_SCHEMA if "objects" in observed else FEATURE_SCHEMA


def response_features(samples):
    """Object pose blocks plus ONE shared wrist block, never duplicated sensors."""
    if not samples:
        raise ValueError("Measured prediction requires a trajectory")
    first = samples[0]["observation"]
    if "objects" not in first:
        if any("objects" in sample["observation"] for sample in samples):
            raise ValueError("Response changed from an object to a scene")
        return _object_features(samples)
    ids = object_ids(first)
    for sample in samples:
        if "objects" not in sample["observation"]:
            raise ValueError("Scene response lost its object tracks")
        check_objects(sample["observation"], ids)
    blocks=[_object_features([dict(observation=dict(sample["observation"],
        object=sample["observation"]["objects"][key])) for sample in samples]) for key in ids]
    return [block[i] for block in blocks for i in OBJECT_CHANNELS]+[blocks[0][i] for i in SHARED_CHANNELS]


OBJECT_CHANNELS = (0,1,2,3,4,7)
SHARED_CHANNELS = (5,6,8,9,10,11,12,13,14,15)


def scene_scales(scales, count):
    if len(scales)!=len(FEATURE_NAMES):raise ValueError("Feature scales must contain 16 base channels")
    return tuple(scales[i] for _ in range(count) for i in OBJECT_CHANNELS)+tuple(scales[i] for i in SHARED_CHANNELS)


def time_mean(observations, channel, axis):
    # Keep one value per physical instant. Extra observe() RPCs cannot increase
    # dwell time or evidence strength. At a boundary the last measured state wins.
    by_time={}
    previous=-float("inf")
    for observed in observations:
        t=finite(observed["time_s"],"trajectory time")
        if t<previous:raise ValueError("Trajectory timestamps must be nondecreasing")
        previous=t
        by_time[t]=finite(observed["wrench"][channel][axis])
    points=list(by_time.items())
    duration=points[-1][0]-points[0][0]
    if duration==0:return points[-1][1]
    return sum((t1-t0)*(v0+v1)*.5 for (t0,v0),(t1,v1) in zip(points,points[1:]))/duration


def _object_features(samples):
    if not samples:
        raise ValueError("Measured prediction requires a trajectory")
    observations = [s["observation"] for s in samples]
    first, last = observations[0], observations[-1]
    origin = first["object"]["position"]
    initial_relative = [a-b for a,b in zip(origin, first["tcp"]["position"])]
    q = first["object"]["quaternion_wxyz"]
    def angle(obs):
        other = obs["object"]["quaternion_wxyz"]
        norm = math.sqrt(sum(x*x for x in q)*sum(x*x for x in other))
        if norm == 0:
            raise ValueError("Invalid measured quaternion")
        return 2*math.acos(min(1., abs(sum(a*b for a,b in zip(q, other)))/norm))
    # A response observable, not a contact or slip oracle.
    relative = max(math.dist([a-b for a,b in zip(o["object"]["position"],o["tcp"]["position"])],
        initial_relative) for o in observations)
    wrist = [o["wrench"] for o in observations]
    if any(w["valid"] is not True for w in wrist):
        raise ValueError("Measured prediction requires valid wrist channels")
    force, torque = [w["force_n"] for w in wrist], [w["torque_nm"] for w in wrist]
    values = [*[b-a for a,b in zip(origin,last["object"]["position"])],max(map(angle,observations)),relative,
        max(math.dist(f,force[0]) for f in force),max(math.dist(t,torque[0]) for t in torque),
        max(o["object"]["position"][2]-origin[2] for o in observations),
        max(math.sqrt(sum(x*x for x in f)) for f in force),
        max(math.sqrt(sum(x*x for x in t)) for t in torque),
        *[time_mean(observations,"force_n",j)-force[0][j] for j in range(3)],
        *[time_mean(observations,"torque_nm",j)-torque[0][j] for j in range(3)]]
    return [finite(v,"measured response") for v in values]
