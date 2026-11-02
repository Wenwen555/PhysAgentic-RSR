"""Task-independent projections of measured pose and wrist trajectories."""
import math

from .contracts import finite

FEATURE_SCHEMA = "measured_response_v2"
FEATURE_NAMES = ("object_dx_m", "object_dy_m", "object_dz_m", "max_rotation_rad",
    "max_relative_translation_m", "max_force_change_n", "max_torque_change_nm",
    "max_rise_m", "peak_force_n", "peak_torque_nm", "mean_force_dx_n", "mean_force_dy_n",
    "mean_force_dz_n", "mean_torque_dx_nm", "mean_torque_dy_nm", "mean_torque_dz_nm")
FEATURE_SCALES = (.005, .005, .005, .05, .005, .05, .005, .005, .05, .005, .05, .05, .05, .005, .005, .005)


def response_features(samples):
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
        *[sum(f[j]-force[0][j] for f in force)/len(force) for j in range(3)],
        *[sum(t[j]-torque[0][j] for t in torque)/len(torque) for j in range(3)]]
    return [finite(v,"measured response") for v in values]
