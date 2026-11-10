"""Import an existing M0/M1/M2 mechanism into the common robot scene.

Only object-joint parameters and the door body's local mass properties are
copied. Robot parameters, contact geometry and the new handle's mass properties
stay unchanged. In particular, M2's ``I_d`` describes the door leaf, not the
combined inertia of the leaf and the new handle.

Call this before episode initialization. Afterward the caller creates MjData,
runs ``mj_setConst(model, data)``, resets it and applies the robot home pose.
The returned latch configuration belongs to the backend, not to the policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def apply_mechanism(
    model: Any,
    mechanism_mjcf: str | Path,
    default_unlock_threshold: float = 0.556,
) -> dict[str, Any]:
    """Apply a source MJCF plus its optional ``.repair.json`` sidecar.

    Both scenes use ``door_hinge``, ``handle_hinge`` and the ``door`` body.
    The door body's local frame is shared; the handle's joint axis and contact
    geometry can differ. A source without ``latch_weld`` has no latch even if
    the common target scene contains that equality constraint.
    """
    import mujoco as mj

    source_path = Path(mechanism_mjcf).resolve()
    source = mj.MjModel.from_xml_path(str(source_path))
    sidecar_path = source_path.with_suffix(source_path.suffix + ".repair.json")
    sidecar_bytes = sidecar_path.read_bytes() if sidecar_path.exists() else None
    repair = json.loads(sidecar_bytes) if sidecar_bytes is not None else {}

    parameters: dict[str, Any] = {}
    for name in ("door_hinge", "handle_hinge"):
        src = source.joint(name)
        dst = model.joint(name)
        src_dof = int(src.dofadr[0])
        dst_dof = int(dst.dofadr[0])
        src_qpos = int(src.qposadr[0])
        dst_qpos = int(dst.qposadr[0])

        for field in ("dof_damping", "dof_frictionloss", "dof_armature"):
            getattr(model, field)[dst_dof] = getattr(source, field)[src_dof]
        for field in ("jnt_stiffness", "jnt_range", "jnt_limited"):
            getattr(model, field)[dst.id] = getattr(source, field)[src.id]
        model.qpos_spring[dst_qpos] = source.qpos_spring[src_qpos]

        parameters[name] = {
            "damping": float(model.dof_damping[dst_dof]),
            "frictionloss": float(model.dof_frictionloss[dst_dof]),
            "armature": float(model.dof_armature[dst_dof]),
            "stiffness": float(model.jnt_stiffness[dst.id]),
            "range": model.jnt_range[dst.id].tolist(),
            "limited": bool(model.jnt_limited[dst.id]),
            "springref": float(model.qpos_spring[dst_qpos]),
        }

    # Copy the compiled, body-local properties rather than changing density:
    # MjModel has no geom-density field, and inertia includes the COM frame.
    src_body = source.body("door").id
    dst_body = model.body("door").id
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        getattr(model, field)[dst_body] = getattr(source, field)[src_body]
    parameters["door_body"] = {
        "mass_kg": float(model.body_mass[dst_body]),
        "principal_inertia_kg_m2": model.body_inertia[dst_body].tolist(),
        "inertial_position_m": model.body_ipos[dst_body].tolist(),
        "inertial_quaternion_wxyz": model.body_iquat[dst_body].tolist(),
    }

    latch = repair.get("latch", {})
    equality_id = mj.mj_name2id(source, mj.mjtObj.mjOBJ_EQUALITY, "latch_weld")
    enabled = equality_id >= 0 and source.eq_type[equality_id] == mj.mjtEq.mjEQ_WELD
    return {
        "source_mjcf": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "sidecar_sha256": (
            hashlib.sha256(sidecar_bytes).hexdigest()
            if sidecar_bytes is not None else None
        ),
        "latch": {
            "enabled": bool(enabled),
            "unlock_threshold": float(latch.get("unlock_threshold", default_unlock_threshold)),
            "relock_door_eps": float(latch.get("relock_door_eps", 0.05)),
            "latch_weld": "latch_weld",
        },
        "parameters": parameters,
    }
