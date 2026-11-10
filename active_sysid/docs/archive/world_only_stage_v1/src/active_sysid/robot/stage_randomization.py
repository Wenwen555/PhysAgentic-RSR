"""Seeded in-memory World parameter draws for the visual_randomized stage.

Robot kinematics/servo gains, camera poses, textures, and source XML stay fixed.
"""
from __future__ import annotations

import numpy as np
import mujoco


def apply_world_randomization(backend, seed: int, ranges: dict) -> dict:
    """Return the private evaluator draw; never expose it through observe()."""
    rng = np.random.default_rng(seed)
    model = backend.model
    draw = {name: float(rng.uniform(*bounds)) for name, bounds in sorted(ranges.items())}
    door = backend.door_jid
    handle = backend.handle_jid
    if 'door_damping' in draw:
        model.dof_damping[model.jnt_dofadr[door]] *= draw['door_damping']
    if 'door_frictionloss' in draw:
        model.dof_frictionloss[model.jnt_dofadr[door]] *= draw['door_frictionloss']
    if 'handle_damping' in draw:
        model.dof_damping[model.jnt_dofadr[handle]] *= draw['handle_damping']
    if 'handle_stiffness' in draw:
        model.jnt_stiffness[handle] *= draw['handle_stiffness']
    if 'handle_contact_friction' in draw:
        model.geom_friction[backend.handle_geom, 0] *= draw['handle_contact_friction']
    if 'illumination' in draw:
        model.light_diffuse[:] = np.clip(model.light_diffuse * draw['illumination'], 0, 1)
    mujoco.mj_forward(model, backend.data)
    return draw
