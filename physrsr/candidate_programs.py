"""Fixed candidate CaP programs (P1-P6) for the physics-CaP experiment.

Each candidate varies *where* along the rod it grasps (geometric centre, CoM,
near left/right end) and *how* it transports (direct lift vs small steps). They
are written against the CaP-X control API (``goto_pose`` / ``open_gripper`` /
``close_gripper``), i.e. exactly the code an agent would generate, so they
hot-swap to the real arm unchanged (see AGENTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from physrsr.physics_spec import PhysicsSpec

#: Gripper orientation (wxyz) pointing straight down — same as the cube-lift oracle.
DOWN_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])


from physrsr.geometry import rod_pose, down_grasp_quat


def grasp_point(env, physics: PhysicsSpec, where: str) -> np.ndarray:
    """Resolve a symbolic grasp location to a position in the robot base frame.

    ``where`` is one of ``"center"``, ``"com"``, ``"left"``, ``"right"``.
    """
    centre, rod_x = rod_pose(env)
    half = physics.rod_half_length
    offsets = {
        "center": 0.0,
        "com": physics.com_offset,
        "left": -0.7 * half,
        "right": 0.7 * half,
    }
    return centre + rod_x * offsets[where]


@dataclass(frozen=True)
class CandidateProgram:
    """A single fixed program. ``where``/``transport`` define its behaviour."""

    id: str
    where: str  # "center" | "com" | "left" | "right"
    transport: str  # "direct" | "small_step"
    description: str
    #: Fraction of the controller's max joint speed for the transport moves
    #: (``None`` = as fast as the controller allows). Slower transport means
    #: smaller inertial torque about the grasp point, which is what decides
    #: whether a long rod stays level when grasped away from its CoM.
    speed: float | None = None

    def run(self, env, api, physics: PhysicsSpec) -> None:
        gp = grasp_point(env, physics, self.where)
        quat = down_grasp_quat(env)
        api.open_gripper()
        api.goto_pose(gp, quat, z_approach=0.1)
        api.goto_pose(gp, quat)
        api.close_gripper()
        if self.transport == "direct":
            api.goto_pose(gp + np.array([0.0, 0.0, 0.1]), quat, speed=self.speed)
        else:
            for dz in (0.03, 0.06, 0.1):
                api.goto_pose(gp + np.array([0.0, 0.0, dz]), quat, speed=self.speed)


CANDIDATES: dict[str, CandidateProgram] = {
    "P1": CandidateProgram("P1", "center", "direct", "grasp geometric centre, direct lift"),
    "P2": CandidateProgram("P2", "com", "direct", "grasp near CoM, direct lift"),
    "P3": CandidateProgram("P3", "com", "small_step", "grasp near CoM, small-step lift"),
    "P4": CandidateProgram("P4", "center", "small_step", "grasp centre, small-step lift"),
    "P5": CandidateProgram("P5", "left", "direct", "grasp near left end, then lift"),
    "P6": CandidateProgram("P6", "right", "direct", "grasp near right end, then lift"),
    # Speed axis: identical grasps to P1/P2, but the transport is rate-limited to
    # 35% of the controller's joint speed. Same code, one extra argument.
    "P7": CandidateProgram(
        "P7", "center", "direct", "grasp centre, slow lift", speed=0.35
    ),
    "P8": CandidateProgram(
        "P8", "com", "direct", "grasp near CoM, slow lift", speed=0.35
    ),
}
