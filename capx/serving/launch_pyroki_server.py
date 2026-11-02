from __future__ import annotations

import asyncio
import functools
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk  # type: ignore
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.transform import Rotation, Slerp

import capx.integrations.motion.pyroki_snippets as pks


def slerp_quaternions(
    q_start: np.ndarray,  # wxyz format
    q_end: np.ndarray,  # wxyz format
    num_steps: int,
) -> np.ndarray:
    """SLERP interpolation between two quaternions.

    Args:
        q_start: Start quaternion in wxyz format
        q_end: End quaternion in wxyz format
        num_steps: Number of interpolation steps

    Returns:
        Array of shape (num_steps, 4) with interpolated quaternions in wxyz format
    """
    # scipy uses xyzw format, so convert
    r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])

    # Create slerp interpolator
    key_rots = Rotation.concatenate([r_start, r_end])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    # Interpolate
    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)

    # Convert back to wxyz format
    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.column_stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]]
    )
    return quats_wxyz


def plan_trajectory_linear_ik(
    robot: pk.Robot,
    target_link_name: str,
    start_pos: np.ndarray,
    start_wxyz: np.ndarray,
    end_pos: np.ndarray,
    end_wxyz: np.ndarray,
    num_waypoints: int = 25,
    use_prev_cfg: bool = True,
    jump_threshold: float = 0.5,
) -> np.ndarray:
    """Plan a trajectory by linear interpolation + IK at each waypoint.

    Args:
        robot: PyRoKi robot model
        target_link_name: Name of the end-effector link
        start_pos: Start position (3,)
        start_wxyz: Start orientation quaternion in wxyz format (4,)
        end_pos: End position (3,)
        end_wxyz: End orientation quaternion in wxyz format (4,)
        num_waypoints: Number of waypoints in the trajectory
        use_prev_cfg: If True, use previous IK solution to bias next solve
        jump_threshold: Max allowed joint change (radians) between waypoints before warning

    Returns:
        Array of shape (num_waypoints, num_joints) with joint configurations
    """
    # Linear interpolation for positions
    positions = np.linspace(start_pos, end_pos, num_waypoints)

    # SLERP for orientations
    orientations = slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)

    # Solve IK for each waypoint
    trajectory = []
    prev_cfg = None
    jump_warnings = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations)):
        if use_prev_cfg and prev_cfg is not None:
            # Use velocity-cost IK to stay close to previous solution
            for _ in range(15):
                cfg = pks.solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                )
                print(
                    f"Error: {np.linalg.norm(cfg - prev_cfg)}",
                    np.allclose(cfg, prev_cfg, atol=1e-3),
                )
                if np.allclose(cfg, prev_cfg, atol=1e-3):
                    break
                else:
                    prev_cfg = cfg
        else:
            cfg = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
            )
        cfg = np.array(cfg)

        # Check for large joint jumps
        if prev_cfg is not None:
            joint_diff = np.abs(cfg - prev_cfg)
            large_jumps = np.where(joint_diff > jump_threshold)[0]
            if len(large_jumps) > 0:
                for joint_idx in large_jumps:
                    jump_warnings.append(
                        f"  Waypoint {i}: joint {joint_idx} jumped {np.degrees(joint_diff[joint_idx]):.1f}° "
                        f"({joint_diff[joint_idx]:.3f} rad)"
                    )

        trajectory.append(cfg)
        prev_cfg = cfg

    # Print warnings summary
    if jump_warnings:
        print(
            f"\nWARNING: {len(jump_warnings)} large joint jump(s) detected (threshold: {np.degrees(jump_threshold):.1f}°):"
        )
        for warning in jump_warnings:
            print(warning)
        print()

    return np.array(trajectory)


# =====================================================
# Logging and FastAPI app
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyroki_server")

app = FastAPI()

_ROBOT = None
_ROBOT_COLL = None
_TARGET_LINK = None


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking CPU-bound function without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# =====================================================
# Pydantic Models
# =====================================================


class IkRequest(BaseModel):
    """IK request with optional prev_cfg for vel-cost IK."""

    target_pose_wxyz_xyz: list[float]  # length 7 (wxyz + xyz)
    prev_cfg: list[float] | None = None  # optional
    #: Extra deterministic restarts to try when the primary solve lands short.
    #: 1 (the default) keeps the historical single-solve behaviour.
    num_starts: int = 1


class IkResponse(BaseModel):
    joint_positions: list[float]


class ObstacleEntry(BaseModel):
    type: str
    point: list[float] | None = None
    normal: list[float] | None = None
    center: list[float] | None = None
    radius: float | None = None
    position: list[float] | None = None
    height: float | None = None
    extent: list[float] | None = None  # for box obstacles


class PlanRequest(BaseModel):
    start_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    end_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    obstacles: list[dict] | None = None  # optional list of ObstacleEntry
    timesteps: int = 20
    dt: float = 0.02


class PlanResponse(BaseModel):
    waypoints: list[list[float]]
    dt: float


# =====================================================
# INTERNAL HELPERS
# =====================================================


def _build_world_coll(obstacles: list[dict[str, Any]] | None):
    if obstacles is None:
        return []
    world = []
    for obj in obstacles:
        t = obj.get("type")
        if t == "halfspace":
            p = np.array(obj["point"], dtype=np.float64)
            n = np.array(obj["normal"], dtype=np.float64)
            world.append(pk.collision.HalfSpace.from_point_and_normal(p, n))
        elif t == "sphere":
            c = np.array(obj["center"], dtype=np.float64)
            r = float(obj["radius"])
            world.append(pk.collision.Sphere.from_center_and_radius(c, np.array([r])))
        elif t == "capsule":
            pos = np.array(obj["position"], dtype=np.float64)
            rad = float(obj["radius"])
            h = float(obj["height"])
            world.append(
                pk.collision.Capsule.from_radius_height(
                    position=pos,
                    radius=np.array([rad]),
                    height=np.array([h]),
                )
            )
        elif t == "box":
            extent = np.array(obj["extent"], dtype=np.float64)
            pos = np.array(obj.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            world.append(
                pk.collision.Box.from_extent(
                    extent=extent,
                    position=pos,
                )
            )
        else:
            logger.warning(f"Unknown obstacle type '{t}', ignoring")
    return world


# =====================================================
# SERVER INIT — Load Pyroki Only Once
# =====================================================
def set_min_distance_from_limits(urdf: yourdfpy.URDF, min_distance_from_limits: float = 0.15) -> yourdfpy.URDF:
    """
    Set the minimum distance from limits for the robot.
    min_distance_from_limits: float in radians
    """
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower = joint.limit.lower + min_distance_from_limits
                joint.limit.upper = joint.limit.upper - min_distance_from_limits
    return urdf


def init_pyroki_server(
    robot_urdf_name: str = "panda_description", target_link_name: str = "panda_hand"
):
    global _ROBOT, _ROBOT_COLL, _TARGET_LINK

    logger.info(f"Loading robot URDF '{robot_urdf_name}' with Pyroki...")

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf = set_min_distance_from_limits(load_robot_description(robot_urdf_name))

    _ROBOT = pk.Robot.from_urdf(urdf)
    # _ROBOT_COLL = pk.collision.RobotCollision.from_urdf(urdf)
    sphere_decomposition = json.load(open(Path(__file__).parent / "assets" / "panda_spheres.json"))
    _ROBOT_COLL = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )
    _TARGET_LINK = target_link_name

    # Warm up the multi-start solver so the first real request does not pay for
    # JAX tracing (a few seconds) while a client is blocking on it.
    n = int(_ROBOT.joints.num_actuated_joints)
    _solve_ik_from(
        _ROBOT,
        jnp.asarray(_ROBOT.links.names.index(target_link_name)),
        jnp.asarray(np.array([1.0, 0.0, 0.0, 0.0])),
        jnp.asarray(np.zeros(3)),
        jnp.asarray(np.zeros(n)),
    ).block_until_ready()

    logger.info("PyRoki loaded and ready!")


# =====================================================
# ROUTES
# =====================================================


#: Fixed seed for the multi-start restarts. Renewing the rng each request keeps
#: the solver deterministic for a fixed ``(target, prev_cfg, num_starts)``.
_MULTISTART_RNG_SEED = 0
#: Position residual (metres) below which a candidate solution is accepted.
_IK_POS_TOL = 0.002
#: Weight converting an orientation error (rad) into a position-equivalent (m),
#: matching the 50:10 positional/orientation weights of the pose cost.
_IK_ORI_WEIGHT = 0.1


@jdc.jit
def _solve_ik_from(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    init_cfg: jax.Array,
) -> jax.Array:
    """Pose + limit IK seeded at ``init_cfg`` (no velocity regularisation).

    ``pks.solve_ik_vel_cost`` adds a cost pulling the solution toward
    ``prev_cfg``; that is good for continuity but can leave a 1-2 cm residual
    when reaching the target requires reconfiguring the wrist. This variant
    keeps only the pose and limit costs, so each restart can settle in its own
    basin.
    """
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(robot, joint_var, weight=100.0),
    ]
    initial_vals = jaxls.VarValues.make((joint_var.with_value(init_cfg),))
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=initial_vals,
        )
    )
    return sol[joint_var]


def _pose_residual(
    q: np.ndarray, target_wxyz: np.ndarray, target_position: np.ndarray, link_index: int
) -> tuple[float, float]:
    """Return ``(combined_score, position_error)`` of config ``q`` vs the target."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    fk = np.asarray(_ROBOT.forward_kinematics(q[None, :])[0][link_index], dtype=np.float64)
    dp = float(np.linalg.norm(fk[4:] - target_position))
    rot_err = Rotation.from_quat(
        np.array([fk[1], fk[2], fk[3], fk[0]])
    ) * Rotation.from_quat(
        np.array([target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]])
    ).inv()
    dtheta = float(np.linalg.norm(rot_err.as_rotvec()))
    return dp + _IK_ORI_WEIGHT * dtheta, dp


def _do_solve_ik(
    target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None, num_starts: int = 1
) -> list[float]:
    """Blocking IK solve (CPU-bound).

    The primary solve is the historical one (cold from zeros, or velocity-cost
    from ``prev_cfg``). If it misses the target by more than ``_IK_POS_TOL``,
    ``num_starts`` extra pose-only restarts are tried and the best solution is
    returned. The pose-only least-squares solver has local minima that wedge the
    wrist against the joint limits (a 1-2 cm position residual despite an exact
    solution existing), which is exactly the failure that makes a 3 cm-wide
    object ungraspable.
    """
    target_wxyz = np.asarray(target_pose_wxyz_xyz[:-3], dtype=np.float64)
    target_position = np.asarray(target_pose_wxyz_xyz[-3:], dtype=np.float64)
    link_index = _ROBOT.links.names.index(_TARGET_LINK)

    if prev_cfg is None:
        q = pks.solve_ik(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_position,
            target_wxyz=target_wxyz,
        )
    else:
        q = pks.solve_ik_vel_cost(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_position,
            target_wxyz=target_wxyz,
            prev_cfg=prev_cfg,
        )
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    score, dp = _pose_residual(q, target_wxyz, target_position, link_index)
    if num_starts <= 1 or dp <= _IK_POS_TOL:
        return list(map(float, q))

    # Restart from deterministic seeds spread over the joint range. ``prev_cfg``
    # (or zeros) stays the first candidate so that a good primary solution is
    # never replaced by a worse restart.
    n_joints = int(_ROBOT.joints.num_actuated_joints)
    lower = np.asarray(_ROBOT.joints.lower_limits, dtype=np.float64).reshape(-1)
    upper = np.asarray(_ROBOT.joints.upper_limits, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(_MULTISTART_RNG_SEED)
    best_q, best_score, best_dp = q, score, dp
    for _ in range(max(1, int(num_starts) - 1)):
        init = lower + rng.random(n_joints) * (upper - lower)
        cand = np.asarray(
            _solve_ik_from(
                _ROBOT,
                jnp.asarray(link_index),
                jnp.asarray(target_wxyz),
                jnp.asarray(target_position),
                jnp.asarray(init),
            ),
            dtype=np.float64,
        ).reshape(-1)
        cand_score, cand_dp = _pose_residual(
            cand, target_wxyz, target_position, link_index
        )
        # Prefer accuracy first; on ties prefer the candidate closest to the
        # previous configuration so the arm does not jump between branches.
        if (cand_dp <= _IK_POS_TOL and best_dp > _IK_POS_TOL) or (
            cand_score < best_score - 1e-6
        ):
            best_q, best_score, best_dp = cand, cand_score, cand_dp
        if best_dp <= _IK_POS_TOL:
            break
    return list(map(float, best_q))


@app.post("/ik", response_model=IkResponse)
async def solve_ik(req: IkRequest):
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")

    target_pose_wxyz_xyz = np.array(req.target_pose_wxyz_xyz, dtype=np.float64)
    prev_cfg = np.array(req.prev_cfg, dtype=np.float64) if req.prev_cfg is not None else None

    try:
        joints = await _run_in_thread(
            _do_solve_ik, target_pose_wxyz_xyz, prev_cfg, int(req.num_starts)
        )
    except Exception as e:
        logger.exception("IK failed")
        raise HTTPException(500, f"IK solve failed: {e}")

    return IkResponse(joint_positions=joints)


def _do_plan_motion(req: PlanRequest) -> PlanResponse:
    """Blocking motion planning (CPU-bound)."""
    start_pose = np.array(req.start_pose_wxyz_xyz, dtype=np.float64)
    end_pose = np.array(req.end_pose_wxyz_xyz, dtype=np.float64)

    start_wxyz = start_pose[:4]
    start_position = start_pose[4:]
    end_wxyz = end_pose[:4]
    end_position = end_pose[4:]

    timesteps = req.timesteps
    dt = req.dt

    sol_traj = plan_trajectory_linear_ik(
        robot=_ROBOT,
        target_link_name=_TARGET_LINK,
        start_pos=start_position,
        start_wxyz=start_wxyz,
        end_pos=end_position,
        end_wxyz=end_wxyz,
        num_waypoints=timesteps,
    )
    sol_traj = np.asarray(sol_traj)

    return PlanResponse(
        waypoints=sol_traj.tolist(),
        dt=float(dt),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_motion(req: PlanRequest):
    if _ROBOT is None or _ROBOT_COLL is None:
        raise HTTPException(503, "Pyroki not initialized")

    try:
        return await _run_in_thread(_do_plan_motion, req)
    except Exception as e:
        logger.exception("Planning failure")
        raise HTTPException(500, f"Motion planning failed: {e}")


# =====================================================
# ENTRYPOINT
# =====================================================


def main(
    robot: str = "panda_description",
    target_link: str = "panda_hand",
    port: int = 8116,
    host: str = "127.0.0.1",
):
    init_pyroki_server(robot_urdf_name=robot, target_link_name=target_link)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()