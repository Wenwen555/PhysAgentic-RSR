from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    open_gripper as _open_gripper,
)
from capx.integrations.motion.pyroki import init_pyroki


# ------------------------------- Control API ------------------------------
class FrankaControlPrivilegedApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)
    #: Deterministic IK restarts requested when the primary solve lands short
    #: (see ``_do_solve_ik`` on the IK server).
    _IK_NUM_STARTS = 8
    #: Peak joint speed (rad/s) the position controller reaches when ``speed``
    #: is left at its default. Measured on the sim Panda (~2.5 rad/s during a
    #: 0.1 m lift); used only to give the dimensionless ``speed`` argument a
    #: physical meaning. ``speed >= 1`` means "no limit".
    _REFERENCE_JOINT_SPEED = 2.5

    @classmethod
    def _joint_speed_limit(cls, speed: float | None) -> float | None:
        """Map a dimensionless ``speed`` (1.0 = controller max) to rad/s."""
        if speed is None or float(speed) >= 1.0:
            return None
        return max(float(speed), 1e-3) * cls._REFERENCE_JOINT_SPEED

    def __init__(self, env: BaseEnv, multi_turn: bool = False) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        # from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        # from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki(num_starts=self._IK_NUM_STARTS)
        self.cfg = None
        self.multi_turn = multi_turn
        self.motion_joint_tolerance_rad = None
        self.motion_tcp_tolerance_m = self._TCP_CORRECTION_TOL

    def configure_motion(self, *, joint_tolerance_rad: float, tcp_tolerance_m: float) -> None:
        """Explicit precision profile shared by simulated and measured arms.

        The joint controller must not declare convergence at a coarse joint
        threshold while Cartesian feedback still needs a smaller correction.
        Existing callers retain the environment's default joint tolerance.
        """
        if (not np.isfinite(joint_tolerance_rad) or not 0 < joint_tolerance_rad <= .02
                or not np.isfinite(tcp_tolerance_m) or not 0 < tcp_tolerance_m <= .01):
            raise ValueError("Invalid motion convergence profile")
        self.motion_joint_tolerance_rad = float(joint_tolerance_rad)
        self.motion_tcp_tolerance_m = float(tcp_tolerance_m)

    def functions(self) -> dict[str, Any]:
        base_functions = {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            # "home_pose": self.home_pose,
        }
        # if self.multi_turn:
        #     base_functions["breakpoint_code_block"] = self.breakpoint_code_block
        return base_functions

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) object extent in meters of x, y, z axes respectively in the world frame (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        obs = self._env.get_observation()

        if (
            "red" in object_name and "cube" in object_name
        ):  # TODO: Slightly problematic that these are hardcoded language descriptions
            # Could just expose a low level obs dict item with the same object name from the low level env
            return (
                obs["cube_poses"]["primary"][:3],
                obs["cube_poses"]["primary"][3:],
                np.array([0.05, 0.05, 0.05]),
            )
        elif "green" in object_name and "cube" in object_name:
            return (
                obs["cube_poses"]["secondary"][:3],
                obs["cube_poses"]["secondary"][3:],
                np.array([0.05, 0.05, 0.05]),
            )
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()

        if "red" in object_name and "cube" in object_name:
            return obs["cube_poses"]["primary"][:3], np.array([0, 0, 1, 0])
        elif "green" in object_name and "cube" in object_name:
            return obs["cube_poses"]["secondary"][:3], np.array([0, 0, 1, 0])
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
        speed: float | None = None,
    ) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
            speed: (float | None) Fraction of the controller's maximum joint speed
                to command, in (0, 1]. ``None`` or ``>= 1`` moves as fast as the
                position controller allows. A lower value slows the whole move
                down, which matters physically: fast transport of a long object
                adds inertial torque about the grasp point. Implemented in the
                low-level env (sim: commanded-joint rate limit; real: rate-limited
                published setpoints), so the same program runs on hardware.
        Returns:
            None
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)
        rot = SciRotation.from_quat(
            np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        )
        max_joint_speed = self._joint_speed_limit(speed)

        if z_approach != 0.0:
            approach_tcp = pos + rot.apply(np.array([0, 0, -z_approach]))
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
            self._solve_and_move(
                quat_wxyz, z_offset_pos, approach_tcp, max_joint_speed=max_joint_speed
            )

        self._solve_and_move(
            quat_wxyz, offset_pos, pos, max_joint_speed=max_joint_speed
        )

    #: Cartesian feedback loop for :meth:`_solve_and_move`. Joint-space tracking
    #: has a gravity-induced steady-state error (~1.5 cm) and the IK model's
    #: hand->TCP convention differs from the simulator's by ~1 cm, so an
    #: open-loop joint target systematically lands short. Two extra IK iterations
    #: are usually enough to bring the measured TCP onto the requested point.
    _TCP_CORRECTION_TOL = 0.003  # metres
    _TCP_CORRECTION_ITERS = 3

    def _solve_and_move(
        self,
        quat_wxyz: np.ndarray,
        hand_target: np.ndarray,
        tcp_target: np.ndarray,
        max_joint_speed: float | None = None,
    ) -> None:
        """Solve IK and move the *measured* TCP onto ``tcp_target``.

        Closes the loop on the achieved end-effector pose instead of trusting the
        joint target: measure where the arm actually stopped, shift the IK target
        by the residual, and re-solve. This is what makes a 3 cm object graspable
        with a 3 cm-wide parallel jaw. The measurement source is
        ``env.get_eef_pose()`` (simulated TCP in the sim, measured FK of the
        encoder joints on the real arm), so the same code path transfers to
        hardware.

        Args:
            quat_wxyz: (4,) target orientation.
            hand_target: (3,) position to feed the IK (the ``panda_hand`` link).
            tcp_target: (3,) position the *TCP* should reach.
            max_joint_speed: Optional rad/s cap on the commanded joint speed
                (see :meth:`goto_pose`'s ``speed``).
        """
        target = np.asarray(hand_target, dtype=np.float64).reshape(3)
        tcp_target = np.asarray(tcp_target, dtype=np.float64).reshape(3)
        for _ in range(self._TCP_CORRECTION_ITERS):
            if self.cfg is None:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target]),
                )
            else:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target]),
                    prev_cfg=self.cfg,
                )
            joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            motion_options = dict(max_joint_speed=max_joint_speed)
            joint_tolerance = getattr(self, "motion_joint_tolerance_rad", None)
            if joint_tolerance is not None:
                motion_options["tolerance"] = joint_tolerance
            self._env.move_to_joints_blocking(joints, **motion_options)

            measure = getattr(self._env, "get_eef_pose", None)
            if measure is None:
                return  # no feedback available: best effort (open loop)
            _, achieved = measure()
            residual = tcp_target - np.asarray(achieved, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(residual)) <= getattr(self, "motion_tcp_tolerance_m", self._TCP_CORRECTION_TOL):
                return
            # Shift the IK target by the residual. Because the IK target is the
            # hand and the residual is measured at the TCP, this only converges
            # to the desired TCP if the hand->TCP mapping is *constant*: it is
            # (a rigid offset), which is exactly why the loop works.
            target = target + residual

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        _open_gripper(self._env, steps=40)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        _close_gripper(self._env, steps=60)

    def home_pose(self) -> None:
        """
        Move the robot to a safe home pose.
        Args:
            None
        Returns:
            None
        """

        # joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
        joints = np.array(
            [
                -2.95353726e-02,
                1.69197371e-01,
                2.39244731e-03,
                -2.64089311e00,
                -2.01237851e-03,
                2.94565778e00,
                8.31390616e-01,
            ]
        )
        self._env.move_to_joints_blocking(joints)
