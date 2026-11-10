"""DLS inverse kinematics, smooth Cartesian motion and bounded joint impedance.

IK uses a separate MjData. Only actuator controls touch the live simulation;
no object qpos, object torque, mocap or grasp weld is used by this controller.
"""
from __future__ import annotations
import numpy as np
import mujoco as mj


class ExecutionError(RuntimeError):
    """A robot action could not meet its motion or measured-contact limit."""


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("rotation axis must be nonzero")
    x, y, z = axis / length
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def rotation_vector(rotation):
    quat = np.zeros(4)
    mj.mju_mat2Quat(quat, np.asarray(rotation).ravel())
    if quat[0] < 0:
        quat *= -1
    length = np.linalg.norm(quat[1:])
    if length < 1e-10:
        return 2 * quat[1:]
    return 2 * np.arctan2(length, quat[0]) * quat[1:] / length


def quat_to_matrix(quat):
    quat = np.asarray(quat, dtype=float)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-12:
        raise ValueError("orientation must be a finite nonzero wxyz quaternion")
    out = np.zeros(9)
    mj.mju_quat2Mat(out, quat / np.linalg.norm(quat))
    return out.reshape(3, 3)


def matrix_to_quat(rotation):
    out = np.zeros(4)
    mj.mju_mat2Quat(out, np.asarray(rotation).ravel())
    return out


class RobotController:
    def __init__(self, backend, *, max_steps=30000, contact_force_limit_n=80.0,
                 unexpected_contact_limit_n=5.0):
        self.backend = backend
        self.model = backend.model
        self.ik_data = mj.MjData(self.model)
        self.q_target = backend.data.qpos[backend.arm_qpos].copy()
        self.width = 0.08
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.max_steps = int(max_steps)
        # These are experiment limits on measured contacts, not damage bounds.
        self.contact_force_limit_n = float(contact_force_limit_n)
        self.unexpected_contact_limit_n = float(unexpected_contact_limit_n)
        self.last_target = backend.tcp_pose()

    def solve_ik(self, position, rotation, seed=None):
        b = self.backend
        q = self.q_target.copy() if seed is None else np.asarray(seed, dtype=float).copy()
        self.ik_data.qpos[:] = b.data.qpos
        limits = self.model.jnt_range[b.arm_jids]
        for _ in range(80):
            self.ik_data.qpos[b.arm_qpos] = q
            mj.mj_kinematics(self.model, self.ik_data)
            mj.mj_comPos(self.model, self.ik_data)
            current_rotation = self.ik_data.site_xmat[b.tcp_id].reshape(3, 3)
            error = np.r_[np.asarray(position) - self.ik_data.site_xpos[b.tcp_id],
                          rotation_vector(np.asarray(rotation) @ current_rotation.T)]
            if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-4:
                return q
            mj.mj_jacSite(self.model, self.ik_data, self.jacp, self.jacr, b.tcp_id)
            jac = np.vstack((self.jacp[:, b.arm_dofs], self.jacr[:, b.arm_dofs]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.002**2 * np.eye(6), error)
            q = np.clip(q + delta * min(1, 0.12 / max(np.linalg.norm(delta), 1e-12)),
                        limits[:, 0] + 0.02, limits[:, 1] - 0.02)
        raise ExecutionError(f"IK target unreachable: position error={np.linalg.norm(error[:3]):.4f} m, "
                             f"orientation error={np.linalg.norm(error[3:]):.4f} rad")

    def tick(self):
        b = self.backend
        if len(b.samples) >= self.max_steps:
            raise ExecutionError("robot simulation step budget exceeded")
        b.command(self.q_target, self.width)
        sample = b.step()
        if sample["total_handle_contact_force_n"] > self.contact_force_limit_n:
            raise ExecutionError("measured aggregate handle contact force exceeded experiment limit")
        if max(sample["unexpected_contact_force_n"], sample["self_contact_force_n"]) > self.unexpected_contact_limit_n:
            raise ExecutionError("unexpected robot/environment or robot/self contact exceeded experiment limit")
        return sample

    def _steps(self, seconds):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("duration_s must be positive")
        steps = int(np.ceil(seconds / self.backend.dt))
        if len(self.backend.samples) + steps > self.max_steps:
            raise ExecutionError("action would exceed robot simulation step budget")
        return steps

    def hold(self, seconds):
        for _ in range(self._steps(seconds)):
            self.tick()
        return self.backend.sample()

    def move_joints(self, positions, duration=2.0):
        """Smooth joint-space transit; only arm actuators are commanded."""
        target = np.asarray(positions, dtype=float)
        if target.shape != self.q_target.shape or not np.all(np.isfinite(target)):
            raise ValueError("joint target must be seven finite joint positions")
        limits = self.model.jnt_range[self.backend.arm_jids]
        if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
            raise ExecutionError("joint target exceeds the model joint limits")
        start = self.q_target.copy()
        steps = self._steps(duration)
        for step in range(steps):
            u = (step + 1) / steps
            blend = u**3 * (10 - 15*u + 6*u*u)
            self.q_target = start + blend * (target - start)
            self.tick()
        self.last_target = self.backend.tcp_pose()
        return self.backend.sample()

    def _follow(self, trajectory, duration):
        steps = self._steps(duration)
        for step in range(steps):
            u = (step + 1) / steps
            blend = u**3 * (10 - 15*u + 6*u*u)
            target, rotation = trajectory(blend)
            if step % 5 == 0 or step == steps - 1:
                self.q_target = self.solve_ik(target, rotation)
            self.tick()
        self.last_target = np.array(target), np.array(rotation)
        return self.backend.sample()

    def move_ee(self, position, rotation, duration=1.0):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_m must be a finite three-vector")
        start_pos, start_rot = self.backend.tcp_pose()
        rv = rotation_vector(np.asarray(rotation) @ start_rot.T)
        angle = np.linalg.norm(rv)
        return self._follow(lambda t: (
            start_pos + t * (position - start_pos),
            axis_rotation(rv / angle, t * angle) @ start_rot if angle > 1e-9 else start_rot,
        ), duration)

    def move_arc(self, center, axis, angle, duration=2.0):
        """Generic Cartesian arc; no door or handle semantics in this primitive."""
        center = np.asarray(center, dtype=float)
        start_pos, start_rot = self.backend.tcp_pose()
        return self._follow(lambda t: (
            center + axis_rotation(axis, t * angle) @ (start_pos - center),
            axis_rotation(axis, t * angle) @ start_rot,
        ), duration)

    def set_gripper(self, width_m, duration=0.6):
        if not np.isfinite(width_m) or not 0 <= width_m <= 0.08:
            raise ValueError("gripper width must be within [0, 0.08] m")
        start = self.width
        steps = self._steps(duration)
        for step in range(steps):
            self.width = start + (width_m - start) * min(1, (step + 1) / (0.7 * steps))
            self.tick()
        return self.backend.sample()
