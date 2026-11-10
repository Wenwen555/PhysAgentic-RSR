"""Robot-space policy API. Privileged object state is opt-in and labelled.

All motion methods take SI units, world-frame poses and wxyz quaternions.
The API contains no direct door/handle joint commands and no open-door macro.
"""
from __future__ import annotations
import numpy as np

from active_sysid.robot.adapter import RobotExecutionAdapter


class RobotRuntime:
    def __init__(self, backend, *, observations="robot", **controller_options):
        if observations not in {"robot", "visual", "privileged"}:
            raise ValueError("observations must be robot, visual or privileged")
        self._backend = backend
        self._adapter = RobotExecutionAdapter(backend, **controller_options)
        self.observations = observations
        self.actions = []
        self._renderer = None

    def phase(self, name):
        self._backend.phase = str(name)

    def observe(self, *, rgb=False, camera="overview"):
        b = self._backend
        s = b.sample()
        keys = ("time_s", "q", "dq", "joint_torque_nm", "wrist_wrench_ee", "ee_pos_m", "ee_quat_wxyz",
                "gripper_width_m", "finger_contact_normal_n", "handle_contact_force_world_n",
                "total_handle_contact_force_n", "unexpected_contact_force_n", "self_contact_force_n")
        result = {k: s[k] for k in keys}
        if self.observations == "privileged":
            handle = b.model.body("handle").id
            door = b.model.body("door").id
            result["privileged"] = {
                "door_angle_rad": s["door_angle_rad"], "handle_angle_rad": s["handle_angle_rad"],
                "handle_pivot_m": b.data.xpos[handle].copy(),
                "handle_axis_world": b.data.xaxis[b.object_jids[1]].copy(),
                "handle_grasp_pos_m": b.data.site_xpos[b.grasp_site_id].copy(),
                "handle_rotation_world": b.data.xmat[handle].reshape(3, 3).copy(),
                "door_pivot_m": b.data.xpos[door].copy(),
                "door_axis_world": b.data.xaxis[b.object_jids[0]].copy(),
            }
        if rgb:
            if self._renderer is None:
                import mujoco
                self._renderer = mujoco.Renderer(b.model, height=480, width=640)
            self._renderer.update_scene(b.data, camera=camera)
            result["rgb"] = self._renderer.render().copy()
        return result

    def _action(self, name, **arguments):
        entry = {"action": name, "arguments": arguments, "phase": self._backend.phase,
                 "start_time_s": float(self._backend.data.time)}
        self.actions.append(entry)
        try:
            getattr(self._adapter, name)(**arguments)
            entry["status"] = "completed"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["end_time_s"] = float(self._backend.data.time)
        return self.observe()

    def move_ee(self, position_m, quaternion_wxyz, *, duration_s=1.0):
        return self._action("move_ee", position_m=np.asarray(position_m).tolist(),
                            quaternion_wxyz=np.asarray(quaternion_wxyz).tolist(), duration_s=duration_s)

    def move_joints(self, positions_rad, *, duration_s=2.0):
        return self._action("move_joints", positions_rad=np.asarray(positions_rad).tolist(),
                            duration_s=duration_s)

    def move_arc(self, center_m, axis_world, angle_rad, *, duration_s=2.0):
        return self._action("move_arc", center_m=np.asarray(center_m).tolist(),
                            axis_world=np.asarray(axis_world).tolist(), angle_rad=float(angle_rad), duration_s=duration_s)

    def set_gripper(self, width_m, *, duration_s=0.6):
        return self._action("set_gripper", width_m=float(width_m), duration_s=duration_s)

    def hold(self, duration_s):
        return self._action("hold", duration_s=float(duration_s))

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
