"""Panda joint actuation plus passive door/handle contact dynamics (M3-R)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np
import mujoco as mj

from active_sysid.robot.mechanism import apply_mechanism

ROOT = Path(__file__).resolve().parents[3]
HOME = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.7853981634])


class RobotDoorBackend:
    def __init__(self, scene: str | Path | None = None, *, mechanism_mjcf=None):
        self.scene = Path(scene or ROOT / "assets/fixtures/panda_single_door/oracle_robot_door.xml").resolve()
        self.model = mj.MjModel.from_xml_path(str(self.scene))
        self.mechanism = None
        if mechanism_mjcf is not None:
            self.mechanism = apply_mechanism(self.model, mechanism_mjcf)
        self.data = mj.MjData(self.model)
        mj.mj_setConst(self.model, self.data)
        self.dt = float(self.model.opt.timestep)
        self.arm_jids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 8)])
        self.arm_qpos = self.model.jnt_qposadr[self.arm_jids]
        self.arm_dofs = self.model.jnt_dofadr[self.arm_jids]
        self.arm_actuators = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.finger_jids = np.array([self.model.joint(f"finger_joint{i}").id for i in (1, 2)])
        self.finger_qpos = self.model.jnt_qposadr[self.finger_jids]
        self.gripper_actuator = self.model.actuator("actuator8").id
        self.object_jids = np.array([self.model.joint(n).id for n in ("door_hinge", "handle_hinge")])
        self.object_qpos = self.model.jnt_qposadr[self.object_jids]
        self.object_dofs = self.model.jnt_dofadr[self.object_jids]
        self.object_bodies = [self.model.body(n).id for n in ("door", "handle")]
        self.finger_bodies = [self.model.body(n).id for n in ("left_finger", "right_finger")]
        self.handle_body = self.model.body("handle").id
        self.tcp_id = self.model.site("tcp").id
        self.grasp_site_id = self.model.site("handle_grasp").id
        self.latch_id = self.model.equality("latch_weld").id
        self.robot_bodies = set()
        for bid in range(1, self.model.nbody):
            ancestor = bid
            while ancestor and ancestor != self.model.body("link0").id:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor:
                self.robot_bodies.add(bid)
        latch = self.mechanism["latch"] if self.mechanism else {
            "enabled": bool(self.model.eq_active0[self.latch_id]),
            "unlock_threshold": 0.556, "relock_door_eps": 0.05,
        }
        self.latch_enabled = bool(latch["enabled"])
        self.unlock_threshold = float(latch["unlock_threshold"])
        self.relock_door_eps = float(latch["relock_door_eps"])
        self.callbacks: list[Callable] = []
        self.phase = "reset"
        self.samples: list[dict] = []
        self.reset()

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos] = HOME
        self.data.qpos[self.finger_qpos] = 0.04
        self.latched = self.latch_enabled
        self.data.eq_active[self.latch_id] = int(self.latched)
        mj.mj_forward(self.model, self.data)
        self.command(HOME, 0.08)
        self.samples.clear()

    def command(self, joint_positions, width_m):
        """Only robot actuator controls are written; there is no object action API."""
        gain = self.model.actuator_gainprm[self.arm_actuators, 0]
        targets = np.asarray(joint_positions) + self.data.qfrc_bias[self.arm_dofs] / gain
        ranges = self.model.actuator_ctrlrange[self.arm_actuators]
        self.data.ctrl[self.arm_actuators] = np.clip(targets, ranges[:, 0], ranges[:, 1])
        self.data.ctrl[self.gripper_actuator] = np.clip(width_m, 0, 0.08) / 0.08 * 255.0

    def step(self):
        door, handle = self.data.qpos[self.object_qpos]
        if self.latch_enabled:
            if self.latched and handle >= self.unlock_threshold:
                self.latched = False
            elif not self.latched and handle < self.unlock_threshold and abs(door) < self.relock_door_eps:
                self.latched = True
        self.data.eq_active[self.latch_id] = int(self.latched)
        mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        sample = self.sample()
        self.samples.append(sample)
        for callback in self.callbacks:
            callback(self, sample)
        return sample

    def tcp_pose(self):
        rotation = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        return self.data.site_xpos[self.tcp_id].copy(), rotation

    def sample(self):
        pos, rotation = self.tcp_pose()
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, rotation.ravel())
        normal = np.zeros(2)
        handle_force = np.zeros(3)
        max_force = total_handle_force = unexpected = self_force = 0.0
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            bodies = [int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            is_robot = [b in self.robot_bodies for b in bodies]
            if not any(is_robot):
                continue
            wrench = np.zeros(6)
            mj.mj_contactForce(self.model, self.data, idx, wrench)
            magnitude = float(np.linalg.norm(wrench[:3]))
            max_force = max(max_force, magnitude)
            if all(is_robot):
                self_force = max(self_force, magnitude)
            elif self.handle_body in bodies:
                total_handle_force += magnitude
                # mj_contactForce is on geom2; reverse when the handle is geom1.
                force_world = contact.frame.reshape(3, 3).T @ wrench[:3]
                handle_force += force_world if bodies[1] == self.handle_body else -force_world
                for side, finger in enumerate(self.finger_bodies):
                    if finger in bodies:
                        normal[side] += max(0.0, float(wrench[0]))
            else:
                unexpected = max(unexpected, magnitude)
        door, handle = self.data.qpos[self.object_qpos]
        relative_quat = np.zeros(4)
        mj.mju_mat2Quat(relative_quat, (rotation.T @ self.data.xmat[self.handle_body].reshape(3, 3)).ravel())
        return {
            "time_s": float(self.data.time), "phase": self.phase,
            "q": self.data.qpos[self.arm_qpos].copy(),
            "dq": self.data.qvel[self.arm_dofs].copy(),
            "joint_torque_nm": self.data.qfrc_actuator[self.arm_dofs].copy(),
            "wrist_wrench_ee": np.r_[self.data.sensor("wrist_force").data, self.data.sensor("wrist_torque").data].copy(),
            "ee_pos_m": pos, "ee_quat_wxyz": quat,
            "gripper_width_m": float(sum(self.data.qpos[self.finger_qpos])),
            "finger_contact_normal_n": normal,
            "handle_contact_force_world_n": handle_force,
            "max_contact_force_n": max_force,
            "total_handle_contact_force_n": total_handle_force,
            "unexpected_contact_force_n": unexpected,
            "self_contact_force_n": self_force,
            "door_angle_rad": float(door), "handle_angle_rad": float(handle),
            "latch_active": bool(self.latched),
            "handle_in_tcp_m": rotation.T @ (self.data.site_xpos[self.grasp_site_id] - pos),
            "handle_in_tcp_quat_wxyz": relative_quat,
            "object_applied_torque_nm": self.data.qfrc_applied[self.object_dofs].copy(),
            "object_actuator_torque_nm": self.data.qfrc_actuator[self.object_dofs].copy(),
            "object_xfrc_max": float(np.max(np.abs(self.data.xfrc_applied[self.object_bodies]))),
        }

    def model_audit(self):
        robot_joints = set(self.arm_jids) | set(self.finger_jids)
        transmissions = []
        for aid in range(self.model.nu):
            trntype = int(self.model.actuator_trntype[aid])
            target = int(self.model.actuator_trnid[aid, 0])
            if trntype == int(mj.mjtTrn.mjTRN_JOINT):
                allowed = target in robot_joints
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, target)
            elif trntype == int(mj.mjtTrn.mjTRN_TENDON):
                start = self.model.tendon_adr[target]
                count = self.model.tendon_num[target]
                allowed = all(int(self.model.wrap_type[i]) == int(mj.mjtWrap.mjWRAP_JOINT)
                              and int(self.model.wrap_objid[i]) in robot_joints
                              for i in range(start, start + count))
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_TENDON, target)
            else:
                allowed, name = False, str(target)
            transmissions.append({"actuator": mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, aid),
                                  "target": name, "robot_only": bool(allowed)})
        forbidden_weld = False
        for i in range(self.model.neq):
            if int(self.model.eq_type[i]) in (int(mj.mjtEq.mjEQ_WELD), int(mj.mjtEq.mjEQ_CONNECT)):
                b1, b2 = int(self.model.eq_obj1id[i]), int(self.model.eq_obj2id[i])
                forbidden_weld |= (b1 in self.robot_bodies) != (b2 in self.robot_bodies)
        return {"actuators": transmissions, "robot_only_actuators": all(t["robot_only"] for t in transmissions),
                "robot_object_constraint": bool(forbidden_weld), "mocap_bodies": int(self.model.nmocap),
                "handle_collision_enabled": bool(self.model.geom_contype[self.model.geom("handle_bar").id]
                                                 and self.model.geom_conaffinity[self.model.geom("handle_bar").id])}
