"""Native, unmodified double-door World controlled through robot actuators only."""
from __future__ import annotations

from pathlib import Path
import mujoco as mj
import numpy as np
import yaml

from .fixture_backend import ROOT
from .controller import matrix_to_quat

DEFAULT_BINDING = ROOT / 'configs/robot/world_bindings/unidoor_native.yaml'


class WorldRobotBackend:
    """Expose the existing World robot to the generic Cartesian controller.

    The source XML and its native handle-bolt equality are never patched. Only
    arm/finger actuator controls are written after the episode reset.
    """

    def __init__(self, binding=DEFAULT_BINDING, *, mechanism='lever'):
        self.binding_path = Path(binding).resolve()
        cfg = yaml.safe_load(self.binding_path.read_text())
        self.cfg = cfg
        self.scene = ROOT / cfg['scene']
        self.model = mj.MjModel.from_xml_path(str(self.scene))
        self.data = mj.MjData(self.model)
        m = self.model
        self.dt = float(m.opt.timestep)
        self.arm_jids = np.array([m.joint(n).id for n in cfg['arm_joints']])
        self.arm_qpos = m.jnt_qposadr[self.arm_jids]
        self.arm_dofs = m.jnt_dofadr[self.arm_jids]
        self.arm_actuators = np.array([m.actuator(n).id for n in cfg['arm_actuators']])
        self.finger_jids = np.array([m.joint(n).id for n in cfg['finger_joints']])
        self.finger_qpos = m.jnt_qposadr[self.finger_jids]
        self.finger_actuators = np.array([m.actuator(n).id for n in cfg['finger_actuators']])
        self.tcp_id = m.site(cfg['tcp_site']).id
        self.mechanism = cfg['mechanisms'][mechanism]
        self.mechanism_name = mechanism
        self.door_jid = m.joint(self.mechanism['door_joint']).id
        self.handle_jid = m.joint(self.mechanism['handle_joint']).id
        self.bolt_jid = m.joint(self.mechanism['bolt_joint']).id
        self.door_qpos = int(m.jnt_qposadr[self.door_jid])
        self.handle_qpos = int(m.jnt_qposadr[self.handle_jid])
        self.bolt_qpos = int(m.jnt_qposadr[self.bolt_jid])
        # Audit *both* doors, not just the selected manipulation target.
        all_object_joints = [m.joint(spec[key]).id
                             for spec in cfg['mechanisms'].values()
                             for key in ('door_joint','handle_joint','bolt_joint')]
        self.object_dofs = m.jnt_dofadr[all_object_joints]
        self.object_bodies = [m.body(spec[key]).id
                              for spec in cfg['mechanisms'].values()
                              for key in ('door_body','handle_body','bolt_body')]
        self.handle_body = m.body(self.mechanism['handle_body']).id
        self.handle_geom = m.geom(self.mechanism['handle_geom']).id
        self.door_body = m.body(self.mechanism['door_body']).id
        self.equality_id = m.equality(self.mechanism['constraint']).id
        self.finger_bodies = [m.body(n).id for n in cfg['finger_bodies']]
        self.finger_geom_sets = [set() for _ in self.finger_bodies]
        for gid in range(m.ngeom):
            parent = int(m.geom_bodyid[gid])
            for i, body in enumerate(self.finger_bodies):
                current = parent
                while current and current != body:
                    current = int(m.body_parentid[current])
                if current == body:
                    self.finger_geom_sets[i].add(gid)
        self.robot_bodies = set()
        robot_base = m.body('robot0_base').id
        for body in range(m.nbody):
            current = body
            while current and current != robot_base:
                current = int(m.body_parentid[current])
            if current == robot_base:
                self.robot_bodies.add(body)
        self.home_id = m.key(cfg['home_keyframe']).id
        self.phase = 'reset'
        self.samples = []
        self.callbacks = []
        self.reset()

    def reset(self):
        mj.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        # Initial gripper reset: open fingers; no object joint is touched.
        self.data.qpos[self.finger_qpos] = [0.04, -0.04]
        self.phase = 'reset'
        self.samples.clear()
        mj.mj_forward(self.model, self.data)
        self.command(self.data.qpos[self.arm_qpos], 0.08)

    def command(self, joint_positions, width_m):
        """Drive only the nine robot actuators; never write object qpos/forces."""
        m, d = self.model, self.data
        kp = m.actuator_gainprm[self.arm_actuators, 0]
        gravity_compensated = np.asarray(joint_positions) + d.qfrc_bias[self.arm_dofs] / kp
        d.ctrl[self.arm_actuators] = gravity_compensated
        d.ctrl[self.finger_actuators] = [0.5 * width_m, -0.5 * width_m]

    def tcp_pose(self):
        d = self.data
        return d.site_xpos[self.tcp_id].copy(), d.site_xmat[self.tcp_id].reshape(3,3).copy()

    def sample(self):
        m, d = self.model, self.data
        pos, rot = self.tcp_pose()
        fingers = np.zeros(2)
        total_force = 0.0
        net = np.zeros(3)
        max_contact = 0.0
        unexpected = 0.0
        self_contact = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            force = np.zeros(6)
            mj.mj_contactForce(m, d, i, force)
            norm = float(np.linalg.norm(force[:3]))
            normal = float(max(0, force[0]))
            max_contact = max(max_contact, norm)
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.handle_geom in (g1, g2):
                other = g2 if g1 == self.handle_geom else g1
                finger_indices = [i for i, geoms in enumerate(self.finger_geom_sets) if other in geoms]
                if finger_indices:
                    fingers[finger_indices[0]] += normal
                    total_force += norm
                    # Direction is not needed for acceptance, but is logged.
                    world_force = c.frame.reshape(3, 3).T @ force[:3]
                    net += world_force * (1 if g2 == self.handle_geom else -1)
                elif int(m.geom_bodyid[other]) in self.robot_bodies:
                    unexpected = max(unexpected, norm)
            b1,b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            r1,r2 = b1 in self.robot_bodies,b2 in self.robot_bodies
            if r1 and r2:
                # Finger-finger contacts are expected after closing but report separately.
                self_contact = max(self_contact,norm)
            elif r1 != r2 and not (self.handle_geom in (g1,g2)):
                unexpected = max(unexpected,norm)
        sensor = np.r_[d.sensor(self.cfg['wrist_force_sensor']).data,
                       d.sensor(self.cfg['wrist_torque_sensor']).data]
        return {
            'phase': self.phase,'time_s':float(d.time),
            'q':d.qpos[self.arm_qpos].copy(),'dq':d.qvel[self.arm_dofs].copy(),
            'joint_torque_nm':d.actuator_force[self.arm_actuators].copy(),
            'ee_pos_m':pos,'ee_quat_wxyz':matrix_to_quat(rot),
            'gripper_width_m':float(d.qpos[self.finger_qpos[0]]-d.qpos[self.finger_qpos[1]]),
            'wrist_wrench_ee':sensor,
            'finger_contact_normal_n':fingers,
            'handle_contact_force_world_n':net,
            'total_handle_contact_force_n':float(total_force),
            'max_contact_force_n':float(max_contact),
            'unexpected_contact_force_n':float(unexpected),
            'self_contact_force_n':float(self_contact),
            'door_angle_rad':float(d.qpos[self.door_qpos]),
            'handle_angle_rad':float(d.qpos[self.handle_qpos]),
            'bolt_extension_m':float(d.qpos[self.bolt_qpos]),
            'native_constraint_active':bool(d.eq_active[self.equality_id]),
            'coupling_residual_m':float(d.qpos[self.bolt_qpos] -
                m.eq_data[self.equality_id, 1] * d.qpos[self.handle_qpos]),
            'object_applied_torque_nm':d.qfrc_applied[self.object_dofs].copy(),
            'object_actuator_torque_nm':np.zeros(len(self.object_dofs)),
            'object_xfrc_max':float(np.max(np.abs(d.xfrc_applied[self.object_bodies]))),
            'handle_in_tcp_m':rot.T @ (d.geom_xpos[self.handle_geom]-pos),
            'handle_in_tcp_quat_wxyz':matrix_to_quat(rot.T @ d.xmat[self.handle_body].reshape(3, 3)),
        }

    def step(self):
        mj.mj_step(self.model,self.data)
        mj.mj_forward(self.model,self.data)
        s=self.sample()
        self.samples.append(s)
        for cb in self.callbacks:cb(self,s)
        return s

    def privileged(self):
        d=self.data
        return {
            'handle_grasp_pos_m':d.geom_xpos[self.handle_geom].copy(),
            'handle_pivot_m':d.xanchor[self.handle_jid].copy(),
            'handle_axis_world':d.xaxis[self.handle_jid].copy(),
            'handle_rotation_world':d.xmat[self.handle_body].reshape(3,3).copy(),
            'door_pivot_m':d.xanchor[self.door_jid].copy(),
            'door_axis_world':d.xaxis[self.door_jid].copy(),
            'door_angle_rad':float(d.qpos[self.door_qpos]),
            'handle_angle_rad':float(d.qpos[self.handle_qpos]),
            'bolt_extension_m':float(d.qpos[self.bolt_qpos]),
        }
