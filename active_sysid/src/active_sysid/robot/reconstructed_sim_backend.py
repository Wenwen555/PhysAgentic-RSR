"""Independent robot Simulator from visual reconstruction, NOT reference World.

The same joint controller is reused. Initial has no latch. M1/M2 use a learned
functional latch and legacy fitted scalar mechanism parameters. No native bolt
geometry, World parameters, or World XML are loaded here.
"""
from pathlib import Path
import json
import numpy as np
import yaml
import mujoco as mj
from .fixture_backend import ROOT
from .sensors import robot_observation, contact_audit

CONFIG=ROOT/'configs/robot/simulators/legacy_visual_stage3.yaml'


class SimRobotBackend:
    role='development_sim'

    def __init__(self, variant='initial', config=CONFIG):
        task_variant=variant.startswith('robot_m')
        if task_variant:config=ROOT/'configs/robot/simulators/robot_grounding_m0_m2.yaml'
        self.cfg=yaml.safe_load(Path(config).read_text());cfg=self.cfg
        if task_variant:cfg['scene']=cfg['variants'][variant]
        if cfg['role']!=self.role:raise ValueError('Simulator role must be development_sim')
        self.scene=(ROOT/cfg['scene']).resolve()
        if not self.scene.is_relative_to(ROOT/'assets/reconstructed_sim'):
            raise ValueError('Sim cannot load a World or fixture scene')
        if variant not in cfg['variants']:raise ValueError('Unknown Sim variant')
        self.variant=variant
        self.model=mj.MjModel.from_xml_path(str(self.scene));m=self.model
        self.arm_jids=np.array([m.joint(n).id for n in cfg['arm_joints']])
        self.arm_qpos=m.jnt_qposadr[self.arm_jids];self.arm_dofs=m.jnt_dofadr[self.arm_jids]
        self.arm_actuators=np.array([m.actuator(n).id for n in cfg['arm_actuators']])
        self.finger_jids=np.array([m.joint(n).id for n in cfg['finger_joints']])
        self.finger_qpos=m.jnt_qposadr[self.finger_jids]
        self.finger_actuators=np.array([m.actuator(n).id for n in cfg['finger_actuators']])
        self.tcp_id=m.site(cfg['tcp_site']).id
        self.door_jid=m.joint(cfg['door_joint']).id;self.handle_jid=m.joint(cfg['handle_joint']).id
        self.door_qpos=int(m.jnt_qposadr[self.door_jid]);self.handle_qpos=int(m.jnt_qposadr[self.handle_jid])
        self.door_body=m.body(cfg['door_body']).id;self.handle_body=m.body(cfg['handle_body']).id
        self.handle_geom=m.geom(cfg['handle_geom']).id
        self.object_dofs=m.jnt_dofadr[[m.joint(n).id for n in cfg['object_joints']]]
        self.object_bodies=[m.body(n).id for n in cfg['object_bodies']]
        def descendants(root):
            result=set()
            for body in range(1,m.nbody):
                cur=body
                while cur and cur!=root:cur=int(m.body_parentid[cur])
                if cur==root:result.add(body)
            return result
        self.robot_bodies=descendants(m.body(cfg['robot_base']).id)
        self.finger_geom_sets=[]
        for name in cfg['finger_bodies']:
            bodies=descendants(m.body(name).id)
            self.finger_geom_sets.append({g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies})
        self.latch_id=m.equality('sim_latch_lever').id
        self.latch_enabled=variant!='initial';self.unlock_threshold=None
        self.mechanism_parameters={}
        if task_variant:
            self.mechanism_parameters=json.loads(self.scene.with_suffix('.json').read_text())
            self.latch_enabled=self.mechanism_parameters['latch_enabled']
            self.unlock_threshold=self.mechanism_parameters['unlock_threshold']
        elif self.latch_enabled:self._apply_mechanism(ROOT/cfg['variants'][variant])
        self.data=mj.MjData(m);mj.mj_setConst(m,self.data)
        self.dt=float(m.opt.timestep);self.callbacks=[];self.samples=[]
        self.reset()

    def _apply_mechanism(self,path):
        source=mj.MjModel.from_xml_path(str(path))
        repair=json.loads(Path(str(path)+'.repair.json').read_text())
        self.mechanism_parameters=repair['params']
        self.unlock_threshold=float(repair['latch']['unlock_threshold'])
        for src_name,dst in [('door_hinge',self.door_jid),('handle_hinge',self.handle_jid)]:
            src=source.joint(src_name).id;sd=source.jnt_dofadr[src];dd=self.model.jnt_dofadr[dst]
            for field in ('dof_damping','dof_frictionloss','dof_armature'):
                getattr(self.model,field)[dd]=getattr(source,field)[sd]
            self.model.jnt_stiffness[dst]=source.jnt_stiffness[src]
            self.model.qpos_spring[self.model.jnt_qposadr[dst]]=source.qpos_spring[source.jnt_qposadr[src]]
        # Map inferred scalar hinge inertia to the reconstructed leaf without
        # overwriting its geometry/COM with the source object-space box frame.
        m=self.model;bid=self.door_body;axis=m.jnt_axis[self.door_jid]
        rot=np.zeros(9);mj.mju_quat2Mat(rot,m.body_iquat[bid]);rot=rot.reshape(3,3)
        inertia=rot@np.diag(m.body_inertia[bid])@rot.T
        r=m.body_ipos[bid]-m.jnt_pos[self.door_jid]
        effective=float(axis@inertia@axis+m.body_mass[bid]*(r@r-(r@axis)**2))
        scale=float(repair['params']['I_d'])/effective
        m.body_mass[bid]*=scale;m.body_inertia[bid]*=scale

    def reset(self):
        mj.mj_resetDataKeyframe(self.model,self.data,self.model.key(self.cfg['home_keyframe']).id)
        self.data.qpos[self.finger_qpos]=[.04,-.04]
        self.latched=self.latch_enabled;self.data.eq_active[self.latch_id]=self.latched
        self.phase='reset';self.samples.clear();mj.mj_forward(self.model,self.data)
        self.command(self.data.qpos[self.arm_qpos],.08)

    def command(self,joint_positions,width_m):
        kp=self.model.actuator_gainprm[self.arm_actuators,0]
        self.data.ctrl[self.arm_actuators]=np.asarray(joint_positions)+self.data.qfrc_bias[self.arm_dofs]/kp
        self.data.ctrl[self.finger_actuators]=[width_m/2,-width_m/2]

    def tcp_pose(self):
        return self.data.site_xpos[self.tcp_id].copy(),self.data.site_xmat[self.tcp_id].reshape(3,3).copy()

    def sample(self):
        s=robot_observation(self);s.update(contact_audit(self));d=self.data
        s.update(phase=self.phase,door_angle_rad=float(d.qpos[self.door_qpos]),
                 handle_angle_rad=float(d.qpos[self.handle_qpos]),
                 latch_active=bool(d.eq_active[self.latch_id]),
                 object_applied_torque_nm=d.qfrc_applied[self.object_dofs].copy(),
                 object_actuator_torque_nm=d.qfrc_actuator[self.object_dofs].copy(),
                 object_xfrc_max=float(np.abs(d.xfrc_applied[self.object_bodies]).max()))
        return s

    def step(self):
        if self.latch_enabled:
            door=self.data.qpos[self.door_qpos];handle=self.data.qpos[self.handle_qpos]
            if self.latched and handle>=self.unlock_threshold:self.latched=False
            elif not self.latched and handle<self.unlock_threshold and abs(door)<.05:self.latched=True
        self.data.eq_active[self.latch_id]=self.latched
        mj.mj_step(self.model,self.data);mj.mj_forward(self.model,self.data)
        s=self.sample();self.samples.append(s)
        for cb in self.callbacks:cb(self,s)
        return s

    def privileged(self):
        d=self.data
        return {'source_role':self.role,'door_angle_rad':float(d.qpos[self.door_qpos]),
                'handle_angle_rad':float(d.qpos[self.handle_qpos]),
                'handle_grasp_pos_m':d.geom_xpos[self.handle_geom].copy(),
                'handle_pivot_m':d.xanchor[self.handle_jid].copy(),
                'handle_axis_world':d.xaxis[self.handle_jid].copy(),
                'handle_rotation_world':d.xmat[self.handle_body].reshape(3,3).copy(),
                'door_pivot_m':d.xanchor[self.door_jid].copy(),
                'door_axis_world':d.xaxis[self.door_jid].copy()}
