"""Trusted passive instrumentation; policies retain the existing robot RPC API."""
from dataclasses import asdict
import numpy as np
from physrsr.coding.runtime import CodeProgram
from .feedback import world_to_base, summarize, completion_error


class TracedProgram(CodeProgram):
    def run(self,env,api,physics=None):
        self.samples=[];self.stage='approach';self.closed=False;self.physics=physics;self.last_requested_pose=None
        self.contact_check=dict(count=0,minimum=None,maximum=None)
        self.env=env;self.start_time=float(env.robosuite_env.sim.data.time)
        self.original_step=env._do_robosuite_step
        def step(action):
            self.original_step(action);self.sample('control_step')
        env._do_robosuite_step=step
        self.sample('initial')
        owner=self
        class Events:
            def open_gripper(self):
                owner.stage='release_recovery';owner.sample('open_gripper:before')
                api.open_gripper();owner.closed=False;owner.sample('open_gripper:after')
                owner.stage='approach'
            def close_gripper(self):
                owner.stage='contact_grasp';owner.sample('close_gripper:before')
                api.close_gripper();owner.closed=True;owner.sample('close_gripper:after')
            def goto_pose(self,position,quaternion_wxyz,z_approach=0.,speed=None):
                _,tcp=env.get_eef_pose()
                owner.stage=('lift' if position[2]-tcp[2]>.005 else 'transport_hold') if owner.closed else 'approach'
                owner.last_requested_pose=dict(coordinate_frame='robot_base',position=np.asarray(position).tolist(),
                    quaternion_wxyz=np.asarray(quaternion_wxyz).tolist(),speed=speed,z_approach=z_approach)
                owner.sample('goto_pose:before')
                api.goto_pose(position,quaternion_wxyz,z_approach=z_approach,speed=speed)
                owner.sample('goto_pose:after')
        super().run(env,Events(),physics)

    def sample(self,event):
        env=self.env;rod=env.robosuite_env;sim=rod.sim
        bid=sim.model.body_name2id(rod.robots[0].robot_model.root_body)
        obj=rod.cube_body_id
        tcp_quat,tcp=env.get_eef_pose()
        axis=np.asarray(sim.data.body_xmat[obj]).reshape(3,3)[:,0]
        base_rot=np.asarray(sim.data.body_xmat[bid]).reshape(3,3)
        for ci in range(sim.data.ncon):
            c=sim.data.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
            other=g2 if sim.model.geom_bodyid[g1]==obj else g1 if sim.model.geom_bodyid[g2]==obj else None
            if other is not None and 'finger' in (sim.model.geom_id2name(other) or ''):
                mu=float(c.friction[0]);check=self.contact_check
                if not np.isclose(mu,self.physics.friction,atol=1e-7):raise RuntimeError('Mixed contact friction mismatch')
                check['count']+=1
                check['minimum']=mu if check['minimum'] is None else min(mu,check['minimum'])
                check['maximum']=mu if check['maximum'] is None else max(mu,check['maximum'])
        self.samples.append(dict(time_s=float(sim.data.time)-self.start_time,event=event,
            phase=self.stage,coordinate_frame='robot_base',
            rod_position=world_to_base(sim.data.body_xpos[obj],sim.data.body_xpos[bid],base_rot),
            rod_axis=(base_rot.T@axis).tolist(),tcp_position=np.asarray(tcp).tolist(),
            tcp_quaternion_wxyz=np.asarray(tcp_quat).tolist(),
            last_requested_pose=self.last_requested_pose,
            rod_tilt=float(abs(axis[2])),grasped=bool(rod._is_grasped()),
            lifted=bool(rod._is_lifted())))


def run_batch(physics,programs,seeds,role):
    from physrsr.worker import RolloutWorker
    worker=RolloutWorker(physics);out=[]
    try:
        env,_=worker._ensure_env();env.robosuite_env.ignore_done=True
        assert np.array_equal(worker.reset_fingerprint(seeds[0]),worker.reset_fingerprint(seeds[0]))
        for name,source in programs.items():
            for seed in seeds:
                p=TracedProgram(name,source);step=env._do_robosuite_step
                try: record=asdict(worker.rollout(p,seed,role=role))
                finally:env._do_robosuite_step=step
                samples=getattr(p,'samples',[])
                row=dict(program=p.specification(),record=record,error_kind=p.error_kind if record['status']=='error' else None,
                         rpc=p.runtime,trajectory=samples)
                sim=env.robosuite_env.sim
                base=sim.model.body_name2id('robot0_base')
                row['frame_metadata']=dict(raw_phase_rod_frame='mujoco_world',tcp_frame='robot_base',
                    base_position_world=sim.data.body_xpos[base].tolist(),
                    base_rotation_world=np.asarray(sim.data.body_xmat[base]).reshape(3,3).tolist())
                row['feedback']=summarize(samples,record)
                row['record']['nominal_target_error_m']=completion_error(samples)
                # Check actual mixed rod-finger contacts on every executed trajectory.
                row['contact_friction_check']=getattr(p,'contact_check',dict(count=0))
                out.append(row)
                if row['error_kind']=='infrastructure':return out
    finally:worker.close()
    return out
