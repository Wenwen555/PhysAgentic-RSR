"""One three-rod physical World behind the measured robot and scoring APIs."""
from copy import deepcopy

from ..api.client import APIError
from ..core.contracts import IDENTIFICATION_API_VERSION
from ..core.policy import Policy
from ..runtime.instrumented import InstrumentedTarget, observation
from .rod_lift_world import SimulatedMeasuredRobot
from .rod_scene import OBJECT_IDS, private_scene


class SceneBackend:
    def __init__(self, private_config):
        from physrsr.physics_spec import PhysicsSpec
        self.config=private_scene(private_config)
        self.seeds=tuple(self.config["seeds"])
        first=self.config["objects"][OBJECT_IDS[0]]
        self.physics=PhysicsSpec(**first["physics"],rod_half_size=first["half_size_m"],
            inertial_mode="fixed_mass_com",grip_force=2.)
        self.env=self.api=None
        self.ready=False

    def ensure(self):
        if self.env is None:
            from physrsr.paths import CAPX_ROOT
            from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
            from .rod_scene_env import ThreeRodLowLevel
            self.env=ThreeRodLowLevel(self.config,privileged=True,enable_render=False,max_steps=20000,
                controller_cfg=str(CAPX_ROOT/"capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json"))
            self.env.robosuite_env.ignore_done=True
            self.api=FrankaControlPrivilegedApi(self.env)
        return self.env,self.api

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env=self.api=None


class SceneMeasuredRobot(SimulatedMeasuredRobot):
    def describe(self):
        description=super().describe()
        env,_=self.backend.ensure()
        sim=env.robosuite_env.sim
        base=sim.model.body_name2id("robot0_base")
        # The Panda base in this environment is upright. Public bounds cover
        # the table's reachable portion and the allowed 20 cm lift height.
        import numpy as np
        rot=np.asarray(sim.data.body_xmat[base]).reshape(3,3)
        origin=np.asarray(sim.data.body_xpos[base])
        table=np.asarray(env.robosuite_env.table_offset)
        corners=[rot.T@(table+np.array([x,y,z])-origin)
            for x in (-.35,.35) for y in (-.35,.35) for z in (-.01,.23)]
        description["workspace"]=dict(min_position_m=np.min(corners,axis=0).tolist(),
            max_position_m=np.max(corners,axis=0).tolist())
        description["scene"]=dict(object_ids=list(OBJECT_IDS),one_robot=True,shared_clock=True)
        return description

    def observe(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        env,_=self.backend.ensure()
        suite=env.robosuite_env
        suite.sim.forward()
        suite._update_observables(force=True)
        state=super().observe()
        sim=suite.sim
        base=sim.model.body_name2id("robot0_base")
        rot=np.asarray(sim.data.body_xmat[base]).reshape(3,3)
        origin=np.asarray(sim.data.body_xpos[base])
        plane_z=float((rot.T@(np.asarray(suite.table_offset)-origin))[2])
        scene={}
        for key,bid in suite.rod_body_ids.items():
            pos=rot.T@(sim.data.body_xpos[bid]-origin)
            matrix=rot.T@sim.data.body_xmat[bid].reshape(3,3)
            axes=matrix.T
            q=Rotation.from_matrix(matrix).as_quat()
            yaw=float(np.arctan2(axes[0,1],axes[0,0]))
            grasp=(Rotation.from_euler("z",yaw)*Rotation.from_quat([0.,1.,0.,0.])).as_quat()
            half=np.asarray(self.backend.config["objects"][key]["half_size_m"])
            scene[key]=dict(position=pos.tolist(),quaternion_wxyz=q[[3,0,1,2]].tolist(),axes=axes.tolist(),
                dimensions_m=(half*2).tolist(),grasp_quaternion_wxyz=grasp[[3,0,1,2]].tolist(),
                support_clearance_m=float(pos[2]-np.dot(np.abs(axes[:,2]),half)-plane_z))
        return observation(dict(state,objects=scene))


class ThreeRodWorld(InstrumentedTarget):
    def __init__(self, private_config):
        self.backend=SceneBackend(private_config)
        super().__init__(SceneMeasuredRobot(self.backend))

    def call(self, method, params):
        if method=="world.inspect_compilation" and not params:
            env,_=self.backend.ensure()
            suite=env.robosuite_env;sim=suite.sim;model=sim.model
            actuators=suite.robots[0]._ref_joint_gripper_actuator_indexes["right"]
            objects={}
            for key,bid in suite.rod_body_ids.items():
                geom=suite.rods[key].contact_geoms[0]
                gid=model.geom_name2id(geom)
                objects[key]=dict(body_id=int(bid),free_joint=suite.rods[key].joints[0],
                    mass_kg=float(model.body_mass[bid]),com_body_m=model.body_ipos[bid].tolist(),
                    principal_inertia_kg_m2=model.body_inertia[bid].tolist(),
                    friction=model.geom_friction[gid].tolist())
            return dict(world_count=1,robot_count=len(suite.robots),objects=objects,
                gripper_actuator_force_ranges_n=model.actuator_forcerange[actuators].tolist())
        if method not in ("target.execute_policy","world.evaluate_policy"):
            return super().call(method,params)
        if set(params)!={"policy"}:
            raise APIError("Scene task execution accepts only one frozen policy")
        policy=Policy.from_wire(params["policy"])
        if policy.api_version!=IDENTIFICATION_API_VERSION:
            raise APIError("Scene execution requires measured API v2")
        result=super().call("target.execute_policy",params)
        env,_=self.backend.ensure()
        report=env.robosuite_env.task_report()
        status="error" if result["status"]!="completed" else "success" if report["success"] else "fail"
        return dict(result,task_result=deepcopy(report),error_kind="program" if status=="error" else None,
            record=dict(status=status,reason=result.get("error") or report["reason"],
                sim_duration_s=result["cost_s"],policy_sha256=policy.sha256))

    def close(self):
        self.backend.close()
