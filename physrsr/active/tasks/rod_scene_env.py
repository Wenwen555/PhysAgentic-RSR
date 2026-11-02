"""Actual three-body MJCF scene, sharing one table, Panda and dynamics clock.

The single-rod task remains unchanged. Its tested object inertial construction
and low-level controller are reused here, not three independent environments.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from capx.envs.simulators.robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel
from capx.envs.simulators.robosuite_rod_lift import RodLift, RodObject
from .rod_scene import OBJECT_IDS, SCENE_SUCCESS, SequentialLiftScore


class ScenePlacement:
    def __init__(self, rods, specs, table_offset):
        self.rods,self.specs,self.table_offset=rods,specs,np.asarray(table_offset)
        self.rng=None

    def sample(self):
        result={}
        for key,rod in self.rods.items():
            spec=self.specs[key]
            pos=self.table_offset+np.array([*spec["table_xy_m"],spec["half_size_m"][2]+.002])
            yaw=spec["yaw_rad"]
            result[key]=(pos,np.array([math.cos(yaw/2),0.,0.,math.sin(yaw/2)]),rod)
        return result


class ThreeRodLift(RodLift):
    def __init__(self, robots, scene_config, **kwargs):
        self.scene_config=deepcopy(scene_config)
        self.score=None
        super().__init__(robots=robots,grip_force=scene_config["grip_force"],**kwargs)

    def _load_model(self):
        # Load the robot base directly; Lift/RodLift._load_model would also
        # construct and discard a single-object task and its sampler.
        ManipulationEnv._load_model(self)
        xpos=self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena=TableArena(table_full_size=self.table_full_size,table_friction=self.table_friction,
            table_offset=self.table_offset)
        arena.set_origin([0,0,0])
        colors=([.85,.15,.12,1.],[.15,.7,.25,1.],[.15,.35,.85,1.])
        self.rods={}
        for index,key in enumerate(OBJECT_IDS):
            spec=self.scene_config["objects"][key]
            self.rods[key]=RodObject(name="cube" if index==0 else key,size=spec["half_size_m"],
                **spec["physics"],inertial_mode="fixed_mass_com",rgba=colors[index],rng=self.rng)
        self.cube=self.rods[OBJECT_IDS[0]]
        self.placement_initializer=ScenePlacement(self.rods,self.scene_config["objects"],self.table_offset)
        self.model=ManipulationTask(mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],mujoco_objects=list(self.rods.values()))

    def _setup_references(self):
        super()._setup_references()
        self.rod_body_ids={key:self.sim.model.body_name2id(rod.root_body) for key,rod in self.rods.items()}

    def _reset_internal(self):
        self.score=None
        super()._reset_internal()
        scene=self.scene_config.get("initial_scene")
        if scene is not None:
            if scene.get("coordinate_frame")!="robot_base" or set(scene.get("objects",{}))!=set(OBJECT_IDS):
                raise ValueError("Predictive reset requires the same three base-frame object tracks")
            base=self.sim.model.body_name2id("robot0_base")
            rot=np.asarray(self.sim.data.body_xmat[base]).reshape(3,3)
            origin=np.asarray(self.sim.data.body_xpos[base])
            for key,rod in self.rods.items():
                obj=scene["objects"][key]
                pos=origin+rot@np.asarray(obj["position"],dtype=float)
                q=np.asarray(obj["quaternion_wxyz"],dtype=float)
                world_rot=Rotation.from_matrix(rot)*Rotation.from_quat(q[[1,2,3,0]])
                xyzw=world_rot.as_quat()
                self.sim.data.set_joint_qpos(rod.joints[0],np.r_[pos,xyzw[[3,0,1,2]]])
        self.sim.forward()
        self.rest_positions={key:self.sim.data.body_xpos[bid].copy() for key,bid in self.rod_body_ids.items()}
        self.score=SequentialLiftScore(float(self.sim.data.time))

    def _scene_states(self):
        sim=self.sim
        robot=self.robots[0]
        fingers=robot._ref_gripper_joint_pos_indexes["right"]
        open_gripper=float(np.abs(sim.data.qpos[fingers]).sum())>=.075
        table=float(self.table_offset[2])
        states={}
        for key,bid in self.rod_body_ids.items():
            pos=sim.data.body_xpos[bid]
            axes=sim.data.body_xmat[bid].reshape(3,3)
            half=np.asarray(self.scene_config["objects"][key]["half_size_m"])
            clearance=float(pos[2]-np.dot(np.abs(axes[2,:]),half)-table)
            vertical=float(abs(axes[2,0]))
            grasped=bool(self._check_grasp(gripper=robot.gripper,object_geoms=self.rods[key]))
            # Free-joint velocity is measured from the shared simulator state.
            speed=np.asarray(sim.data.get_joint_qvel(self.rods[key].joints[0]))
            at_rest=(abs(clearance)<=SCENE_SUCCESS["support_tolerance_m"] and
                np.linalg.norm(pos[:2]-self.rest_positions[key][:2])<=SCENE_SUCCESS["return_xy_tolerance_m"] and
                np.linalg.norm(speed[:3])<=SCENE_SUCCESS["max_return_speed_m_s"] and
                np.linalg.norm(speed[3:])<=SCENE_SUCCESS["max_return_angular_speed_rad_s"] and
                vertical<SCENE_SUCCESS["max_abs_axis_z"] and not grasped and open_gripper)
            height=float(pos[2]-table)
            states[key]=dict(lifted=height>SCENE_SUCCESS["lift_above_table_m"],grasped=grasped,
                level=vertical<SCENE_SUCCESS["max_abs_axis_z"],at_rest=bool(at_rest),
                lift_margin=min(height-SCENE_SUCCESS["lift_above_table_m"],SCENE_SUCCESS["max_abs_axis_z"]-vertical))
        # Sequential means one independently grasped rod at a time. A bundled
        # grasp must not manufacture three separate successful holds.
        if sum(state["grasped"] for state in states.values())>1:
            for state in states.values(): state["grasped"]=False
        return states

    def _check_success(self):
        if self.score is None:
            return False
        self.score.update(float(self.sim.data.time),self._scene_states())
        return self.score.report()["success"]

    def reward(self, action=None):
        return float(self._check_success())

    def task_report(self):
        # Read-only: repeated scoring cannot manufacture elapsed hold time.
        return self.score.report()


class ThreeRodLowLevel(FrankaRobosuiteCubeLiftLowLevel):
    _robosuite_env_cls=ThreeRodLift

    def __init__(self, scene_config, **kwargs):
        self.scene_config=scene_config
        super().__init__(**kwargs)

    def _robosuite_env_kwargs(self):
        return dict(scene_config=self.scene_config)
