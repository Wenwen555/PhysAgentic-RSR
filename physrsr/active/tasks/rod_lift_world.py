"""Rod Lift simulator measurements and World scoring through the shared v2 API.

Task-specific geometry, Robosuite sensor bindings and the original task evaluator
stay here. Generated policies still use the same API as the real-arm adapter.
"""
from dataclasses import asdict
import copy

from ..api.client import APIError
from ..core.contracts import IDENTIFICATION_API_VERSION
from ..core.identification import ROBOT_API
from ..core.policy import Policy
from ..runtime.instrumented import (
    MOTION_CONTRACT, observation, move_with_feedback, InstrumentedTarget, InstrumentedProgram)


class SimulatedMeasuredRobot:
    """Task adapter: visual geometry + wrist transducer + encoder channels."""
    def __init__(self, backend):
        self.backend, self.belief = backend, {}

    def prepare(self):
        env, api = self.backend.ensure()
        seed = int(self.backend.seeds[0])
        api.cfg = None
        env.robosuite_env.seed_episode(seed)
        env.reset(seed=seed)
        self.backend.ready = True
        self.reference_position = self.observe()["object"]["position"]
        # A simulator reset consumes no simulated robot time. Hardware reports
        # setup cost explicitly instead of inheriting this zero-cost convention.
        self.setup_cost_s = 0.

    def describe(self):
        env, _ = self.backend.ensure()
        robot = env.robosuite_env.robots[0]
        return dict(api_version=IDENTIFICATION_API_VERSION, robot_api=ROBOT_API,
            trajectory_sampling=dict(mode="simulation_control_ticks",sample_rate_hz=20.),
            motion_contract=copy.deepcopy(MOTION_CONTRACT),
            workspace=dict(xy_radius_m=.25, z_relative_min_m=-.02, z_relative_max_m=.20,
                reference_position_m=getattr(self, "reference_position", self.observe()["object"]["position"])),
            gravity_m_s2=[0., 0., -9.81],
            gripper=dict(normal_force_limit_per_finger_n=self.backend.physics.grip_force,
                finger_count=2, max_opening_m=.08, pad_half_extents_m=[.008, .004, .008],
                contact_moment_capacity_nm=None,
                note="Force cap is actuator limit, not measured contact normal force. Moment capacity depends on unknown contact/friction; never equate arm torque limits with finger capacity."),
            arm=dict(joint_torque_limits_nm=robot.torque_limits[1].tolist()),
            sensor_model=dict(position_resolution_m=.001, force_resolution_n=.02, torque_resolution_nm=.002,
                resolution_provenance="Conservative engineering analysis floors, not calibrated sensor specifications or statistical confidence bounds",
                wrench="Raw physical wrist force/torque transducer, including gripper weight; tare in the same orientation. No mass/COM-based synthetic wrench."),
            identification=copy.deepcopy(self.belief))

    def observe(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from ..runtime.observations import observation_provider
        env, _ = self.backend.ensure()
        # lite_physics uses split step1/step2. Refresh acceleration-stage
        # transducers from the current solved dynamics; otherwise sensordata
        # can retain the reset sample even as poses change. No time is advanced.
        env.robosuite_env.sim.forward()
        legacy = observation_provider(env)
        sensed = env.get_observation()["cube_poses"]["primary"]
        obj_r = Rotation.from_quat([*sensed[4:7], sensed[3]])
        axes = obj_r.as_matrix().T
        # Visual dimensions and support plane are perception configuration,
        # independent of the hidden object's mass, friction and inertial centre.
        dimensions = np.asarray(self.backend.physics.rod_half_size)*2
        tcp_q, tcp_p = env.get_eef_pose()
        sim, rob = env.robosuite_env.sim, env.robosuite_env.robots[0]
        base = sim.model.body_name2id("robot0_base")
        rb = np.asarray(sim.data.body_xmat[base]).reshape(3, 3)
        # Command orientation uses the arm's hand frame. The gripper model's
        # eef body has a fixed mounting rotation, so its raw quaternion is not
        # the command quaternion. Keep the measured TCP point, align its axes
        # with the hand as the real URDF hand_tcp frame does.
        hand = sim.model.body_name2id(rob.robot_model.eef_name["right"])
        hand_xyzw = Rotation.from_matrix(rb.T @ np.asarray(sim.data.body_xmat[hand]).reshape(3, 3)).as_quat()
        tcp_q = np.array([hand_xyzw[3], *hand_xyzw[:3]])
        sensor_name = rob.gripper["right"].important_sensors["force_ee"]
        sid = int(sim.model.sensor_objid[sim.model.sensor_name2id(sensor_name)])
        rs = np.asarray(sim.data.site_xmat[sid]).reshape(3, 3)
        sensor_p = rb.T @ (sim.data.site_xpos[sid]-sim.data.body_xpos[base])
        force = rb.T @ rs @ rob.ee_force["right"]
        torque = rb.T @ rs @ rob.ee_torque["right"] + np.cross(sensor_p-tcp_p, force)
        plane_z = float(env.robosuite_env.model.mujoco_arena.table_offset[2]-sim.data.body_xpos[base][2])
        clearance = legacy["rod_position"][2] - float(np.dot(np.abs(axes[:, 2]), dimensions/2)) - plane_z
        fingers = rob._ref_gripper_joint_pos_indexes["right"]
        return observation(dict(time_s=float(sim.data.time), coordinate_frame="robot_base",
            object=dict(position=legacy["rod_position"], quaternion_wxyz=sensed[3:].tolist(),
                axes=axes.tolist(), dimensions_m=dimensions.tolist(),
                grasp_quaternion_wxyz=legacy["grasp_quat"], support_clearance_m=clearance),
            tcp=dict(position=tcp_p.tolist(), quaternion_wxyz=tcp_q.tolist()),
            wrench=dict(force_n=force.tolist(), torque_nm=torque.tolist(), valid=True, frame="robot_base", about="tcp",
                provenance="wrist force_ee / torque_ee transducer; transformed to robot_base and shifted to measured TCP; includes tool self-weight"),
            joints=dict(position_rad=sim.data.qpos[rob._ref_joint_pos_indexes].tolist(),
                actuator_torque_nm=sim.data.qfrc_actuator[rob._ref_joint_vel_indexes].tolist()),
            gripper=dict(opening_m=float(np.abs(sim.data.qpos[fingers]).sum()))))

    def action(self, method, params, control=None):
        env,_=self.backend.ensure()
        self.control_samples=[]
        original=env._step_once
        last=env.robosuite_env.sim.data.time
        def sample_step(*args,**kwargs):
            nonlocal last
            result=original(*args,**kwargs)
            now=float(env.robosuite_env.sim.data.time)
            if now-last>=.05-1e-9:
                self.control_samples.append(self.observe());last=now
            return result
        env._step_once=sample_step
        try:return self._action(method,params,control)
        finally:env._step_once=original

    def drain_observations(self):
        samples=getattr(self,"control_samples",[])
        self.control_samples=[]
        return samples

    def _action(self, method, params, control=None):
        env, api = self.backend.ensure()
        if method == "goto_pose":
            self.last_motion = move_with_feedback(control or api, self.observe, params,
                env.robosuite_env.robots[0].torque_limits[1].tolist())
        elif method == "hold":
            # A hold captures the measured configuration. reset()'s cached
            # controller target can differ from the encoders (including a
            # legacy wrist-angle offset); forwarding it would command a move.
            robot = env.robosuite_env.robots[0]
            env._current_joints = env.robosuite_env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
            deadline = float(env.robosuite_env.sim.data.time)+params["duration_s"]
            while float(env.robosuite_env.sim.data.time) < deadline-1e-9:
                env._step_once()
        else:
            getattr(control or api, method)(**params)


class InstrumentedWorld(InstrumentedTarget):
    def __init__(self, private_config):
        from ..runtime.backends import SimulatedRobotService
        from .rod_lift import ROD_SUCCESS
        self.task_contract=dict(id="rod_lift",success=ROD_SUCCESS,constraints=dict(grip_force_per_finger_n=2.))
        self.backend = SimulatedRobotService(private_config)
        super().__init__(SimulatedMeasuredRobot(self.backend))

    def call(self, method, params):
        if method == "world.inspect_compilation" and not params:
            env, _ = self.backend.ensure()
            suite = env.robosuite_env
            model = suite.sim.model
            body = suite.cube_body_id
            actuators = suite.robots[0]._ref_joint_gripper_actuator_indexes["right"]
            return dict(mass_kg=float(model.body_mass[body]), com_body_m=model.body_ipos[body].tolist(),
                principal_inertia_kg_m2=model.body_inertia[body].tolist(),
                gripper_actuator_force_ranges_n=model.actuator_forcerange[actuators].tolist())
        if method not in ("world.evaluate_policy","target.execute_policy"):
            return super().call(method, params)
        if set(params) not in ({"policy"},{"policy","expected"}):
            raise APIError("Evaluation accepts only a frozen policy")
        self.check_execution(params)
        policy = Policy.from_wire(params["policy"])
        if policy.api_version != IDENTIFICATION_API_VERSION:
            raise APIError("Expected identification API")
        program = InstrumentedProgram(policy, self.robot,self.replay_budget)
        self.backend.ensure()
        record = asdict(self.backend.worker.rollout(program, int(self.backend.seeds[0]), role="world"))
        samples = [] if program.dispatcher is None else program.dispatcher.samples
        samples.append(dict(event="terminal", observation=self.robot.observe()))
        return dict(policy_sha256=policy.sha256, record=record, observations=samples,
            status="program_error" if program.error_kind=="program" else "infrastructure_error" if record["status"]=="error" else "completed",
            task_result=dict(success=record["status"]=="success",reason=record["reason"]),
            program_return=program.runtime.get("return_value"), error_kind=program.error_kind)

    def close(self):
        self.backend.close()
