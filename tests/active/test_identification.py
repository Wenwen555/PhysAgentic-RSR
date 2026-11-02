"""Generic experimental contracts and real/sim sensor parity, without task motion."""
import copy

import pytest

from physrsr.active.core.contracts import IDENTIFICATION_API_VERSION
from physrsr.active.core.policy import Policy, execute
from physrsr.active.agents.physic import probe_prompt
from physrsr.active.core.programs import task_candidate, source_policy
from physrsr.active.runtime.instrumented import (
    Dispatcher, MeasuredRobot, observation, InstrumentedTarget, move_with_feedback)
from physrsr.active.tasks.rod_lift_world import SimulatedMeasuredRobot
from physrsr.active.runtime.robot import RobotContractError


def sensed():
    return dict(time_s=1., coordinate_frame="robot_base",
        object=dict(position=[.5, 0, .1], quaternion_wxyz=[1, 0, 0, 0],
            axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], dimensions_m=[.2, .1, .04],
            grasp_quaternion_wxyz=[0, 0, 1, 0], support_clearance_m=0., hidden_com=.08),
        tcp=dict(position=[.5, 0, .1], quaternion_wxyz=[0, 0, 1, 0]),
        wrench=dict(force_n=[0, 0, 2], torque_nm=[0, .01, 0], valid=True,
            frame="robot_base", about="tcp", provenance="Measured wrist transducer"),
        joints=dict(position_rad=[0]*7, actuator_torque_nm=[1]*7),
        gripper=dict(opening_m=.03), secret_mass=999.)


def test_sensor_projection_rejects_wrong_wrench_reference_and_strips_truth():
    result = observation(sensed())
    assert "hidden_com" not in result["object"]
    assert "secret_mass" not in result
    raw = sensed()
    raw["wrench"]["about"] = "sensor_origin"
    with pytest.raises(ValueError, match="measured TCP"):
        observation(raw)


def test_new_api_preserves_old_budget_and_allows_explicit_identification_budget():
    source = "def run(robot):\n    for i in range(100):\n        robot.observe()\n    return {'observations':100}\n"
    dispatch = lambda method, params: {}
    from physrsr.coding.runtime import ProgramError
    with pytest.raises(ProgramError, match="budget"):
        execute(Policy(source), dispatch)
    policy = Policy(source, api_version=IDENTIFICATION_API_VERSION)
    assert Policy.from_wire(policy.wire()).api_version == IDENTIFICATION_API_VERSION
    result = execute(policy, dispatch)
    assert result["return_value"]["observations"] == 100


class Robot:
    def __init__(self):
        self.state = observation(sensed())
        self.actions = []
        self.belief = {}
    def observe(self):
        return copy.deepcopy(self.state)
    def describe(self):
        return {"identification":self.belief}
    def prepare(self):
        self.prepared = True
    def action(self, method, params):
        self.actions.append((method, params))
        if method == "hold":
            self.state["time_s"] += params["duration_s"]


def test_timed_hold_and_release_check_use_measured_object_support():
    robot = Robot()
    dispatcher = Dispatcher(robot, robot.observe())
    dispatcher("hold", dict(duration_s=.4))
    assert dispatcher("observe", {})["time_s"] == pytest.approx(1.4)
    robot.state["object"]["support_clearance_m"] = .03
    with pytest.raises(RobotContractError, match="support"):
        dispatcher("open_gripper", {})
    assert all(method != "open_gripper" for method, _ in robot.actions)


def test_hardware_adapter_requires_synchronized_measured_channels():
    class API:
        def __init__(self, result):self.result=result
        def call(self, method, params):return copy.deepcopy(self.result)
    state = sensed()
    obj = dict(time_s=1., object=state.pop("object"))
    robot = MeasuredRobot(None, API(obj), API(state), None, {})
    assert robot.observe()["wrench"]["force_n"] == [0, 0, 2]
    robot.proprioception.result["time_s"] = 5.
    with pytest.raises(Exception, match="clocks differ"):
        robot.observe()
    robot.proprioception.result["time_s"] = 1.
    del robot.proprioception.result["wrench"]
    with pytest.raises(KeyError):
        robot.observe()


def test_prompt_has_no_task_specific_branch_or_preselected_parameter():
    prompt = probe_prompt(dict(goal=dict(instruction="Translate and reorient the object")))
    for forbidden in ("rod_lift", "rod_axis", "com_offset", "15 mm", "binary search", "bisection"):
        assert forbidden not in prompt
    assert "IDENTIFICATION" in prompt
    assert "Translate and reorient" in prompt


def test_learnt_memory_is_api_data_and_not_baked_into_policy():
    robot = Robot()
    target = InstrumentedTarget(robot)
    target.call("target.update_identification", dict(belief=dict(estimates=[dict(name="stiffness", value=20.)])))
    assert target.call("target.describe", {})["identification"]["estimates"][0]["value"] == 20.
    with pytest.raises(Exception, match="explicit learned estimate"):
        target.call("target.update_identification", dict(belief=dict(true_parameters={})))


def test_task_synthesis_rejects_noop_certifier_and_accepts_bounded_candidate():
    with pytest.raises(ValueError, match="no-op certifier"):
        task_candidate("def run(robot):\n    return {'status':'limited','observation':robot.observe()}\n")
    policy = task_candidate("def run(robot):\n    o=robot.observe()\n    robot.goto_pose(o['tcp']['position'],o['tcp']['quaternion_wxyz'],speed=0.1)\n")
    assert policy.api_version == IDENTIFICATION_API_VERSION
    wrapped = task_candidate("def run(robot):\n    def rpc(fn):\n        return fn()\n    rpc(robot.close_gripper)\n")
    assert wrapped.api_version == IDENTIFICATION_API_VERSION
    with pytest.raises(ValueError, match="unavailable measured API"):
        source_policy("def run(robot):\n    fn=robot.apply_force\n    fn(100)\n")


def test_measured_real_target_runs_same_source_for_probe_and_final_policy():
    robot = Robot()
    target = InstrumentedTarget(robot)
    policy = Policy("def run(robot):\n    return {'measured':robot.observe()}\n", api_version=IDENTIFICATION_API_VERSION)
    for method in ("target.execute_probe", "target.execute_policy"):
        response = target.call(method, dict(policy=policy.wire()))
        assert response["status"] == "completed"
        assert response["policy_sha256"] == policy.sha256
        assert response["program_return"]["measured"]["wrench"]["provenance"] == "Measured wrist transducer"
        assert robot.prepared is True


def test_sim_sensor_refresh_tcp_axes_and_wrench_reference_without_object_truth(monkeypatch):
    """Regression for stale split-step FT and fixed gripper mounting rotation."""
    import numpy as np
    from types import SimpleNamespace as NS
    class GeometryOnly:
        rod_half_size = (.15, .015, .015)
        def __getattr__(self, name):
            raise AssertionError("A sensor must not read hidden object physics: "+name)
    data = NS(time=1., body_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
        body_xpos=np.zeros((2, 3)), site_xpos=np.array([[.51, 0, .2]]),
        site_xmat=np.eye(3).reshape(1, 9), qpos=np.zeros(9), qfrc_actuator=np.zeros(9))
    model = NS(body_name2id=lambda name: 0 if name == "robot0_base" else 1,
        sensor_name2id=lambda name: 0, sensor_objid=np.array([0]))
    class Sim:
        def __init__(self):
            self.model, self.data, self.updated = model, data, False
        def forward(self):self.updated=True
    sim = Sim()
    class Robot:
        robot_model = NS(eef_name={"right":"hand"})
        gripper = {"right": NS(important_sensors={"force_ee":"force"})}
        _ref_gripper_joint_pos_indexes = {"right":[7, 8]}
        _ref_joint_pos_indexes = list(range(7))
        _ref_joint_vel_indexes = list(range(7))
        @property
        def ee_force(self):
            return {"right":np.array([0., 0., 3.]) if sim.updated else np.zeros(3)}
        @property
        def ee_torque(self):return {"right":np.zeros(3)}
    suite = NS(sim=sim, robots=[Robot()], model=NS(mujoco_arena=NS(table_offset=[0, 0, .08])))
    env = NS(robosuite_env=suite,
        get_observation=lambda:{"cube_poses":{"primary":np.array([.5, 0, .1, 1., 0, 0, 0])}},
        # Raw eef axes include a fixed 90-degree mounting rotation.
        get_eef_pose=lambda:(np.array([2**-.5, 0, 0, -2**-.5]), np.array([.5, 0, .1])))
    backend = NS(ensure=lambda:(env, None), physics=GeometryOnly())
    from physrsr.active.runtime import simulator
    monkeypatch.setattr(simulator, "observation_provider", lambda env:
        dict(rod_position=[.5, 0, .1], grasp_quat=[0, 0, 1, 0]))
    result = SimulatedMeasuredRobot(backend).observe()
    assert result["wrench"]["force_n"] == [0., 0., 3.]
    assert result["tcp"]["quaternion_wxyz"] == [1., 0., 0., 0.]
    assert result["time_s"] == 1.  # Sensor refresh must not advance physics.
    assert result["wrench"]["torque_nm"] == pytest.approx([0., -.03, 0.])


def test_sim_hold_captures_encoders_instead_of_stale_reset_target():
    import numpy as np
    from types import SimpleNamespace as NS
    robot = NS(_ref_joint_pos_indexes=list(range(7)))
    data = NS(time=0., qpos=np.arange(7, dtype=float))
    env = NS(_current_joints=np.full(7, 99.), _gripper_fraction=.2,
        robosuite_env=NS(robots=[robot], sim=NS(data=data)))
    targets = []
    def step():
        targets.append(env._current_joints.copy())
        data.time += .05
    env._step_once = step
    backend = NS(ensure=lambda:(env, None))
    SimulatedMeasuredRobot(backend).action("hold", dict(duration_s=.1))
    assert len(targets) == 2
    assert all(np.array_equal(target, data.qpos) for target in targets)
    assert env._gripper_fraction == .2


def test_motion_api_uses_bounded_measured_feedback_on_both_backends():
    state = observation(sensed())
    class Control:
        calls = 0
        def goto_pose(self, **params):
            self.calls += 1
            state["tcp"]["position"] = params["position"][:]
            state["tcp"]["position"][2] += .005 if self.calls == 1 else .001
    control = Control()
    params = dict(position=[.5, 0, .12], quaternion_wxyz=[0, 0, 1, 0], z_approach=0., speed=.15)
    diagnostics = move_with_feedback(control, lambda:state, params, [80.]*7)
    assert control.calls == 2
    assert diagnostics[0]["position_error_m"] == pytest.approx(.005)
    assert diagnostics[-1]["position_error_m"] <= .003


def test_known_motion_failure_reaches_guest_for_bounded_recovery():
    from physrsr.coding.runtime import RobotActionError
    source = """def run(robot):
    try:
        robot.goto_pose([0.5,0,0.12],[0,0,1,0],speed=0.15)
    except RuntimeError:
        return {'recovery_observation':robot.observe()}
"""
    def dispatch(method, params):
        if method == "goto_pose":
            raise RobotActionError("Target not attained")
        return {"measured":True}
    result = execute(Policy(source, api_version=IDENTIFICATION_API_VERSION), dispatch)
    assert result["return_value"]["recovery_observation"]["measured"] is True


def test_precision_profile_reaches_shared_sim_real_joint_controller():
    import numpy as np
    from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
    class Environment:
        def move_to_joints_blocking(self, joints, **options):self.options=options
        def get_eef_pose(self):return np.array([1.,0,0,0]),np.array([.5,0,.1])
    control=FrankaControlPrivilegedApi.__new__(FrankaControlPrivilegedApi)
    control._env=Environment()
    control.cfg=None
    control.ik_solve_fn=lambda **kwargs:np.zeros(8)
    control.configure_motion(joint_tolerance_rad=.004,tcp_tolerance_m=.003)
    control._solve_and_move(np.array([1.,0,0,0]),np.array([.5,0,.1]),np.array([.5,0,.1]),max_joint_speed=.3)
    assert control._env.options == dict(tolerance=.004,max_joint_speed=.3)
