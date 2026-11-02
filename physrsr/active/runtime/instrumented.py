"""Measured capability API shared by World adapters and real robot integrations.

Benchmark geometry lives in the trusted adapter, never in the Physic prompt.
No force/torque here is synthesized from an object's mass or inertial centre.
"""
from __future__ import annotations

import copy
import math
import time
import uuid

from ..api.client import APIError
from ..core.contracts import IDENTIFICATION_API_VERSION, finite
from ..core.identification import ROBOT_API
from ..core.policy import Policy, execute
from ..core.measurements import (FEATURE_SCHEMA, SCENE_FEATURE_SCHEMA, FEATURE_NAMES, FEATURE_SCALES,
    response_features, response_schema)
from ..core.scene import objects, object_ids, check_objects, supported
from ..core.exploration import checked_checkpoint, PAUSED_STATUSES, resumable
from .robot import vector, RobotContractError


MOTION_CONTRACT = dict(position_tolerance_m=.003, orientation_tolerance_rad=.03,
    joint_tolerance_rad=.004, max_attempts=3,
    completion="Measured command-TCP convergence or explicit API failure; no best-effort success")


def settle_ready_state(robot, *, max_holds=16):
    """A ready episode has settled measured geometry/TCP/gripper, on either backend."""
    previous=robot.observe();start=previous["time_s"];consecutive=0;diagnostics=[]
    def angle(a,b):
        scale=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
        return 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b)))/scale))
    for index in range(max_holds):
        robot.action("hold",dict(duration_s=.5))
        current=robot.observe()
        if current["time_s"]-previous["time_s"]<.4:
            raise APIError("Episode preparation hold did not advance measured controller time")
        check_objects(current, object_ids(previous))
        stable=(all(math.dist(obj["position"],objects(previous)[key]["position"])<=.001 and
            angle(obj["quaternion_wxyz"],objects(previous)[key]["quaternion_wxyz"])<=.003
            for key,obj in objects(current).items()) and
            math.dist(current["tcp"]["position"],previous["tcp"]["position"])<=.001 and
            angle(current["tcp"]["quaternion_wxyz"],previous["tcp"]["quaternion_wxyz"])<=.003 and
            abs(current["gripper"]["opening_m"]-previous["gripper"]["opening_m"])<=.001)
        diagnostics.append(dict(time_s=current["time_s"],opening_m=current["gripper"]["opening_m"],stable=stable))
        consecutive=consecutive+1 if stable else 0
        if consecutive>=2:
            old=getattr(robot,"setup_cost_s",None)
            if old is not None: robot.setup_cost_s=old+current["time_s"]-start
            robot.last_setup_diagnostics=diagnostics
            return current
        previous=current
    raise APIError("Episode did not reach a measured settled ready state within its setup budget")


def move_with_feedback(control, observe, params, joint_limits, contract=None):
    """Task-independent, bounded convergence; identical on simulation/hardware."""
    contract = MOTION_CONTRACT if contract is None else contract
    from physrsr.coding.runtime import RobotActionError
    configure = getattr(control, "configure_motion", None)
    if configure is not None:
        configure(joint_tolerance_rad=contract["joint_tolerance_rad"],
            tcp_tolerance_m=contract["position_tolerance_m"])
    diagnostics = []
    for attempt in range(int(contract["max_attempts"])):
        control.goto_pose(**params)
        state = observe()
        error = math.dist(state["tcp"]["position"], params["position"])
        q, target = state["tcp"]["quaternion_wxyz"], params["quaternion_wxyz"]
        similarity = abs(sum(a*b for a, b in zip(q, target))) / math.sqrt(sum(x*x for x in q)*sum(x*x for x in target))
        angle = 2*math.acos(min(1., similarity))
        diagnostics.append(dict(attempt=attempt+1, position_error_m=error,
            orientation_error_rad=angle, measured_time_s=state["time_s"]))
        if any(abs(t) >= .85*limit for t, limit in zip(state["joints"]["actuator_torque_nm"], joint_limits)):
            raise RobotActionError("Joint torque headroom exhausted during motion convergence")
        if error <= contract["position_tolerance_m"] and angle <= contract["orientation_tolerance_rad"]:
            return diagnostics
    raise RobotActionError("Measured motion target not attained after bounded feedback: "+str(diagnostics))


def observation(raw):
    """Whitelist measurable channels; private backend fields cannot leak."""
    if raw["coordinate_frame"] != "robot_base":
        raise ValueError("Expected robot_base observations")
    tcp, wrench, joints = (raw[k] for k in ("tcp", "wrench", "joints"))
    if wrench.get("frame") != "robot_base" or wrench.get("about") != "tcp":
        raise ValueError("Wrench must be transformed to robot_base about the measured TCP")
    scene = {key: object_observation(obj) for key,obj in objects(raw).items()}
    # The singular field is a deterministic compatibility view, never a second
    # World or a track that switches when the robot grasps another object.
    result = dict(time_s=finite(raw["time_s"]), coordinate_frame="robot_base",
        object=scene[sorted(scene)[0]],
        tcp=dict(position=vector(tcp["position"], 3), quaternion_wxyz=vector(tcp["quaternion_wxyz"], 4)),
        wrench=dict(force_n=vector(wrench["force_n"], 3), torque_nm=vector(wrench["torque_nm"], 3),
            frame="robot_base", about="tcp", valid=wrench["valid"] is True,
            provenance=str(wrench["provenance"])),
        joints=dict(position_rad=vector(joints["position_rad"], 7),
            actuator_torque_nm=vector(joints["actuator_torque_nm"], 7)),
        gripper=dict(opening_m=finite(raw["gripper"]["opening_m"])))
    if "objects" in raw:
        result["objects"] = scene
    return result


def object_observation(obj):
    if len(obj["axes"]) != 3:
        raise ValueError("Object axes require three body-frame directions")
    return dict(position=vector(obj["position"], 3), quaternion_wxyz=vector(obj["quaternion_wxyz"], 4),
        axes=[vector(a, 3) for a in obj["axes"]], dimensions_m=vector(obj["dimensions_m"], 3),
        grasp_quaternion_wxyz=vector(obj["grasp_quaternion_wxyz"], 4),
        support_clearance_m=finite(obj["support_clearance_m"]))


class MeasuredRobot:
    """Hardware counterpart: explicit measured perception/proprioception APIs.

    perception.object supplies the object fields in `observation`. Proprioception
    robot.state supplies TCP, raw TCP-referenced wrench, joint torques, gripper,
    a synchronized capture time and frame. No default estimated sensor values.
    setup and hold are provided by the lab controller API, not simulated waits.
    """
    def __init__(self, control, perception, proprioception, setup, capabilities):
        self.control, self.perception, self.proprioception, self.setup = control, perception, proprioception, setup
        self.capabilities = copy.deepcopy(capabilities)
        self.belief = {}

    def describe(self):
        capabilities = copy.deepcopy(self.capabilities)
        capabilities.setdefault("motion_contract", copy.deepcopy(MOTION_CONTRACT))
        return dict(api_version=IDENTIFICATION_API_VERSION, robot_api=ROBOT_API,
            **capabilities, identification=copy.deepcopy(self.belief))

    def observe(self):
        obj = self.perception.call("perception.object", {})
        state = self.proprioception.call("robot.state", {})
        if abs(finite(obj["time_s"])-finite(state["time_s"])) > .1:
            raise APIError("Object and proprioception clocks differ by more than 0.1 s")
        geometry = dict(objects=obj["objects"]) if "objects" in obj else dict(object=obj["object"])
        return observation(dict(**state, **geometry))

    def prepare(self):
        response = self.setup.call("episode.prepare", {})
        expected = self.capabilities["gripper"]["normal_force_limit_per_finger_n"]
        if response.get("ready") is not True or response.get("normal_force_limit_per_finger_n") != expected:
            raise APIError("Setup must acknowledge readiness and the unchanged gripper force limit")
        self.setup_cost_s = response.get("setup_cost_s")
        if self.setup_cost_s is not None and finite(self.setup_cost_s) < 0:
            raise APIError("Negative measured setup cost")

    def action(self, method, params):
        if method == "goto_pose":
            self.last_motion = move_with_feedback(self.control, self.observe, params,
                self.capabilities["arm"]["joint_torque_limits_nm"],
                self.capabilities.get("motion_contract", MOTION_CONTRACT))
        elif method == "hold":
            # A real controller holds commanded joints/fingers while sampling
            # encoders/FT; a wall-clock sleep is not an implementation of hold.
            before = self.observe()["time_s"]
            self.proprioception.call("robot.hold", params)
            after = self.observe()["time_s"]
            if after-before < params["duration_s"]*.8:
                raise APIError("Hardware hold did not advance measured controller time")
        else:
            getattr(self.control, method)(**params)


class Dispatcher:
    def __init__(self, robot, initial, control=None, program_sha256=None):
        self.robot, self.initial, self.control = robot, initial, control
        self.calls, self.samples = 0, [dict(event="initial", observation=initial)]
        self.program_sha256=program_sha256
        self.last_checkpoint=None
        self.actions=0
        declared=robot.describe().get("experiment_budget",{}).get("max_rpc_calls",192)
        if isinstance(declared,bool) or not isinstance(declared,int) or not 1<=declared<=192:
            raise RobotContractError("Invalid declared RPC budget")
        self.max_calls=declared
        self.scene_ids=object_ids(initial)
        self.workspace=robot.describe().get("workspace", {})
        if "objects" in initial:
            # A scene can extend beyond the old 25 cm single-object workspace.
            # Its trusted robot adapter must publish the actual shared bounds.
            if not {"min_position_m","max_position_m"} <= self.workspace.keys():
                raise RobotContractError("Scene robot must declare shared workspace min_position_m/max_position_m")
            self.workspace_min=vector(self.workspace["min_position_m"],3)
            self.workspace_max=vector(self.workspace["max_position_m"],3)
            if any(a>=b for a,b in zip(self.workspace_min,self.workspace_max)):
                raise RobotContractError("Invalid scene workspace bounds")

    def safe_boundary(self):
        current=self.robot.observe()
        earlier=next((s["observation"] for s in reversed(self.samples)
            if current["time_s"]-s["observation"]["time_s"]>=.1),None)
        if earlier is None:return False
        cap=self.robot.describe();tol=cap.get("motion_contract",MOTION_CONTRACT)
        check_objects(current,self.scene_ids)
        check_objects(earlier,self.scene_ids)
        def stable(a,b):
            qa,qb=a["quaternion_wxyz"],b["quaternion_wxyz"]
            similarity=abs(sum(x*y for x,y in zip(qa,qb)))/math.sqrt(sum(x*x for x in qa)*sum(x*x for x in qb))
            return (math.dist(a["position"],b["position"])<=tol["position_tolerance_m"] and
                2*math.acos(min(1.,similarity))<=tol["orientation_tolerance_rad"])
        return (supported(current) and
            all(stable(obj,objects(earlier)[key]) for key,obj in objects(current).items()) and
            math.dist(current["tcp"]["position"],earlier["tcp"]["position"])<=tol["position_tolerance_m"] and
            current["gripper"]["opening_m"]>=cap["gripper"]["max_opening_m"]-.003)

    def checkpoint_result(self, status, returned):
        if self.last_checkpoint is None:return None
        result=copy.deepcopy(self.last_checkpoint)
        result["resumable"]=(status=="completed" and isinstance(returned,dict)
            and returned.get("status") in PAUSED_STATUSES and self.actions==result["action_sequence"]
            and self.safe_boundary())
        return result

    def __call__(self, method, params):
        self.calls += 1
        if self.calls > self.max_calls or not isinstance(params, dict):
            raise RobotContractError("Identification RPC budget/arguments invalid")
        if method == "describe" and not params:
            description=copy.deepcopy(self.robot.describe())
            memory=description.get("identification",{})
            checkpoint=memory.get("probe_progress",{}).get(self.program_sha256,{})
            description["probe_execution"]=dict(program_sha256=self.program_sha256,
                remaining_rpc_calls=self.max_calls-self.calls,max_rpc_calls=self.max_calls,
                resume_state=copy.deepcopy(checkpoint.get("state")) if resumable(memory,self.program_sha256) else None)
            return description
        if method=="checkpoint" and set(params)=={"state"}:
            scope=self.robot.describe().get("identification",{}).get("instance_id")
            if not scope or not self.program_sha256 or not self.safe_boundary():
                raise RobotContractError("Checkpoint requires stable measured support and an open gripper")
            try:state=checked_checkpoint(params["state"])
            except (ValueError,TypeError) as exc:raise RobotContractError(str(exc)) from exc
            self.last_checkpoint=dict(program_sha256=self.program_sha256,state=state,
                instance_id=scope,
                measured_time_s=self.robot.observe()["time_s"],action_sequence=self.actions,resumable=False)
            self.samples.append(dict(event="checkpoint",params=dict(state=state),observation=self.robot.observe()))
            return dict(saved=True,program_sha256=self.program_sha256)
        if method == "observe" and not params:
            result = self.robot.observe()
        else:
            if method == "goto_pose":
                if set(params) != {"position", "quaternion_wxyz", "z_approach", "speed"}:
                    raise RobotContractError("Invalid pose arguments")
                p, q = vector(params["position"], 3), vector(params["quaternion_wxyz"], 4)
                centre = self.initial["object"]["position"]
                outside=(any(not lo<=v<=hi for v,lo,hi in zip(p,self.workspace_min,self.workspace_max))
                    if "objects" in self.initial else
                    math.dist(p[:2],centre[:2])>.25 or not -.02<=p[2]-centre[2]<=.2)
                if (outside
                        or abs(sum(x*x for x in q)-1) > .02 or not 0 <= finite(params["z_approach"]) <= .12
                        or (params["speed"] is not None and not .1 <= finite(params["speed"]) <= 1)):
                    raise RobotContractError("Workspace, quaternion or speed constraint violated")
            elif method == "hold" and set(params) == {"duration_s"}:
                if not 0 < finite(params["duration_s"]) <= 2:
                    raise RobotContractError("Hold duration outside (0,2] seconds")
            elif method in ("open_gripper", "close_gripper") and not params:
                if method == "open_gripper" and not supported(self.robot.observe()):
                    raise RobotContractError("Release requires measured support proximity")
            else:
                raise RobotContractError("Unavailable identification API method")
            from physrsr.coding.runtime import RobotActionError
            self.actions+=1
            try:
                if self.control is None:
                    self.robot.action(method, params)
                else:
                    self.robot.action(method, params, control=self.control)
            except RobotActionError as exc:
                self.samples.append(dict(event=method, params=copy.deepcopy(params),
                    observation=self.robot.observe(), action_error=str(exc)))
                raise
            result = self.robot.observe()
        sample = dict(event=method, params=copy.deepcopy(params), observation=result)
        check_objects(result,self.scene_ids)
        if method == "goto_pose":
            sample["motion_feedback"] = copy.deepcopy(getattr(self.robot, "last_motion", []))
        self.samples.append(sample)
        return result if method == "observe" else None


class InstrumentedProgram:
    def __init__(self, policy, robot):
        self.id, self.policy, self.robot = policy.sha256, policy, robot
        self.runtime, self.dispatcher, self.error_kind = {}, None, None

    def run(self, env, control, physics=None):
        settle_ready_state(self.robot)
        self.dispatcher = Dispatcher(self.robot, self.robot.observe(), control,program_sha256=self.policy.sha256)
        from physrsr.coding.runtime import ProgramError
        try:
            self.runtime = execute(self.policy, self.dispatcher)
        except (ProgramError, RobotContractError):
            self.error_kind = "program"
            raise
        except Exception:
            self.error_kind = "infrastructure"
            raise


class InstrumentedTarget:
    """Probe and task API; belief memory is supplied only by measured inference."""
    def __init__(self, robot):
        self.robot = robot
        self.instance_scope=getattr(robot,"belief",{}).get("instance_id") or uuid.uuid4().hex
        self.robot.belief=dict(getattr(robot,"belief",{}),instance_id=self.instance_scope)

    def call(self, method, params):
        if method == "target.prepare" and not params:
            self.robot.prepare()
            return settle_ready_state(self.robot)
        if method == "target.describe" and not params:
            return dict(self.robot.describe(), feature_schema=FEATURE_SCHEMA,
                supported_feature_schemas=[FEATURE_SCHEMA,SCENE_FEATURE_SCHEMA],
                feature_names=list(FEATURE_NAMES), feature_scales=list(FEATURE_SCALES))
        if method == "target.observe" and not params:
            return self.robot.observe()
        if method == "target.update_identification" and set(params) == {"belief"}:
            # The robot sees learned estimates, never evaluator parameters.
            belief = params["belief"]
            if not isinstance(belief, dict) or not isinstance(belief.get("estimates"), list):
                raise APIError("Expected an explicit learned estimate list")
            scope=belief.get("instance_id",self.instance_scope)
            if not isinstance(scope,str) or not scope:raise APIError("Invalid identification instance scope")
            self.instance_scope=scope
            self.robot.belief = dict(copy.deepcopy(belief),instance_id=scope)
            return dict(updated=True)
        if method not in ("target.execute_probe", "target.execute_policy") or set(params) != {"policy"}:
            raise APIError("Unknown measured target operation")
        policy = Policy.from_wire(params["policy"])
        if policy.api_version != IDENTIFICATION_API_VERSION:
            raise APIError("Expected versioned identification API")
        self.robot.prepare()
        settle_ready_state(self.robot)
        dispatcher = Dispatcher(self.robot, self.robot.observe(),program_sha256=policy.sha256)
        start = time.monotonic()
        from physrsr.coding.runtime import ProgramError
        status, error, runtime = "completed", None, {}
        try:
            runtime = execute(policy, dispatcher)
        except (ProgramError, RobotContractError) as exc:
            status, error = "program_error", str(exc)
        dispatcher.samples.append(dict(event="terminal", observation=self.robot.observe()))
        return dict(policy_sha256=policy.sha256, status=status, error=error, runtime=runtime,
            program_return=runtime.get("return_value"), rpc_calls=dispatcher.calls,
            experiment_checkpoint=dispatcher.checkpoint_result(status,runtime.get("return_value")),
            observations=dispatcher.samples, wall_s=time.monotonic()-start,
            cost_s=dispatcher.samples[-1]["observation"]["time_s"]-dispatcher.initial["time_s"],
            setup_cost_s=getattr(self.robot,"setup_cost_s",None),
            evidence_id=uuid.uuid4().hex, feature_schema=response_schema(dispatcher.initial),
            features=response_features(dispatcher.samples))
