"""Public observation and control adapters, shared by simulation and hardware.

RobotService accepts an existing control API and measured pose provider. Neither
constructing an adapter nor importing this module connects to a physical arm.
"""
from __future__ import annotations

import math
import time
import uuid

from ..api.client import APIError
from ..core.contracts import API_VERSION, finite
from ..core.policy import Policy, execute


class RobotContractError(ValueError):
    pass


def vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("Invalid public observation vector")
    return [finite(x) for x in value]


def public_observation(raw):
    if raw.get("coordinate_frame") != "robot_base":
        raise ValueError("Observations must explicitly use robot_base")
    result = {key: vector(raw[key], n) for key, n in (
        ("rod_position", 3), ("rod_axis", 3), ("grasp_quat", 4), ("tcp_position", 3))}
    for key in ("rod_axis", "grasp_quat"):
        if abs(math.sqrt(sum(x*x for x in result[key])) - 1) > .01:
            raise ValueError(f"Observation {key} must be normalized")
    return dict(**result, coordinate_frame="robot_base", time_s=finite(raw["time_s"]))


class RobotService:
    """Mount around sim or real FrankaControlPrivilegedApi and a pose tracker.

    observe_fn supplies measured rod pose, grasp orientation and measured TCP;
    prepare_fn establishes a repeatable, supported task start (on hardware it
    can call the lab's setup API). There is no automatic fake/reset fallback.
    """
    def __init__(self, control, observe_fn, prepare_fn):
        self.control, self.observe_fn, self.prepare_fn = control, observe_fn, prepare_fn

    def call(self, method, params):
        if method == "episode.prepare" and not params:
            setup = self.prepare_fn() or {}
            return dict(**public_observation(self.observe_fn()), setup_cost_s=finite(setup.get("cost_s", 0)))
        if method == "robot.observe" and not params:
            return public_observation(self.observe_fn())
        if method in ("robot.open_gripper", "robot.close_gripper") and not params:
            getattr(self.control, method.split(".")[1])()
            return None
        if method == "robot.goto_pose" and set(params) == {"position", "quaternion_wxyz", "z_approach", "speed"}:
            self.control.goto_pose(**params)
            return None
        raise APIError(f"Unknown robot API operation: {method}")


class RobotDispatcher:
    def __init__(self, robot, initial):
        self.robot, self.initial = robot, public_observation(initial)
        self.samples = [dict(event="initial", observation=self.initial)]
        self.calls = 0
        self.unsafe_release = False

    def __call__(self, method, params):
        self.calls += 1
        if self.calls > 96 or not isinstance(params, dict):
            raise RobotContractError("Robot RPC budget or argument violation")
        if method == "observe" and not params:
            result = public_observation(self.robot.call("robot.observe", {}))
        elif method in ("open_gripper", "close_gripper") and not params:
            before = public_observation(self.robot.call("robot.observe", {}))
            if method == "open_gripper" and before["rod_position"][2] - self.initial["rod_position"][2] > .02:
                self.unsafe_release = True
                raise RobotContractError("Release requires a supported rest state")
            self.robot.call("robot." + method, {})
            result = public_observation(self.robot.call("robot.observe", {}))
        elif method == "goto_pose" and set(params) == {"position", "quaternion_wxyz", "z_approach", "speed"}:
            try:
                p = vector(params["position"], 3)
                q = vector(params["quaternion_wxyz"], 4)
                z = finite(params["z_approach"])
                speed = params["speed"]
                if abs(math.sqrt(sum(x*x for x in q))-1) > .01 or not 0 <= z <= .12:
                    raise ValueError()
                centre = self.initial["rod_position"]
                if any(abs(p[i]-centre[i]) > .25 for i in (0, 1)) or not -.02 <= p[2]-centre[2] <= .20:
                    raise ValueError()
                if speed is not None and not .1 <= finite(speed) <= 1:
                    raise ValueError()
            except (ValueError, TypeError):
                raise RobotContractError("Invalid pose, workspace, quaternion or speed") from None
            self.robot.call("robot.goto_pose", params)
            result = public_observation(self.robot.call("robot.observe", {}))
        else:
            raise RobotContractError(f"Unavailable policy API: {method}")
        self.samples.append(dict(event=method, observation=result))
        # Existing generated-code guest uses the original four-field contract.
        return result if method == "observe" else None


def response_features(samples):
    observations = [s["observation"] for s in samples]
    first, last = observations[0], observations[-1]
    rise = last["rod_position"][2] - first["rod_position"][2]
    relative_z = rise - (last["tcp_position"][2] - first["tcp_position"][2])
    return [max(o["rod_axis"][2] for o in observations),
            min(o["rod_axis"][2] for o in observations), rise, relative_z,
            max(o["rod_position"][2]-first["rod_position"][2] for o in observations)]


class TargetService:
    """No target success predicate or target physics is exposed to the loop."""
    def __init__(self, robot, runner=execute):
        self.robot, self.runner = robot, runner

    def call(self, method, params):
        if method == "target.describe" and not params:
            return dict(api_version=API_VERSION, feature_schema="rod_response_v1")
        if method == "target.observe" and not params:
            return public_observation(self.robot.call("robot.observe", {}))
        if method not in ("target.execute_probe", "target.execute_policy"):
            raise APIError(f"Unknown target operation: {method}")
        policy = Policy.from_wire(params["policy"])
        prepared = self.robot.call("episode.prepare", {})
        initial = public_observation(prepared)
        setup_cost = finite(prepared.get("setup_cost_s", 0))
        if setup_cost < 0:
            raise APIError("Negative setup cost")
        dispatcher = RobotDispatcher(self.robot, initial)
        start = time.monotonic()
        from physrsr.coding.runtime import ProgramError
        status, error, runtime = "completed", None, {}
        try:
            runtime = self.runner(policy, dispatcher) or {}
        except (ProgramError, RobotContractError) as exc:
            status, error = "program_error", str(exc)
        observations = dispatcher.samples
        # One final *measured* sample, also on program errors; provider failures
        # remain infrastructure errors and cannot become physical observations.
        observations.append(dict(event="terminal", observation=public_observation(
            self.robot.call("robot.observe", {}))))
        cost = observations[-1]["observation"]["time_s"] - initial["time_s"]
        if cost < 0:
            raise APIError("Robot observation clock moved backwards")
        if method == "target.execute_probe" and abs(response_features(observations)[2]) > .015:
            status, error = "invalid_probe_terminal_state", "Probe did not restore a supported start state"
        return dict(status=status, error=error, policy_sha256=policy.sha256,
                    program_return=runtime.get("return_value"),
                    evidence_id=uuid.uuid4().hex, features=response_features(observations),
                    feature_schema="rod_response_v1", cost=cost+setup_cost, setup_cost_s=setup_cost,
                    wall_s=time.monotonic()-start,
                    observations=observations, rpc_calls=dispatcher.calls)

    def close(self):
        close = getattr(self.robot, "close", None)
        if close:
            close()
