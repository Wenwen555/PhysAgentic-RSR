"""Public API, immutable policy and offline CLI checks. No physical rollout."""
import io
import json
from pathlib import Path

import pytest

from physrsr.active.api.client import APIError, HTTPClient
from physrsr.active.core.contracts import API_VERSION, LoopConfig, read_config
from physrsr.active.core.policy import Policy, PolicyStore
from physrsr.active.runtime.robot import RobotContractError, RobotDispatcher, RobotService, TargetService, public_observation


def observation():
    return dict(rod_position=[.2, .1, .1], rod_axis=[1, 0, 0], grasp_quat=[0, 0, 1, 0],
                tcp_position=[.2, .1, .1], coordinate_frame="robot_base", time_s=1., hidden_mass=999.)


class Control:
    def __init__(self):
        self.calls = []
    def open_gripper(self):
        self.calls.append("open")
    def close_gripper(self):
        self.calls.append("close")
    def goto_pose(self, **params):
        self.calls.append(params)


def test_same_policy_uses_same_rpc_and_hash_for_sim_and_real_adapters():
    policy = Policy("def run(robot):\n    robot.observe()\n    robot.open_gripper()\n    robot.close_gripper()\n")
    results, controls = [], []
    for backend in ("sim", "real"):
        control = Control()
        controls.append(control)
        # The names denote adapter slots, not real devices. This uses the
        # production namespace sandbox with only deterministic API fixtures.
        service = TargetService(RobotService(control, observation, lambda: dict(cost_s=2.)))
        results.append(service.call("target.execute_policy", dict(policy=policy.wire())))
    assert [r["status"] for r in results] == ["completed", "completed"]
    assert [r["policy_sha256"] for r in results] == [policy.sha256, policy.sha256]
    assert controls[0].calls == controls[1].calls == ["open", "close"]
    assert results[0]["cost"] == 2.
    assert "hidden_mass" not in json.dumps(results)


def test_active_executor_retains_labeled_json_return_and_legacy_default_is_unchanged():
    from physrsr.coding.runtime import execute_source
    policy = Policy("def run(robot):\n    o=robot.observe()\n    return {'records':[{'label':'baseline','observation':o}]}\n")
    service = TargetService(RobotService(Control(), observation, lambda: None))
    result = service.call("target.execute_policy", dict(policy=policy.wire()))
    assert result["program_return"]["records"][0]["label"] == "baseline"
    assert result["program_return"]["records"][0]["observation"]["time_s"] == 1.
    assert execute_source("def run(robot):\n    return object()\n", lambda method, params: None) == {"calls": 0}


@pytest.mark.parametrize("source", [
    "def main(robot): pass", "def run(robot, env): pass", "import os\ndef run(robot): pass",
    "def run(robot):\n    return robot._env", "def run(robot):\n    open('/tmp/file')"])
def test_policy_contract_rejects_backend_access(source):
    with pytest.raises(ValueError):
        Policy(source)


def test_hash_api_mismatch_and_immutable_revision_store(tmp_path):
    policy = Policy("def run(robot):\n    robot.observe()\n")
    with pytest.raises(ValueError, match="hash/API"):
        Policy.from_wire(dict(policy.wire(), source="def run(robot): pass"))
    store = PolicyStore(tmp_path / "policies")
    store.add(policy, 0)
    revision = Policy(policy.source + "# change\n", parent=policy.sha256)
    store.add(revision, 1)
    assert (store.root / revision.sha256 / "revision.diff").exists()
    with pytest.raises(FileExistsError):
        PolicyStore(store.root)


def test_observation_whitelist_and_frame_validation():
    assert "hidden_mass" not in public_observation(observation())
    with pytest.raises(ValueError, match="robot_base"):
        public_observation(dict(observation(), coordinate_frame="mujoco_world"))
    with pytest.raises(ValueError, match="finite"):
        public_observation(dict(observation(), rod_position=[float("nan"), 0, 0]))


def test_workspace_violation_is_rejected_before_actuation():
    control = Control()
    dispatcher = RobotDispatcher(RobotService(control, observation, lambda: None), observation())
    with pytest.raises(RobotContractError):
        dispatcher("goto_pose", dict(position=[999, 0, 0], quaternion_wxyz=[1, 0, 0, 0], z_approach=0, speed=.3))
    assert control.calls == []


def test_http_failure_is_not_retried(monkeypatch):
    calls = []
    def fail(request, timeout):
        calls.append(request)
        raise OSError("acknowledgement lost")
    monkeypatch.setattr("physrsr.active.api.client.urlopen", fail)
    with pytest.raises(OSError):
        HTTPClient("http://127.0.0.1:9999").call("target.execute_probe", {})
    assert len(calls) == 1


def test_http_serializes_policy_without_modifying_source(monkeypatch):
    policy = Policy("def run(robot):\n    robot.observe()\n")
    def response(request, timeout):
        data = json.loads(request.data)
        assert data["params"]["policy"] == policy.wire()
        return io.BytesIO(b'{"result":{"status":"completed"}}')
    monkeypatch.setattr("physrsr.active.api.client.urlopen", response)
    assert HTTPClient("http://127.0.0.1:9999").call("target.execute_policy", dict(policy=policy.wire()))["status"] == "completed"


def test_offline_example_config_without_api_calls(monkeypatch, capsys):
    from physrsr.active.__main__ import main
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline checks must never call an API")
    monkeypatch.setattr(HTTPClient, "call", forbidden)
    from physrsr.active.core.paths import PACKAGE_ROOT
    example = PACKAGE_ROOT / "examples/rod_lift.json"
    assert main(["validate-config", str(example)]) == 0
    assert json.loads(capsys.readouterr().out)["external_calls"] == 0


@pytest.mark.parametrize("kwargs", [dict(max_rounds=-1), dict(feature_scales=[0]*5), dict(max_candidates=1),
                                   dict(max_probe_cost=float("nan")), dict(min_validation_probes=0)])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        LoopConfig(**kwargs)


def test_real_adapter_uses_tracking_api_and_measured_tcp():
    from physrsr.active.runtime.backends import measured_franka_robot
    class Poses:
        def tolist(self):
            return [.4, .3, .2]
    class Env:
        def get_observation(self):
            return {}
        def get_eef_pose(self):
            return [1, 0, 0, 0], Poses()
    class Tracker:
        def call(self, method, params):
            assert method == "perception.rod"
            return observation()
    class Setup:
        def call(self, method, params):
            return dict(ready=True, cost_s=3., grip_force_per_finger_n=2.)
    service = measured_franka_robot(Control(), Env(), Tracker(), Setup())
    value = service.call("episode.prepare", {})
    assert value["tcp_position"] == [.4, .3, .2]
    assert value["setup_cost_s"] == 3.
    assert "hidden_mass" not in value


def test_simulator_program_error_is_feedback_without_physics_rollout(monkeypatch):
    from types import SimpleNamespace as NS
    from physrsr.active.runtime.simulator import RodSimulatorService, ROD_SUCCESS
    import physrsr.active.runtime.simulator as module
    from physrsr.physics_spec import RolloutRecord
    env = NS(robosuite_env=NS(ignore_done=False, sim=NS(data=NS(time=2., body_xpos=[[0, 0, .2]])),
                             cube_body_id=0, model=NS(mujoco_arena=NS(table_offset=[0, 0, .1]))))
    closed = []
    class Worker:
        def __init__(self, physics):
            self.physics = physics
        def _ensure_env(self):
            return env, None
        def rollout(self, program, seed, role):
            program.dispatcher = NS(initial=dict(time_s=1.), unsafe_release=False,
                                    samples=[dict(event="initial", observation=public_observation(observation()))])
            program.error_kind = "program"
            return RolloutRecord("model", program.id, seed, role, "error", reason="invalid generated code")
        def close(self):
            closed.append(True)
    monkeypatch.setattr(module, "observation_provider", lambda env: public_observation(observation()))
    policy = Policy("def run(robot):\n    robot.observe()\n")
    result = RodSimulatorService(worker_factory=Worker).call("sim.evaluate", dict(
        task=dict(id="rod_lift", success=ROD_SUCCESS, constraints=dict(grip_force_per_finger_n=2.)),
        models=[dict(id="m", parameters=dict(com_offset=0., friction=1.5), weight=1.)],
        seeds=[1], policies=[policy.wire()]))
    assert result["rows"][0]["success_rate"] == 0.
    assert result["rows"][0]["duration_s"] == 1.
    assert result["rows"][0]["feedback"][0]["error_kind"] == "program"
    assert closed == [True]


def test_structured_coding_keeps_revision_context_without_inference(tmp_path):
    from physrsr.active.agents.coding import CodingService, PROPOSAL_SCHEMA
    class Agent:
        def generate(self, messages, destination, *, schema):
            assert schema == PROPOSAL_SCHEMA
            assert "previous_policy" in messages[0]["content"]
            return dict(policies=[], probes=[], uncertainties=[])
    service = CodingService(tmp_path / "coding", agent=Agent())
    assert service.call("agent.propose", dict(previous_policy="same source"))["probes"] == []
