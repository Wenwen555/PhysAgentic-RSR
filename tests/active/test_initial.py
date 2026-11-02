"""Initial CaP authoring only: API fixtures, no model calls or physics."""
import copy
import json

import pytest

from physrsr.active.api.client import APIError
from physrsr.active.agents.coding import CodingService
from physrsr.active.core.contracts import Task
from physrsr.active.workflows.initial_cap import (INITIAL_CAP_SCHEMA, generate_initial_cap,
    initial_cap_prompt, initial_context, policy_source)

SOURCE = "def step_observe(robot):\n    return robot.observe()\n\ndef run(robot):\n    return step_observe(robot)\n"
TASK = Task("rod_lift", "Lift the rod", {"rise": .1}, {"grip_force_per_finger_n": 2.})


def observation():
    return dict(rod_position=[.2, 0, .1], rod_axis=[1, 0, 0], grasp_quat=[0, 0, 1, 0],
        tcp_position=[.2, 0, .1], coordinate_frame="robot_base", time_s=1., hidden_friction=999.)


def payload(source=SOURCE):
    return dict(mode="policy", command="", files=[dict(path="policy.py", source=source)])


def test_initial_authoring_sees_only_public_task_and_observation(tmp_path):
    calls = []
    class Target:
        def call(self, method, params):
            calls.append(method)
            assert method == "target.observe" and params == {}
            return observation()
    class Agent:
        def call(self, method, params):
            calls.append(method)
            assert method == "agent.generate_initial_cap"
            assert set(params) == {"task", "observation"}
            assert "hidden_friction" not in json.dumps(params)
            return payload()
    result = generate_initial_cap(TASK, target=Target(), agent=Agent(), output=tmp_path / "initial")
    assert calls == ["target.observe", "agent.generate_initial_cap"]
    assert (tmp_path / "initial/bundle/policy.py").read_bytes() == result.source.encode()
    manifest = json.loads((tmp_path / "initial/manifest.json").read_text())
    assert manifest["status"] == "authored"
    assert manifest["simulator_executed"] is False
    assert manifest["target_task_executed"] is False
    assert manifest["sha256"] == result.sha256


def test_initial_prompt_is_stage3_style_and_rejects_model_context():
    context = initial_context(TASK.public(), observation())
    prompt = initial_cap_prompt(context)
    for section in ("TASK", "STAGE", "INITIAL PUBLIC OBSERVATION", "OUTPUT", "RULES", "API", "EXECUTION"):
        assert section in prompt
    assert "Lift the rod" in prompt
    assert "def run(robot)" in prompt
    assert '"path":"policy.py"' in prompt
    assert "hidden_friction" not in prompt
    with pytest.raises(ValueError, match="only task"):
        initial_cap_prompt(dict(context, models=[{"friction": 1.5}]))


@pytest.mark.parametrize("response", [
    dict(mode="practice", command="observe", files=[]),
    dict(mode="execute", command="run_policy", files=[]),
    dict(mode="policy", command="", files=[dict(path="../policy.py", source=SOURCE)]),
    dict(mode="policy", command="", files=[dict(path="policy.py", source=SOURCE), dict(path="extra.py", source=SOURCE)]),
])
def test_initial_output_cannot_request_execution_or_other_files(response):
    with pytest.raises(ValueError):
        policy_source(response)


def test_invalid_source_is_repaired_before_return_without_execution(tmp_path):
    class Agent:
        def __init__(self):
            self.calls = []
        def generate(self, messages, destination, *, schema):
            self.calls.append(copy.deepcopy(messages))
            assert schema == INITIAL_CAP_SCHEMA
            return payload("def run(:") if len(self.calls) == 1 else payload()
    agent = Agent()
    service = CodingService(tmp_path / "coding", agent=agent)
    result = service.call("agent.generate_initial_cap", initial_context(TASK.public(), observation()))
    assert policy_source(result) == SOURCE
    assert len(agent.calls) == 2
    assert "Invalid policy.py" in agent.calls[1][-1]["content"]
    assert "no execution has occurred" in agent.calls[1][-1]["content"]
    assert (tmp_path / "coding/turn_0002/policy.py").read_text() == SOURCE


def test_initial_authoring_failure_has_a_finite_repair_budget(tmp_path):
    class Agent:
        def __init__(self):
            self.count = 0
        def generate(self, messages, destination, *, schema):
            self.count += 1
            return payload("def run(:")
    agent = Agent()
    service = CodingService(tmp_path / "coding", agent=agent)
    with pytest.raises(APIError, match="three attempts"):
        service.call("agent.generate_initial_cap", initial_context(TASK.public(), observation()))
    assert agent.count == 3
