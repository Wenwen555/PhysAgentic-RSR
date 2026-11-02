"""Evidence and authoring contracts; synthetic sensor replies, no dynamics."""
import json
import math
from pathlib import Path

import pytest

from physrsr.active.core.contracts import API_VERSION, Task
from physrsr.active.workflows.evidence import episode_evidence, prepare_initial_evidence, replay_commands
from physrsr.active.agents.physic import (DEFAULT_PROBE_BUDGET, PHYSIC_SCHEMA,
    PhysicService, validate_proposal)
from physrsr.active.workflows.probe_authoring import author_active_probe
from physrsr.active.core.policy import Policy


SOURCE = """def run(robot):
    observation = robot.observe()
    robot.goto_pose(observation['rod_position'], observation['grasp_quat'], speed=0.2)
    robot.observe()
"""


def observation(time=1., z=.1, tilt=0.):
    return dict(rod_position=[.2, 0, z], rod_axis=[math.sqrt(1-tilt*tilt), 0, tilt],
        grasp_quat=[0, 0, 1, 0], tcp_position=[.2, 0, z], time_s=time,
        coordinate_frame="robot_base", secret_friction=999.)


def outcome():
    return dict(status="fail", reason="not_lifted", compiled_inertial_check=dict(com_body_m=[.08, 0, 0]),
        task_evaluation=dict(grasp_established=True), observations=[
            dict(event="initial", observation=observation()),
            dict(event="observe", observation=observation()),
            dict(event="goto_pose", observation=observation(2., .12, .30)),
            dict(event="observe", observation=observation(2., .12, .30)),
            dict(event="terminal", observation=observation(2.5))])


def fixture_experiment(tmp_path):
    root = tmp_path / "experiment"
    (root / "initial_cap/bundle").mkdir(parents=True)
    policy = Policy(SOURCE)
    (root / "initial_cap/bundle/policy.py").write_text(SOURCE)
    (root / "initial_cap/manifest.json").write_text(json.dumps(dict(status="authored", sha256=policy.sha256, api_version=API_VERSION)))
    task = Task("rod_lift", "Lift and hold", dict(max_abs_axis_z=.25), dict(grip_force_per_finger_n=2.))
    config = dict(task=task.public(), policy_sha256=policy.sha256,
        friction=999., mass_kg=888., cases=[dict(name="com_plus8cm", com_offset=.08)])
    (root / "config.json").write_text(json.dumps(config))
    (root / "com_plus8cm").mkdir()
    (root / "com_plus8cm/rollout.json").write_text(json.dumps(dict(rows=[dict(policy_sha256=policy.sha256, feedback=[outcome()])]), allow_nan=False))
    return root


def response(context):
    sample = context["episodes"][0]["trajectory"][0]["evidence_id"]
    candidate = dict(id="low_lift", description="Small observation action", target_factors=["effective balance"],
        expected_discriminating_observations=["Signed pose response"], policy_decision_if_outcomes_differ="Change grasp",
        expected_cost_seconds=10., estimated_rpc_calls=20, qualitative_information_value="high",
        risks=["Tracking uncertainty"], selection_tradeoff="Low cost")
    return dict(observed_facts=[dict(statement="Tilt was observed", evidence_ids=[sample])],
        task_relevant_uncertainties=[dict(factor="effective balance", task_decision_affected="Grasp location",
            supporting_evidence_ids=[sample], alternative_explanations=["Contact alignment"], identifiable_with_current_api="Only effective response")],
        candidate_probes=[candidate, dict(candidate, id="alternative")], selected_probe_id="low_lift",
        selection_reason="Decision relevance", code="def run(robot):\n    robot.observe()\n",
        observation_plan=[dict(label="start", observable_fields=["rod_axis"], meaning="Pose")],
        interpretation_rules=[dict(condition="Tilt changes", hypotheses_supported="Imbalance", hypotheses_not_resolved="Friction", policy_change="Grasp")],
        stop_and_recovery=["Restore support before opening"], limitations=["Not physically executed"],
        status="authored_not_executed", information_value_kind="qualitative_not_validated_evsi")


def test_replay_reconstructs_arguments_and_rejects_divergence():
    commands = replay_commands(Policy(SOURCE), outcome()["observations"])
    assert [c["method"] for c in commands] == ["observe", "goto_pose", "observe"]
    assert commands[1]["params"]["speed"] == .2
    assert commands[1]["arguments_source"] == "deterministic_sensor_log_replay"
    bad = outcome()["observations"]
    bad[2]["event"] = "close_gripper"
    with pytest.raises(ValueError, match="diverged"):
        replay_commands(Policy(SOURCE), bad)


def test_timeline_distinguishes_terminal_failure_from_earlier_motion():
    result = episode_evidence("episode_001", Policy(SOURCE), outcome())
    summary = result["summary"]
    assert summary["diagnostic_task_grade"]["terminal_reason"] == "not_lifted"
    assert summary["max_observed_rise_m"] == pytest.approx(.02)
    assert summary["first_excessive_tilt"] == "episode_001:sample_002"
    assert summary["zero_time_observe_calls"] == 2
    assert all("secret_friction" not in s["observation"] for s in result["trajectory"])


def test_public_evidence_removes_case_names_truth_and_contact_oracles(tmp_path):
    root = fixture_experiment(tmp_path)
    public = prepare_initial_evidence(root)
    text = json.dumps(public)
    for private in ("com_plus8cm", "compiled_inertial_check", "com_offset", "grasp_established", "secret_friction", "999.0", "888.0"):
        assert private not in text
    assert public["evidence_contract"]["episodes_are_distinct_physical_instances"] is True
    assert public["initial_policy"]["sha256"] == Policy(SOURCE).sha256
    assert (root / "initial_cap/organized_trajectories/episode_001.json").exists()
    assert json.loads((root / "evaluator_only_episode_index.json").read_text())["episode_001"]["source_directory"] == "com_plus8cm"


def test_author_probe_records_analysis_and_code_together_without_execution(tmp_path):
    root = fixture_experiment(tmp_path)
    class Agent:
        def call(self, method, params):
            assert method == "physic.propose_probe"
            return response(params)
    manifest = author_active_probe(root, agent=Agent())
    assert manifest["probe_executed"] is False
    assert manifest["dynamics_verified"] is False
    assert manifest["replay_added_physical_interactions"] == 0
    stage = root / "active_probe_1"
    for name in ("probe.py", "prompt.txt", "prompt_template.txt", "context.json", "analysis.md", "analysis.json", "manifest.json"):
        assert (stage / name).is_file()
    with pytest.raises(FileExistsError):
        author_active_probe(root, agent=Agent())


@pytest.mark.parametrize("problem", ["fake_evidence", "cost", "rpc", "execution_claim", "unsupported_api"])
def test_bad_probe_proposal_is_rejected(tmp_path, problem):
    context = prepare_initial_evidence(fixture_experiment(tmp_path))
    context["probe_budget"] = DEFAULT_PROBE_BUDGET
    payload = response(context)
    if problem == "fake_evidence":
        payload["observed_facts"][0]["evidence_ids"] = ["unavailable"]
    elif problem == "cost":
        payload["candidate_probes"][0]["expected_cost_seconds"] = 999
    elif problem == "rpc":
        payload["candidate_probes"][0]["estimated_rpc_calls"] = 95
    elif problem == "execution_claim":
        payload["status"] = "verified"
    else:
        payload["code"] = "def run(robot):\n    robot.apply_force(10)\n"
    with pytest.raises(ValueError):
        validate_proposal(payload, context)


def test_physic_service_repairs_schema_or_source_without_tools(tmp_path):
    context = prepare_initial_evidence(fixture_experiment(tmp_path))
    context["probe_budget"] = DEFAULT_PROBE_BUDGET
    class Agent:
        count = 0
        def generate(self, messages, destination, *, schema):
            self.count += 1
            assert schema == PHYSIC_SCHEMA
            payload = response(context)
            if self.count == 1:
                payload["code"] = "def run(:"
            else:
                assert "No probe was executed" in messages[-1]["content"]
            return payload
    agent = Agent()
    result = PhysicService(tmp_path / "inference", agent=agent).call("physic.propose_probe", context)
    assert result["status"] == "authored_not_executed"
    assert agent.count == 2
