"""Probe-to-CaP revision contracts, using API fixtures without physical actions."""
import json

from physrsr.active.agents.coding import CodingService, REVISION_SCHEMA
from physrsr.active.core.artifacts import save_json
from physrsr.active.core.contracts import Task
from physrsr.active.core.policy import Policy
from physrsr.active.workflows import probe_revision
from physrsr.active.runtime.world import WorldService


def observation():
    return dict(rod_position=[.2, 0, .1], rod_axis=[1, 0, 0], grasp_quat=[0, 0, 1, 0],
        tcp_position=[.2, 0, .1], coordinate_frame="robot_base", time_s=1.)


def test_probe_evidence_projects_public_fields_and_checks_returned_sensors(tmp_path, monkeypatch):
    source = Policy("def run(robot):\n    robot.observe()\n")
    task = Task("rod_lift", "lift", {}, {}).public()
    public = dict(task=task, initial_policy=dict(evidence_id="initial_policy", **source.wire()), episodes=[])
    monkeypatch.setattr(probe_revision, "prepare_initial_evidence", lambda path: public)
    stage = tmp_path / "active_probe_1"
    stage.mkdir()
    (stage / "probe.py").write_text(source.source)
    save_json(stage / "manifest.json", dict(probe_executed=True, probe_sha256=source.sha256))
    for eid, _, _ in probe_revision.FAILED_CASES:
        save_json(stage / "world_execution" / eid / "result.json", dict(
            status="completed", error=None, policy_sha256=source.sha256, rpc_calls=1, cost=1., wall_s=.1,
            secret_true_com_offset=.08,
            observations=[dict(event="initial", observation=observation()), dict(event="observe", observation=observation()),
                dict(event="terminal", observation=observation())],
            program_return=dict(reason="inconclusive", records=[dict(label="baseline", observation=observation())])))
    context = probe_revision.revision_context(tmp_path)
    assert "secret_true_com_offset" not in json.dumps(context)
    assert len(context["world_probe_evidence"]) == 2
    assert context["world_probe_evidence"][0]["summary"]["program_reason"] == "inconclusive"
    assert context["comparison_protocol"]["one_shared_revised_policy"] is True
    assert context["world_probe_evidence"][0]["labeled_observations"][0]["observation"] == observation()


def test_one_coding_call_freezes_one_shared_revision_before_evaluation(tmp_path, monkeypatch):
    source = Policy("def run(robot):\n    robot.observe()\n")
    context = dict(task=Task("rod_lift", "lift", {}, {}).public(),
        initial_policy=dict(evidence_id="initial_policy", **source.wire()), initial_evidence=[],
        probe=dict(evidence_id="probe", **source.wire()), world_probe_evidence=[], comparison_protocol={})
    monkeypatch.setattr(probe_revision, "revision_context", lambda *args: context)
    class Agent:
        calls = 0
        def generate(self, messages, destination, *, schema):
            self.calls += 1
            assert schema == REVISION_SCHEMA
            assert "future" not in context["comparison_protocol"]
            return dict(code=source.source+"# one revision\n", rationale="Use observed response", changes=[
                dict(change="feedback handling", evidence_ids=["probe"], expected_effect="bounded recovery")],
                remaining_uncertainties=["contact is not uniquely identified"])
    inference = Agent()
    api = CodingService(tmp_path / "coding", agent=inference)
    manifest = probe_revision.revise_cap_once(tmp_path, agent=api)
    assert inference.calls == 1
    assert manifest["one_shared_policy"] is True
    assert manifest["evaluated"] is False
    assert manifest["parent_sha256"] == source.sha256
    assert (tmp_path / "revised_cap_1/bundle/policy.py").is_file()
    assert (tmp_path / "revised_cap_1/revision.diff").is_file()


def test_world_description_does_not_expose_private_parameters_or_load_env():
    world = WorldService(dict(physics=dict(com_offset=.08, friction=1.5, mass_kg=.27,
        inertial_mode="fixed_mass_com", grip_force=2.), seeds=[19001]))
    description = world.call("target.describe", {})
    assert "com_offset" not in json.dumps(description)
    assert world.robot.worker is None
