"""Contract/state-machine tests only: no inference, MuJoCo or hardware."""
from dataclasses import replace
import json

import pytest

from physrsr.active.core.contracts import API_VERSION, Hypothesis, LoopConfig, Task
from physrsr.active.workflows.legacy_loop import LegacyActiveLoop as ActiveLoop
from physrsr.active.core.policy import Policy
from physrsr.active.core.reasoning import assess, posterior, probe_value, Probe

LEFT = "def run(robot):\n    return 'left'\n"
RIGHT = "def run(robot):\n    return 'right'\n"
FIT = "def run(robot):\n    return 'fit'\n"
CHECK = "def run(robot):\n    return 'check'\n"
NUISANCE = "def run(robot):\n    return 'nuisance'\n"
TASK = Task("rod_lift", "lift the rod", {}, {})
MODELS = (Hypothesis("left", {"com_offset": -.04}, .5), Hypothesis("right", {"com_offset": .04}, .5))


def observed():
    return dict(rod_position=[0, 0, .1], rod_axis=[1, 0, 0], grasp_quat=[0, 0, 1, 0],
                tcp_position=[0, 0, .1], coordinate_frame="robot_base", time_s=0.)


class Agent:
    def __init__(self, validation=True):
        self.requests, self.validation = [], validation

    def call(self, method, params):
        self.requests.append((method, params))
        if method == "agent.generate_initial_cap":
            assert set(params) == {"task", "observation"}
            return dict(mode="policy", command="", files=[dict(path="policy.py", source=LEFT)])
        assert method == "agent.propose"
        policies = [dict(code=LEFT + ("# revised\n" if params["round"] else ""), rationale="track CoM"),
                    dict(code=RIGHT, rationale="alternative")]
        probes = [dict(id="identify", code=FIT, rationale="distinguish grasp side", cost=1.,
                       factors=["com_offset"], family="low_lift", purpose="fit"),
                  dict(id="irrelevant", code=NUISANCE, rationale="same decision", cost=1.,
                       factors=["friction"], family="nuisance", purpose="fit")]
        if self.validation:
            probes.append(dict(id="heldout", code=CHECK, rationale="new trajectory", cost=1.,
                factors=["contact_model"], family="heldout_trajectory", purpose="validation"))
        return dict(policies=policies, probes=probes,
            uncertainties=[dict(factor="CoM", task_effect="changes grasp side", evidence="predicted failure")])


class Simulator:
    def __init__(self, all_robust=False):
        self.requests, self.all_robust = [], all_robust

    def call(self, method, params):
        self.requests.append((method, params))
        if method == "sim.describe":
            return dict(api_version=API_VERSION, mechanisms=[MODELS[0].mechanism], tasks=[TASK.id], feature_schema="rod_response_v1")
        if method == "sim.evaluate":
            rows = []
            for m in params["models"]:
                for p in params["policies"]:
                    ok = self.all_robust or (("'left'" in p["source"]) == (m["parameters"]["com_offset"] < 0))
                    rows.append(dict(model_id=m["id"], policy_sha256=p["sha256"], success_rate=float(ok),
                                     risk=0., duration_s=10., margin=.1 if ok else -.1))
            return dict(rows=rows)
        if method == "sim.predict_probe":
            p = params["probe"]["policy"]
            features = lambda m: [0., 0., m["parameters"]["com_offset"] if p["source"] == FIT else 0., 0., 0.]
            return dict(policy_sha256=p["sha256"], feature_schema="rod_response_v1",
                        predictions={m["id"]: dict(features=features(m), risk=0.) for m in params["models"]})
        raise AssertionError(method)


class Target:
    def __init__(self, validation_error=False, actual_cost=1.):
        self.requests, self.validation_error, self.actual_cost = [], validation_error, actual_cost

    def call(self, method, params):
        self.requests.append((method, params))
        if method == "target.describe":
            return dict(api_version=API_VERSION, feature_schema="rod_response_v1")
        if method == "target.observe":
            return observed()
        assert method == "target.execute_probe"  # Final task must never run.
        features = [0., 0., -.04 if params["purpose"] == "fit" else 0., 0., 0.]
        if params["purpose"] == "validation" and self.validation_error:
            features[0] = 10.
        return dict(status="completed", error=None, policy_sha256=params["policy"]["sha256"],
                    evidence_id=str(len(self.requests)), features=features, cost=self.actual_cost,
                    feature_schema="rod_response_v1", observations=[])


def make_loop(tmp_path, cfg=None, agent=None, simulator=None, target=None):
    return ActiveLoop(TASK, cfg or LoopConfig(), MODELS, agent=agent or Agent(),
        simulator=simulator or Simulator(), target=target or Target(), output=tmp_path / "run")


def test_active_probe_updates_model_revises_policy_and_freezes_identical_source(tmp_path):
    loop = make_loop(tmp_path)
    result = loop.run()
    assert result["status"] == "task_sufficient"
    assert result["fit_probe_count"] == result["validation_probe_count"] == 1
    assert result["probe_count"] == result["probe_cost"] == 2
    assert loop.models[0].weight > .999
    assert len(loop.store.policies) == 3
    assert any(p.parent for p in loop.store.policies.values())
    frozen = Policy((loop.output / "final/policy.py").read_text())
    assert frozen.sha256 == result["assessment"]["selected"]
    assert frozen.source.encode() == loop.store.policies[frozen.sha256].source.encode()
    assert result["target_task_executed"] is False
    assert not any(method == "target.execute_policy" for method, _ in loop.target.requests)
    assert "ground_truth" not in json.dumps(loop.agent.requests)
    chosen = [p["probe_id"] for m, p in loop.target.requests if m == "target.execute_probe"]
    assert chosen == ["identify", "heldout"]
    # Validation does not update the fit posterior or its version.
    assert result["model_version"] == 1
    assert result["models"][0]["weight"] == loop.models[0].weight
    events = [json.loads(line) for line in (loop.output / "events.jsonl").read_text().splitlines()]
    calls = [event["method"] for event in events if event["event"] == "api_request"]
    assert calls.index("agent.generate_initial_cap") < calls.index("sim.evaluate") < calls.index("agent.propose")
    initial_request = next(event["params"] for event in events if event["event"] == "api_request" and event["method"] == "agent.generate_initial_cap")
    assert set(initial_request) == {"task", "observation"}


def test_task_irrelevant_physical_uncertainty_stops_without_fit(tmp_path):
    loop = make_loop(tmp_path, simulator=Simulator(all_robust=True))
    result = loop.run()
    assert result["status"] == "task_sufficient"
    assert result["fit_probe_count"] == 0
    assert result["validation_probe_count"] == 1
    assert [m.weight for m in loop.models] == [.5, .5]


@pytest.mark.parametrize("cfg,status", [
    (LoopConfig(max_probes=0), "probe_budget_exhausted"),
    (LoopConfig(max_probe_cost=0), "cost_budget_exhausted"),
    (LoopConfig(max_rounds=1), "round_budget_exhausted")])
def test_budget_is_never_sufficiency(tmp_path, cfg, status):
    result = make_loop(tmp_path, cfg=cfg).run()
    assert result["status"] == status
    assert result["final_policy"] is None
    assert not (tmp_path / "run/final").exists()


def test_no_validation_evidence_means_no_final_policy(tmp_path):
    result = make_loop(tmp_path, cfg=LoopConfig(max_rounds=2), agent=Agent(validation=False),
                       simulator=Simulator(all_robust=True)).run()
    assert result["status"] == "round_budget_exhausted"
    assert result["final_policy"] is None


def test_failed_independent_prediction_does_not_fit_validation(tmp_path):
    loop = make_loop(tmp_path, cfg=LoopConfig(max_model_revisions=0),
                     target=Target(validation_error=True), simulator=Simulator(all_robust=True))
    result = loop.run()
    assert result["status"] == "model_inadequate"
    assert loop.fit_data == []
    assert result["model_version"] == 0
    assert [m.weight for m in loop.models] == [.5, .5]


def test_measured_cost_overrun_prevents_delivery(tmp_path):
    result = make_loop(tmp_path, cfg=LoopConfig(max_probe_cost=1), target=Target(actual_cost=3)).run()
    assert result["status"] == "cost_budget_exhausted"
    assert result["probe_cost"] == 3
    assert result["final_policy"] is None


def test_bad_simulator_matrix_aborts_as_infrastructure_error(tmp_path):
    class Broken(Simulator):
        def call(self, method, params):
            if method == "sim.evaluate":
                return dict(rows=[])
            return super().call(method, params)
    loop = make_loop(tmp_path, simulator=Broken())
    with pytest.raises(ValueError, match="matrix"):
        loop.run()
    assert json.loads((loop.output / "result.json").read_text())["status"] == "infrastructure_error"


def test_independent_probe_cannot_reuse_fitted_action(tmp_path):
    class Reuse(Agent):
        def call(self, method, params):
            value = super().call(method, params)
            if method == "agent.propose":
                value["probes"][-1]["code"] = FIT
            return value
    result = make_loop(tmp_path, cfg=LoopConfig(max_rounds=3), agent=Reuse()).run()
    assert result["status"] == "round_budget_exhausted"
    assert result["validation_probe_count"] == 0


def test_parameter_information_without_decision_value_is_zero(tmp_path):
    sim = Simulator(all_robust=True)
    policies = [Policy(LEFT), Policy(RIGHT)]
    rows = sim.call("sim.evaluate", dict(models=[m.__dict__ for m in MODELS], policies=[p.wire() for p in policies]))["rows"]
    probe = Probe("com", Policy(FIT), 1., ("com",), "f")
    predictions = {m.id: dict(features=[0, 0, m.parameters["com_offset"], 0, 0], risk=0.) for m in MODELS}
    value = probe_value(probe, predictions, MODELS, policies, rows, LoopConfig())
    assert value["expected_decision_value"] == pytest.approx(0.)
    assert value["net_value"] < 0
    fitted = posterior(MODELS, [0, 0, -.04, 0, 0], predictions, LoopConfig().feature_scales)
    assert fitted[0].weight > .99


def test_speed_can_make_probe_useful_despite_robust_slow_policy():
    slow, fast = Policy(LEFT), Policy(RIGHT)
    rows = [dict(model_id=m.id, policy_sha256=p.sha256,
        success_rate=1. if p == slow or m.id == "left" else 0., risk=0., margin=.1,
        duration_s=100. if p == slow else 1.) for m in MODELS for p in (slow, fast)]
    cfg = LoopConfig(time_weight=.01)
    assert assess(MODELS, [slow, fast], rows, cfg)["selected"] == slow.sha256
    predictions = {m.id: dict(features=[0, 0, m.parameters["com_offset"], 0, 0], risk=0.) for m in MODELS}
    result = probe_value(Probe("p", Policy(FIT), 1., ("friction",), "f"), predictions,
                         MODELS, [slow, fast], rows, cfg)
    assert result["net_value"] > .4


def test_model_revision_replays_fit_history_and_requires_fresh_validation(tmp_path):
    class RevisingAgent(Agent):
        def call(self, method, params):
            if method == "agent.revise_model":
                self.requests.append((method, params))
                return dict(rationale="Expand prior using measured response", models=[
                    dict(id="new_left", parameters={"com_offset": -.04}, weight=1., mechanism=MODELS[0].mechanism)])
            return super().call(method, params)
    loop = ActiveLoop(TASK, LoopConfig(), [Hypothesis("wrong", {"com_offset": .04}, 1.)],
        agent=RevisingAgent(), simulator=Simulator(), target=Target(), output=tmp_path / "run")
    # Directly exercise an out-of-support fit, then the normal loop. No target
    # inference/physics is involved in these supplied numerical API fixtures.
    loop.simulator_info = loop.simulator.call("sim.describe", {})
    probe = Probe("fit", Policy(FIT), 1., ("com",), "fit")
    loop.fit_data.append(dict(probe=probe, observation=dict(features=[0, 0, -.04, 0, 0])))
    loop.validation_data.append(dict(model_version=0, passed=True, policy_sha256="old", family="old"))
    assert loop.revise_model(dict(reason="outside support"))
    assert loop.model_version == 1
    assert loop.models[0].parameters["com_offset"] == -.04
    result = loop.run()
    assert result["status"] == "task_sufficient"
    assert result["probe_count"] == 1  # fresh validation, never the old record
    assert loop.validation_data[-1]["model_version"] == 1


def test_out_of_support_fit_triggers_model_revision_in_the_full_loop(tmp_path):
    class Expanding(Agent):
        def call(self, method, params):
            if method == "agent.revise_model":
                self.requests.append((method, params))
                return dict(rationale="Response lies beyond prior support", models=[dict(
                    id="expanded", parameters={"com_offset": -.08}, weight=1., mechanism=MODELS[0].mechanism)])
            return super().call(method, params)
    class Outside(Target):
        def call(self, method, params):
            result = super().call(method, params)
            if method == "target.execute_probe" and params["purpose"] == "fit":
                result["features"][2] = -.08
            return result
    loop = make_loop(tmp_path, agent=Expanding(), target=Outside())
    result = loop.run()
    assert result["status"] == "task_sufficient"
    assert loop.model_revisions == 1
    assert result["fit_probe_count"] == result["validation_probe_count"] == 1
    assert loop.models[0].parameters["com_offset"] == -.08
    assert any(m == "agent.revise_model" for m, _ in loop.agent.requests)
    progress = json.loads((loop.output / "progress.json").read_text())
    assert progress[-1]["probe_cost"] == result["probe_cost"]


def test_invalid_program_returns_to_coding_without_target_interaction(tmp_path):
    class Fixing(Agent):
        def call(self, method, params):
            value = super().call(method, params)
            if method == "agent.propose" and params["round"] == 0:
                value["policies"] = [dict(code="def run(:", rationale="invalid")]
            elif method == "agent.propose" and params["round"] > 0:
                assert params["prediction_feedback"]["program_errors"]
            return value
    result = make_loop(tmp_path, agent=Fixing(), simulator=Simulator(all_robust=True)).run()
    assert result["status"] == "task_sufficient"
    assert result["rounds"] == 1  # the valid initial policy remains available
    assert result["probe_count"] == 1


def test_unsafe_probes_do_not_execute(tmp_path):
    class Unsafe(Simulator):
        def call(self, method, params):
            result = super().call(method, params)
            if method == "sim.predict_probe":
                for prediction in result["predictions"].values():
                    prediction["risk"] = 1.
            return result
    result = make_loop(tmp_path, cfg=LoopConfig(max_rounds=2), simulator=Unsafe(all_robust=True)).run()
    assert result["status"] == "round_budget_exhausted"
    assert result["probe_count"] == 0


def test_cosmetic_source_change_is_not_a_new_validation_action(tmp_path):
    class Cosmetic(Agent):
        def call(self, method, params):
            result = super().call(method, params)
            if method == "agent.propose":
                result["probes"][-1]["code"] = FIT + "# supposedly new action\n"
            return result
    result = make_loop(tmp_path, cfg=LoopConfig(max_rounds=3), agent=Cosmetic()).run()
    assert result["status"] == "round_budget_exhausted"
    assert result["validation_probe_count"] == 0
