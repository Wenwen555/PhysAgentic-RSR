"""冒烟测试: 配置加载 + 假设更新 + 主链路 mock 闭环。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from active_sysid import TaskSpec
from active_sysid.identification.cma_es import CMASysID
from active_sysid.identification.mechanism import MechanismIdentifier
from active_sysid.pipeline import ProbeLoop
from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.planner.probe_planner import ProbePlanner
from active_sysid.probes.executor import ProbeExecutor
from active_sysid.probes.library import ProbeLibrary
from active_sysid.simulator.model_editor import ModelEditor
from active_sysid.simulator.validator import Validator

ROOT = Path(__file__).resolve().parents[1]
from active_sysid.backends import MockDoorWorld  # noqa: E402


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_probe_library_loads_nine_templates():
    library = ProbeLibrary.load(ROOT / "configs" / "probes.yaml")
    assert len(library.templates) == 9
    assert "hold_and_pull" in library.names()

    spec = library.get("hold_and_pull").instantiate(target="door_0")
    assert spec.params["handle_angle"] == pytest.approx(0.4)

    with pytest.raises(ValueError):
        library.get("hold_and_pull").instantiate(
            target="door_0", overrides={"handle_angle": 5.0}
        )


def test_hypothesis_update_boosts_latch_on_conditioned_evidence():
    hyps = HypothesisSet.initial_door_hypotheses()
    before = hyps.posterior()["H3_hidden_latch"]
    hyps.update(_evidence(summary={"door_immobile": True, "handle_rotates": True,
                                   "unlock_angle": 0.4}))
    after = hyps.posterior()["H3_hidden_latch"]
    assert after > before
    assert hyps.best().hid == "H3_hidden_latch"


def test_probe_loop_identifies_latch_on_mock_world(tmp_path):
    cfg = _load_yaml("pipeline.yaml")
    library = ProbeLibrary.load(ROOT / "configs" / "probes.yaml")
    loop = ProbeLoop(
        planner=ProbePlanner(weights=cfg["weights"]),
        library=library,
        executor=ProbeExecutor(feature_eps=cfg["features"], record_dir=tmp_path / "probes"),
        mechanism_identifier=MechanismIdentifier(),
        parameter_identifier=CMASysID(cfg["cma_es"]),
        model_editor=ModelEditor(),
        validator=Validator(thresholds=cfg["validator"]["thresholds"]),
        config=cfg,
        output_dir=tmp_path / "run",
    )
    task = TaskSpec(task_id="open_left_door",
                    description="打开左侧房门, 使机器人能够进入下一个房间")
    report = loop.run(task, scene={"objects": ["door_0"]}, world=MockDoorWorld())

    # 闩锁结构被辨识, 解锁阈值来自交互证据而非真值文件
    assert report.best_hypothesis["hid"] == "H3_hidden_latch"
    assert not report.predictive
    assert report.stop_reason == "max_probes"
    assert 3 <= report.n_probes <= 12
    assert (tmp_path / "run" / "report.json").exists()
    # 每次执行的 Probe 都有留痕
    probe_records = list((tmp_path / "probes").glob("probe_*.json"))
    assert len(probe_records) == report.n_probes


def test_executor_requires_feature_eps_from_config():
    with pytest.raises(ValueError, match="features"):
        ProbeExecutor()
    with pytest.raises(ValueError, match="door_opens_eps"):
        ProbeExecutor(feature_eps={"door_immobile_eps": 0.02})


def test_components_require_config_sections():
    # 所有研究超参只以 pipeline.yaml 为真值, 组件无默认副本
    with pytest.raises(ValueError, match="weights"):
        ProbePlanner()
    with pytest.raises(ValueError, match="validator.thresholds"):
        Validator()
    with pytest.raises(ValueError, match="cma_es"):
        CMASysID()
    with pytest.raises(ValueError, match="cma_es.loss_weights"):
        CMASysID({"loss_weights": {"w_q": 1.0}, "sigma0": 0.3, "max_iter": 10, "seed": 7,})
    with pytest.raises(ValueError, match="probe_loop"):
        ProbeLoop(planner=None, library=None, executor=None,
                  mechanism_identifier=None, parameter_identifier=None,
                  model_editor=None, validator=None, config=None)


def test_model_editor_writes_repair_sidecar(tmp_path):
    src = (
        ROOT / "worlds/legacy_object_space/reconstructed"
        / "reconstructed_door.mjcf"
    )
    out = ModelEditor().apply(
        src,
        model=HypothesisSet.initial_door_hypotheses().get("H3_hidden_latch"),
        params={"theta_unlock": 0.4, "k_h": 0.5},
        out_path=tmp_path / "repaired" / "scene.mjcf",
    )
    assert out.exists()
    sidecar = out.with_suffix(out.suffix + ".repair.json")
    repair = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    assert repair["repair_mode"] == "functional_latch"
    assert repair["params"]["theta_unlock"] == pytest.approx(0.4)


def _evidence(summary: dict):
    from active_sysid.types import Evidence, ProbeSpec, ProbeTrajectory
    import numpy as np
    return Evidence(
        probe=ProbeSpec(name="hold_and_pull", target="door_0", params={"handle_angle": 0.4}),
        trajectory=ProbeTrajectory(t=np.zeros(2), channels={}),
        summary=summary,
    )
