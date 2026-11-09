"""M0 基准测试: oracle (隐藏闩锁) vs reconstructed (缺闩锁) 双场景。

验收语义 (见 scripts/run_m0_baseline.py 与 docs/m0_baseline.md):
- 未旋转把手的 gentle_pull: oracle 门不动, reconstructed 开门 (初始预测误差);
- 解锁状态拉门: 两个场景都能开门;
- 同一 ProbeSpec 在两场景产生同构观测通道;
- 两场景执行均可重复 (确定性);
- hypothesis 初始先验有界且不以真值为中心 (真值仅从 eval_truth.yaml 读取比较)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from active_sysid.backends import create_backend
from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.probes.executor import ProbeExecutor
from active_sysid.probes.library import ProbeLibrary

ROOT = Path(__file__).resolve().parents[1]


def _load_backend(name: str):
    spec = yaml.safe_load((ROOT / "configs" / "worlds" / f"{name}.yaml")
                          .read_text(encoding="utf-8"))
    spec["mjcf"] = str(ROOT / spec["mjcf"])
    return create_backend(spec)


def _features() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    return cfg["features"]


@pytest.fixture(scope="module")
def library() -> ProbeLibrary:
    return ProbeLibrary.load(ROOT / "configs" / "probes.yaml")


@pytest.fixture(scope="module")
def backends():
    return _load_backend("oracle_door"), _load_backend("reconstructed_door")


def test_locked_gentle_pull_separates_world_and_sim(library, backends):
    world, sim = backends
    spec = library.get("gentle_pull").instantiate(
        "door_0", overrides={"max_force": 10.0, "duration": 1.5})
    eps = _features()

    world_ev = ProbeExecutor(feature_eps=eps).execute(spec, world)
    sim_ev = ProbeExecutor(feature_eps=eps).execute(spec, sim)

    assert world_ev.summary["door_immobile"] is True   # 闩锁挡住
    assert world_ev.summary["door_opens"] is False
    assert sim_ev.summary["door_immobile"] is False    # 缺闩锁: 直接开门
    assert sim_ev.summary["door_opens"] is True


def test_unlocked_pull_opens_both(library, backends):
    world, sim = backends
    spec = library.get("hold_and_pull").instantiate(
        "door_0", overrides={"handle_angle": 0.7, "max_force": 12.0, "duration": 1.5})
    eps = _features()

    for backend in (world, sim):
        ev = ProbeExecutor(feature_eps=eps).execute(spec, backend)
        assert ev.summary["door_opens"] is True


def test_unlock_threshold_bracketed_in_world(library, backends):
    """把手 0.4 rad (低于阈值) 拉不开, 0.7 rad 拉得开 —— 阈值被夹在中间。"""
    world, _ = backends
    eps = _features()
    for angle, expect_open in ((0.4, False), (0.7, True)):
        spec = library.get("hold_and_pull").instantiate(
            "door_0", overrides={"handle_angle": angle, "max_force": 12.0,
                                 "duration": 1.5})
        ev = ProbeExecutor(feature_eps=eps).execute(spec, world)
        assert ev.summary["door_opens"] is expect_open, f"handle_angle={angle}"


def test_same_probespec_same_observation_semantics(library, backends):
    world, sim = backends
    spec = library.get("rotate_handle").instantiate("door_0")
    a, b = world.run_probe(spec), sim.run_probe(spec)
    assert set(a.channels) == set(b.channels)  # 同构观测通道
    # 把手在两场景都可转动 (oracle 有回位弹簧, PD 需压过弹簧, 幅值略小)
    assert np.ptp(a.channels["handle_angle"]) > 0.3
    assert np.ptp(b.channels["handle_angle"]) > 0.3


def test_backends_are_deterministic(library, backends):
    spec = library.get("hold_and_pull").instantiate(
        "door_0", overrides={"handle_angle": 0.7, "max_force": 12.0, "duration": 1.5})
    for backend in backends:
        a, b = backend.run_probe(spec), backend.run_probe(spec)
        np.testing.assert_allclose(a.channels["door_angle"],
                                   b.channels["door_angle"], atol=1e-9)


def test_hypothesis_priors_are_bounded_and_off_truth():
    truth = yaml.safe_load(
        (ROOT / "configs" / "eval_truth.yaml").read_text(encoding="utf-8"))
    # oracle 几何解锁阈值 (行程/耦合) 与把手/铰链真值
    true_vals = {
        "theta_unlock": truth["latch"]["unlock_threshold_rad"],
        "k_h": truth["handle"]["spring_stiffness"],
        "b_h": truth["handle"]["damping"],
        "b_d": truth["door"]["hinge_damping"],
    }
    hyps = HypothesisSet.initial_door_hypotheses()
    h3 = hyps.get("H3_hidden_latch")

    for key, true_val in true_vals.items():
        lo, hi = h3.param_bounds[key]
        assert lo < true_val < hi, f"{key} 真值 {true_val} 不在先验界内"
        assert abs(h3.params[key] - true_val) > 1e-6, \
            f"{key} 先验 {h3.params[key]} 不得等于真值 {true_val}"

    # 阈值先验 (0.85) 应落在 0.4/0.7 的夹逼区间之外也能被交互证据修正
    assert h3.params["theta_unlock"] == pytest.approx(0.85)

def test_strong_pull_cannot_bypass_latch(library, backends):
    world, _ = backends

    spec = library.get("gentle_pull").instantiate(
        "door_0",
        overrides={
            "max_force": 40.0,
            "duration": 1.5,
        },
    )

    traj = world.run_probe(spec)

    assert (
        np.max(np.abs(traj.channels["door_angle"]))
        < _features()["door_immobile_eps"]
    )

    # Oracle 内部状态仅供测试使用
    assert world._latched is True


def test_hold_then_pull_timing_and_observability(library, backends):
    spec = library.get("hold_and_pull").instantiate(
        "door_0",
        overrides={
            "handle_angle": 0.7,
            "max_force": 12.0,
            "duration": 1.5,
        },
    )

    for backend in backends:
        traj = backend.run_probe(spec)

        before_pull = (
            traj.t < spec.params["duration"] * 0.5
        )

        # 前半段只能转动把手，不施加拉门力
        assert np.allclose(
            traj.channels["door_angle_tau_applied"][
                before_pull
            ],
            0.0,
        )

        # 开始拉门前，把手已进入目标角度区域
        pull_start_idx = np.flatnonzero(
            ~before_pull
        )[0]

        assert (
            traj.channels["handle_angle"][pull_start_idx]
            > 0.6
        )

        # 隐藏的解锁状态不能直接泄漏给 Agent
        assert "unlock" not in traj.events

        # 完成解锁动作后可以开门
        assert (
            traj.channels["door_angle"].max()
            > _features()["door_opens_eps"]
        )


def test_latch_reengages_after_handle_returns(library, backends):
    world, _ = backends

    spec = library.get("rotate_handle").instantiate(
        "door_0"
    )

    traj = world.run_probe(spec)

    assert traj.channels["handle_angle"].max() > 0.6
    assert world._latched is False

    # 释放把手 1 秒，让回位弹簧起作用
    for _ in range(200):
        world._apply({
            "handle_angle": 0.0,
            "door_angle": 0.0,
        })
        world._step()

    assert world._latched is True
