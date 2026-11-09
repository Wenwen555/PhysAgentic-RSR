
"""M1: 主动闩锁辨识 -> Simulator Repair -> Held-out 验证。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from active_sysid.backends import create_backend
from active_sysid.identification.mechanism import MechanismIdentifier
from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.probes.executor import ProbeExecutor
from active_sysid.probes.library import ProbeLibrary
from active_sysid.simulator.model_editor import ModelEditor
from active_sysid.simulator.validator import Validator


ROOT = Path(__file__).resolve().parents[1]


def load_world_spec(name: str) -> dict:
    path = (
        ROOT / "configs/worlds" / f"{name}.yaml"
    )

    spec = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    spec["mjcf"] = str(
        ROOT / spec["mjcf"]
    )
    return spec


def run(output_dir: Path) -> dict:
    cfg = yaml.safe_load(
        (ROOT / "configs/pipeline.yaml").read_text(
            encoding="utf-8"
        )
    )

    library = ProbeLibrary.load(
        ROOT / "configs/probes.yaml"
    )

    executor = ProbeExecutor(
        feature_eps=cfg["features"]
    )

    oracle_spec = load_world_spec(
        "oracle_door"
    )
    reconstructed_spec = load_world_spec(
        "reconstructed_door"
    )

    world = create_backend(oracle_spec)

    hypotheses = (
        HypothesisSet.initial_door_hypotheses()
    )

    training = []

    def probe(name: str, overrides: dict):
        spec = library.get(name).instantiate(
            "door_0",
            overrides=overrides,
        )

        evidence = executor.execute(
            spec,
            world,
        )

        hypotheses.update(evidence)
        training.append(evidence)

        handle = (
            evidence.trajectory
            .channels["handle_angle"]
            .max()
        )

        print(
            f"[train {len(training):02d}] "
            f"{name} {overrides} "
            f"handle_max={handle:.4f} "
            f"door_opens="
            f"{evidence.summary['door_opens']}"
        )

        return evidence

    # Step 1: 不操作把手，直接拉门。
    direct = probe(
        "gentle_pull",
        {
            "max_force": 10.0,
            "duration": 1.5,
        },
    )

    # Step 2: 两次条件拉门，建立初始阈值区间。
    low = probe(
        "hold_and_pull",
        {
            "handle_angle": 0.4,
            "max_force": 12.0,
            "duration": 1.5,
        },
    )

    high = probe(
        "hold_and_pull",
        {
            "handle_angle": 0.7,
            "max_force": 12.0,
            "duration": 1.5,
        },
    )

    assert direct.summary["door_immobile"]
    assert not low.summary["door_opens"]
    assert high.summary["door_opens"]

    # Step 3: Active Probe 二分搜索。
    # 此处在 M0 已发现的 0.4-0.7 rad 区间内搜索。
    lo, hi = 0.4, 0.7

    for _ in range(4):
        mid = (lo + hi) / 2.0

        ev = probe(
            "hold_and_pull",
            {
                "handle_angle": mid,
                "max_force": 12.0,
                "duration": 1.5,
            },
        )

        if ev.summary["door_opens"]:
            hi = mid
        else:
            lo = mid

    # Step 4: 从实际轨迹识别机构。
    model = MechanismIdentifier().update(
        hypotheses,
        training,
    )

    inferred_lo, inferred_hi = (
        model.param_bounds["theta_unlock"]
    )

    # Step 5: 修复初始重建 Simulator。
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repaired_path = ModelEditor().apply(
        reconstructed_spec["mjcf"],
        model,
        params={
            "theta_unlock":
                model.params["theta_unlock"]
        },
        out_path=(
            output_dir / "repaired_door.mjcf"
        ),
    )

    # Step 6: 定义新的 held-out Probe。
    # 角度、拉力与时长均不同于训练组合。
    heldout_specs = [
        library.get("gentle_pull").instantiate(
            "door_0",
            {
                "max_force": 15.0,
                "duration": 1.8,
            },
        ),

        library.get("hold_and_pull").instantiate(
            "door_0",
            {
                "handle_angle": 0.45,
                "max_force": 10.0,
                "duration": 1.8,
            },
        ),

        library.get("hold_and_pull").instantiate(
            "door_0",
            {
                "handle_angle": 0.85,
                "max_force": 16.0,
                "duration": 1.8,
            },
        ),
    ]

    train_keys = {
        e.probe.key()
        for e in training
    }

    assert not train_keys.intersection(
        spec.key()
        for spec in heldout_specs
    )

    # 修复模型已经生成，之后才收集 held-out 真值。
    heldout = [
        executor.execute(spec, world)
        for spec in heldout_specs
    ]

    validator = Validator(
        thresholds=cfg["validator"]["thresholds"]
    )

    kwargs = {
        "model": model,
        "heldout": heldout,
        "backend_spec": reconstructed_spec,
        "door_opens_eps": (
            cfg["features"]["door_opens_eps"]
        ),
    }

    # Step 7: 修复前后使用完全相同的 held-out。
    before = validator.evaluate(
        reconstructed_spec["mjcf"],
        **kwargs,
    )

    after = validator.evaluate(
        repaired_path,
        **kwargs,
    )

    acceptance = {
        "identified_latch": (
            model.hid == "H3_hidden_latch"
        ),
        "threshold_interval_tight": (
            inferred_hi - inferred_lo < 0.05
        ),
        "original_sim_wrong": (
            before.event_err > 0.0
        ),
        "repaired_state_correct": (
            after.structure_predictive
        ),
        "trajectory_improved": (
            after.traj_err < before.traj_err
        ),
    }

    result = {
        "training_probes": [
            e.to_dict()
            for e in training
        ],
        "mechanism": model.hid,
        "theta_unlock_estimate": (
            model.params["theta_unlock"]
        ),
        "theta_unlock_observation_interval": [
            inferred_lo,
            inferred_hi,
        ],
        "repaired_mjcf": str(repaired_path),
        "heldout_probes": [
            e.to_dict()
            for e in heldout
        ],
        "before": before.to_dict(),
        "after": after.to_dict(),
        "acceptance": acceptance,
        "pass": all(acceptance.values()),
    }

    report_path = (
        output_dir / "m1_report.json"
    )

    report_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nidentified={model.hid} "
        f"theta={model.params['theta_unlock']:.4f} "
        f"bracket=[{inferred_lo:.4f}, "
        f"{inferred_hi:.4f}]"
    )

    print(
        f"before: traj_err={before.traj_err:.4f} "
        f"event_err={before.event_err:.3f}"
    )

    print(
        f" after: traj_err={after.traj_err:.4f} "
        f"event_err={after.event_err:.3f}"
    )

    print(f"acceptance: {acceptance}")
    print(
        f"M1 STRUCTURE: "
        f"{'PASS' if result['pass'] else 'FAIL'}"
    )
    print(f"artifacts: {output_dir}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir
        or ROOT / "runs"
        / f"m1_structure_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    result = run(output_dir)
    raise SystemExit(
        0 if result["pass"] else 1
    )


if __name__ == "__main__":
    main()

