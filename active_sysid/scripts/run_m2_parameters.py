
"""M2: 固定 M1 闩锁结构，辨识连续参数并验证独立 Probe。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import yaml

from active_sysid.backends import create_backend
from active_sysid.identification.cma_es import CMASysID
from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.probes.executor import ProbeExecutor
from active_sysid.probes.library import ProbeLibrary
from active_sysid.simulator.model_editor import ModelEditor
from active_sysid.simulator.validator import Validator


ROOT = Path(__file__).resolve().parents[1]


def _load_spec(name: str) -> dict:
    path = ROOT / "configs/worlds" / f"{name}.yaml"
    spec = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    spec["mjcf"] = str(ROOT / spec["mjcf"])
    return spec


def _mean_channel(report, key: str) -> float:
    return float(np.mean([
        case[key] for case in report.cases
    ]))


def _truth_for_report(
    mjcf_path: Path,
) -> dict[str, float]:
    """仅用于实验结束后的真值误差统计。"""
    root = ET.parse(mjcf_path).getroot()

    leaf = root.find(
        ".//geom[@name='door_leaf']"
    )
    door = root.find(
        ".//joint[@name='door_hinge']"
    )
    handle = root.find(
        ".//joint[@name='handle_hinge']"
    )

    sx, sy, sz = map(
        float, leaf.get("size").split()
    )
    px, py, _ = map(
        float, leaf.get("pos").split()
    )

    mass = (
        8 * sx * sy * sz
        * float(leaf.get("density"))
    )

    inertia = mass * (
        (sx * sx + sy * sy) / 3
        + px * px + py * py
    )

    return {
        "I_d": inertia,
        "b_d": float(door.get("damping")),
        "mu_d": float(door.get("frictionloss")),
        "k_h": float(handle.get("stiffness")),
        "b_h": float(handle.get("damping")),
    }


def run(
    m1_dir: Path,
    output_dir: Path,
) -> dict:
    cfg = yaml.safe_load(
        (ROOT / "configs/pipeline.yaml").read_text(
            encoding="utf-8"
        )
    )

    m1_report = json.loads(
        (m1_dir / "m1_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert (
        m1_report["pass"]
        and m1_report["mechanism"]
        == "H3_hidden_latch"
    )

    m1_mjcf = m1_dir / "repaired_door.mjcf"

    # M1 已识别的机构与阈值。
    model = (
        HypothesisSet.initial_door_hypotheses()
        .get("H3_hidden_latch")
    )

    model.params["theta_unlock"] = float(
        m1_report["theta_unlock_estimate"]
    )

    model.param_bounds["theta_unlock"] = tuple(
        m1_report[
            "theta_unlock_observation_interval"
        ]
    )

    library = ProbeLibrary.load(
        ROOT / "configs/probes.yaml"
    )

    oracle_spec = _load_spec("oracle_door")
    sim_spec = _load_spec("reconstructed_door")

    world = create_backend(oracle_spec)

    executor = ProbeExecutor(
        feature_eps=cfg["features"]
    )

    # 3 次把手回弹 + 4 次解锁后拉门。
    train_cases = [
        (
            "release_and_observe",
            {"hold_angle": 0.32, "duration": 1.8},
        ),
        (
            "release_and_observe",
            {"hold_angle": 0.48, "duration": 2.4},
        ),
        (
            "release_and_observe",
            {"hold_angle": 0.54, "duration": 2.1},
        ),
        (
            "hold_and_pull",
            {
                "handle_angle": 0.85,
                "max_force": 4.0,
                "duration": 2.4,
            },
        ),
        (
            "hold_and_pull",
            {
                "handle_angle": 0.90,
                "max_force": 8.0,
                "duration": 2.1,
            },
        ),
        (
            "hold_and_pull",
            {
                "handle_angle": 0.80,
                "max_force": 12.0,
                "duration": 1.8,
            },
        ),
        (
            "hold_and_pull",
            {
                "handle_angle": 0.90,
                "max_force": 6.0,
                "duration": 2.7,
            },
        ),
    ]

    # 独立于训练集的角度、力和时间组合。
    heldout_cases = [
        (
            "release_and_observe",
            {"hold_angle": 0.42, "duration": 2.2},
        ),
        (
            "release_and_observe",
            {"hold_angle": 0.52, "duration": 1.9},
        ),
        (
            "hold_and_pull",
            {
                "handle_angle": 0.88,
                "max_force": 5.5,
                "duration": 2.5,
            },
        ),
        (
            "hold_and_pull",
            {
                "handle_angle": 0.95,
                "max_force": 10.0,
                "duration": 2.2,
            },
        ),
        (
            "gentle_pull",
            {
                "max_force": 16.0,
                "duration": 1.7,
            },
        ),
    ]

    def to_spec(case):
        name, overrides = case
        return library.get(name).instantiate(
            "door_0",
            overrides=overrides,
        )

    train_specs = [
        to_spec(c) for c in train_cases
    ]
    heldout_specs = [
        to_spec(c) for c in heldout_cases
    ]

    assert not (
        {p.key() for p in train_specs}
        & {p.key() for p in heldout_specs}
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_dir / "observations").mkdir(
        exist_ok=True
    )

    def collect(specs, split):
        evidences = []

        for i, spec in enumerate(specs):
            ev = executor.execute(spec, world)
            evidences.append(ev)

            base = (
                output_dir / "observations"
                / f"{split}_{i:02d}_{spec.name}"
            )

            # 保存完整的 t、角度、角速度和力矩。
            np.savez_compressed(
                base.with_suffix(".npz"),
                t=ev.trajectory.t,
                **ev.trajectory.channels,
            )

            # 保存 Probe 参数、摘要和事件。
            base.with_suffix(".json").write_text(
                json.dumps(
                    ev.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return evidences

    # Step 1: World 训练数据
    training = collect(
        train_specs, "train"
    )

    print(
        f"Collected {len(training)} training Probes"
    )

    # Step 2: 连续参数辨识
    fit = CMASysID(
        cfg["cma_es"]
    ).fit(
        model,
        training,
        source_mjcf=m1_mjcf,
        backend_spec=sim_spec,
    )

    # Step 3: 将拟合参数写回 Simulator
    m2_mjcf = ModelEditor().apply(
        m1_mjcf,
        model,
        params=fit.params,
        out_path=(
            output_dir / "repaired_m2.mjcf"
        ),
    )

    print(
        f"Fit complete: {fit.n_evals} "
        f"candidate evaluations"
    )
    print(f"Fitted params: {fit.params}")

    # Step 4: 固定修复结果后，收集 held-out
    heldout = collect(
        heldout_specs, "heldout"
    )

    validator = Validator(
        cfg["validator"]["thresholds"]
    )

    kwargs = {
        "model": model,
        "heldout": heldout,
        "backend_spec": sim_spec,
        "door_opens_eps": (
            cfg["features"]["door_opens_eps"]
        ),
    }

    # Step 5: 三种模型统一验证
    initial = validator.evaluate(
        sim_spec["mjcf"],
        **kwargs,
    )

    structure = validator.evaluate(
        m1_mjcf,
        **kwargs,
    )

    calibrated = validator.evaluate(
        m2_mjcf,
        **kwargs,
    )

    # Step 6: 验收条件
    acceptance = {
        "handle_training_improved": (
            fit.stages["handle"]["after"]
            < fit.stages["handle"]["before"]
        ),

        "door_training_improved": (
            fit.stages["door"]["after"]
            < fit.stages["door"]["before"]
        ),

        "heldout_traj_improved": (
            calibrated.traj_err
            < structure.traj_err
        ),

        "heldout_door_improved": (
            _mean_channel(
                calibrated, "door_mae"
            )
            < _mean_channel(
                structure, "door_mae"
            )
        ),

        "heldout_handle_improved": (
            _mean_channel(
                calibrated, "handle_mae"
            )
            < _mean_channel(
                structure, "handle_mae"
            )
        ),

        "structure_preserved": (
            calibrated.event_err
            <= structure.event_err
        ),
    }

    # 只有在优化与验证结束后才读取真值。
    truth = _truth_for_report(
        Path(oracle_spec["mjcf"])
    )

    relative_error = {
        key: abs(fit.params[key] - true) / abs(true)
        for key, true in truth.items()
    }

    result = {
        "m1_source": str(m1_dir),
        "m1_unlock_threshold": (
            model.params["theta_unlock"]
        ),
        "training_count": len(training),
        "heldout_count": len(heldout),

        "fit": {
            "backend": fit.backend,
            "params": fit.params,
            "loss": fit.loss,
            "n_evals": fit.n_evals,
            "stages": fit.stages,
        },

        "initial": initial.to_dict(),
        "structure_only": structure.to_dict(),
        "structure_plus_parameters": (
            calibrated.to_dict()
        ),

        "evaluation_only": {
            "oracle_params": truth,
            "relative_errors": relative_error,
        },

        "acceptance": acceptance,
        "pass": all(acceptance.values()),
    }

    (output_dir / "m2_report.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"held-out traj MAE: "
        f"initial={initial.traj_err:.5f}, "
        f"M1={structure.traj_err:.5f}, "
        f"M2={calibrated.traj_err:.5f} rad"
    )

    print(f"acceptance={acceptance}")
    print(
        "M2 PARAMETERS: "
        f"{'PASS' if result['pass'] else 'FAIL'}"
    )
    print(f"artifacts: {output_dir}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--m1-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    args = parser.parse_args()

    out = (
        args.output_dir
        or ROOT / "runs"
        / f"m2_parameters_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    result = run(args.m1_dir, out)

    raise SystemExit(
        0 if result["pass"] else 1
    )


if __name__ == "__main__":
    main()

