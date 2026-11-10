
"""M3-0: 使用固定程序验证 Sim / World 的 CaP 执行链路。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from active_sysid.cap.evaluator import (
    CaPEvaluator,
    scene_spec,
)
from active_sysid.types import TaskSpec


ROOT = Path(__file__).resolve().parents[1]


def load_spec(name: str) -> dict:
    path = (
        ROOT / "configs/worlds" / f"{name}.yaml"
    )
    spec = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    spec["mjcf"] = str(ROOT / spec["mjcf"])
    return spec


def direct_pull(api):
    api.step(
        door=("force", 12.0),
        steps=300,
    )


def operate_handle_then_pull(api):
    api.step(
        handle=("position", 0.8),
        steps=100,
    )

    api.step(
        handle=("position", 0.8),
        door=("force", 12.0),
        steps=300,
    )


def run(
    m1_dir: Path,
    m2_dir: Path,
    out: Path,
):
    reconstructed = load_spec(
        "reconstructed_door"
    )
    oracle = load_spec("oracle_door")

    scenes = {
        "initial_sim": reconstructed,
        "m1_sim": scene_spec(
            reconstructed,
            m1_dir / "repaired_door.mjcf",
        ),
        "m2_sim": scene_spec(
            reconstructed,
            m2_dir / "repaired_m2.mjcf",
        ),
        "world": oracle,
    }

    task = TaskSpec(
        task_id="open_door",
        description=(
            "Open the door within the force "
            "and time budget."
        ),
        success_criteria={
            "door_angle_min": 0.2,
            "max_steps": 600,
            "max_pull_force_n": 20.0,
        },
    )

    policies = {
        "direct_pull": direct_pull,
        "operate_handle_then_pull":
            operate_handle_then_pull,
    }

    evaluator = CaPEvaluator()
    report = {}

    for policy_name, policy in policies.items():
        report[policy_name] = {}

        for scene_name, spec in scenes.items():
            result = evaluator.evaluate(
                policy, spec, task
            )

            report[policy_name][scene_name] = result

            success = result["rollouts"][0]["success"]
            print(
                f"{policy_name:27s} "
                f"{scene_name:11s} "
                f"success={success}"
            )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m1-dir", type=Path, required=True
    )
    parser.add_argument(
        "--m2-dir", type=Path, required=True
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs/m3_fixed.json",
    )

    args = parser.parse_args()

    run(
        args.m1_dir,
        args.m2_dir,
        args.out,
    )


if __name__ == "__main__":
    main()

