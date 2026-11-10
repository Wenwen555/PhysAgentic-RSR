
"""在指定 Simulator 上执行 Coding Agent 生成的 policy.py。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import yaml

from active_sysid.cap.evaluator import (
    CaPEvaluator,
    scene_spec,
)
from active_sysid.types import TaskSpec


ROOT = Path(__file__).resolve().parents[1]


def load_policy(file: Path):
    spec = importlib.util.spec_from_file_location(
        "candidate_cap", file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy


def run(
    policy_file: Path,
    sim_mjcf: Path,
    output: Path,
):
    config = yaml.safe_load(
        (
            ROOT / "configs/worlds/legacy_object_space/reconstructed_door.yaml"
        ).read_text()
    )

    base_spec = {
        **config,
        "mjcf": str(ROOT / config["mjcf"]),
    }

    sim_spec = scene_spec(
        base_spec,
        sim_mjcf,
    )

    task = TaskSpec(
        task_id="open_door",
        description="Open the door within the execution budget.",
        success_criteria={
            "door_angle_min": 0.2,
            "max_steps": 600,
            "max_pull_force_n": 20.0,
        },
    )

    report = CaPEvaluator().run_in_sim(
        load_policy(policy_file),
        sim_spec,
        task,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    result = report["rollouts"][0]

    print(json.dumps({
        "success": result["success"],
        "elapsed_s": result["elapsed_s"],
        "action_calls": result["action_calls"],
        "final_observation": result["final_observation"],
        "feedback_path": str(output),
    }, indent=2))

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", type=Path, required=True
    )
    parser.add_argument(
        "--sim-mjcf", type=Path, required=True
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs/cap_feedback.json",
    )

    args = parser.parse_args()

    run(
        args.policy,
        args.sim_mjcf,
        args.out,
    )


if __name__ == "__main__":
    main()

