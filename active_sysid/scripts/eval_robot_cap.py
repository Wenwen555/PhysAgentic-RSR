"""Evaluate a new robot-API policy exposing run(api); no policy generation is performed."""
import argparse
import importlib.util
import json
from pathlib import Path
from active_sysid.backends.robot_door_backend import ROOT
from active_sysid.cap.robot_evaluator import evaluate_robot_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", choices=("oracle", "reconstructed"), default="oracle")
    parser.add_argument("--mechanism-mjcf", type=Path)
    parser.add_argument("--observations", choices=("robot", "privileged"), default="robot")
    parser.add_argument("--goal-angle", type=float, default=0.30)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("evaluated_robot_policy", args.policy.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = evaluate_robot_policy(module.run, out_dir=args.out,
        scene=ROOT / f"worlds/robot/{args.scene}_robot_door.xml", mechanism_mjcf=args.mechanism_mjcf,
        observations=args.observations, acceptance="task", video=args.video,
        goal_angle_rad=args.goal_angle, max_steps=args.max_steps)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["success"] else 1)


if __name__ == "__main__":
    main()
