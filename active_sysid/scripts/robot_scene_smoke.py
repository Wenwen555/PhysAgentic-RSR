"""R0: compile the scene, move actual robot joints and establish bilateral contact."""
import argparse
import json
from pathlib import Path
from active_sysid.backends.robot_door_backend import ROOT
from active_sysid.cap.robot_evaluator import evaluate_robot_policy
from policies.robot.fixed_open_door import reach_and_grasp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/m3r_r0"))
    parser.add_argument("--scene", choices=("oracle", "reconstructed"), default="oracle")
    parser.add_argument("--mechanism-mjcf", type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    result = evaluate_robot_policy(reach_and_grasp, out_dir=args.out,
        scene=ROOT / f"worlds/robot/{args.scene}_robot_door.xml", mechanism_mjcf=args.mechanism_mjcf,
        observations="privileged", acceptance="r0", video=args.video)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
