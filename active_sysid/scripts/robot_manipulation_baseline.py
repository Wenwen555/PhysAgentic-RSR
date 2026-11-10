"""R1: execute the fixed Panda reach/grasp/rotate/pull/verify program."""
import argparse
import json
from pathlib import Path

from active_sysid.backends.robot_door_backend import ROOT
from active_sysid.cap.robot_evaluator import evaluate_robot_policy
from policies.robot.fixed_open_door import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("oracle", "reconstructed"), default="oracle")
    parser.add_argument("--mechanism-mjcf", type=Path, help="Existing M1/M2 MJCF; its repair sidecar is read automatically")
    parser.add_argument("--out", type=Path, default=Path("runs/m3r_r1"))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--contact-force-limit", type=float, default=80.0, help="Measured sum of handle contact force magnitudes, N")
    args = parser.parse_args()
    result = evaluate_robot_policy(run, out_dir=args.out,
        scene=ROOT / f"worlds/robot/{args.scene}_robot_door.xml", mechanism_mjcf=args.mechanism_mjcf,
        observations="privileged", acceptance="r1", video=args.video,
        max_steps=args.max_steps, contact_force_limit_n=args.contact_force_limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
