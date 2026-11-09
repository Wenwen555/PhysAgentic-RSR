"""入口脚本: 跑一次完整 probe loop。

    # 冒烟演示 (mock 双门 World)
    python scripts/run_probe_loop.py --backend mock

    # MuJoCo 后端 (object-space 直驱; channel_map 等见 configs/mujoco_world.example.yaml)
    python scripts/run_probe_loop.py --backend mujoco --mjcf worlds/reconstructed/x.mjcf
    python scripts/run_probe_loop.py --backend mujoco --world-config my_world.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from active_sysid import TaskSpec
from active_sysid.backends import create_backend
from active_sysid.identification.cma_es import CMASysID
from active_sysid.identification.mechanism import MechanismIdentifier
from active_sysid.pipeline import ProbeLoop
from active_sysid.planner.probe_planner import ProbePlanner
from active_sysid.probes.executor import ProbeExecutor
from active_sysid.probes.library import ProbeLibrary
from active_sysid.simulator.model_editor import ModelEditor
from active_sysid.simulator.validator import Validator

ROOT = Path(__file__).resolve().parents[1]


def _build_world_spec(args) -> dict:
    if args.backend == "mujoco":
        if args.world_config:
            spec = yaml.safe_load(args.world_config.read_text(encoding="utf-8"))
            if args.mjcf:
                spec["mjcf"] = str(args.mjcf)
            return spec
        if not args.mjcf:
            raise SystemExit("--backend mujoco 需要 --mjcf 或 --world-config")
        return {"backend": "mujoco", "mjcf": str(args.mjcf)}
    return {"backend": "mock"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "mujoco"], default="mock")
    parser.add_argument("--mjcf", type=Path, default=None,
                        help="MuJoCo World 场景文件")
    parser.add_argument("--world-config", type=Path, default=None,
                        help="后端构造 yaml, 见 configs/mujoco_world.example.yaml")
    parser.add_argument("--source-mjcf", type=Path, default=None,
                        help="初始重建 MJCF; 提供后每步输出修复副本")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="默认 runs/<timestamp>")
    parser.add_argument("--max-probes", type=int, default=None)
    args = parser.parse_args()

    world = create_backend(_build_world_spec(args))

    cfg = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    if args.max_probes:
        cfg["probe_loop"]["max_probes"] = args.max_probes
    output_dir = args.output_dir or (ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S"))

    loop = ProbeLoop(
        planner=ProbePlanner(weights=cfg["weights"]),
        library=ProbeLibrary.load(ROOT / "configs" / "probes.yaml"),
        executor=ProbeExecutor(feature_eps=cfg["features"],
                               record_dir=Path(output_dir) / "probes"),
        mechanism_identifier=MechanismIdentifier(),
        parameter_identifier=CMASysID(cfg["cma_es"]),
        model_editor=ModelEditor(),
        validator=Validator(thresholds=cfg["validator"]["thresholds"]),
        config=cfg,
        source_mjcf=args.source_mjcf,
        output_dir=output_dir,
    )
    task = TaskSpec(task_id="open_left_door",
                    description="打开左侧房门, 使机器人能够进入下一个房间")
    report = loop.run(task, scene={"objects": ["door_0"]}, world=world)

    print(f"probes={report.n_probes} stop={report.stop_reason}")
    print(f"best={report.best_hypothesis['hid']}")
    print(f"params={report.best_hypothesis['params']}")
    print(f"report -> {Path(output_dir) / 'report.json'}")


if __name__ == "__main__":
    main()
