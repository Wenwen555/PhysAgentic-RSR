"""M0 基线: 建立初始物理预测误差。

两组实验, 在 oracle World 与重建 Sim 上各执行两遍 (可重复性):
1. locked_gentle_pull: 未旋转把手的 gentle_pull —— 期望 World 闩锁挡住门,
   Sim (缺闩锁) 直接开门, 门角度响应出现明显分歧;
2. unlocked_pull: 把手保持 0.7 rad (超过解锁阈值) 的 hold_and_pull ——
   期望两个场景都能正常开门。

验收 (M0): 同一拉门动作在两场景产生可重复、明显不同的门角度响应;
解锁后两场景都能转动。输出 runs/m0_baseline_<ts>/{baseline.json, curves.png},
作为后续 Simulator Repair 的基线。

    python scripts/run_m0_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from active_sysid.backends import create_backend
from active_sysid.probes.executor import ProbeExecutor
from active_sysid.probes.library import ProbeLibrary

ROOT = Path(__file__).resolve().parents[1]
REPEAT_ATOL = 1e-9  # 后端确定性: 同场景两次执行应逐位一致

EXPERIMENTS = [
    (
        "locked_gentle_pull",
        "gentle_pull",
        {
            "max_force": 10.0,
            "duration": 1.5,
        },
    ),
    (
        "locked_strong_pull",
        "gentle_pull",
        {
            "max_force": 40.0,
            "duration": 1.5,
        },
    ),
    (
        "below_threshold_pull",
        "hold_and_pull",
        {
            "handle_angle": 0.4,
            "max_force": 12.0,
            "duration": 1.5,
        },
    ),
    (
        "unlocked_pull",
        "hold_and_pull",
        {
            "handle_angle": 0.7,
            "max_force": 12.0,
            "duration": 1.5,
        },
    ),
]


def load_backend(name: str):
    spec = yaml.safe_load((ROOT / "configs" / "worlds" / f"{name}.yaml")
                          .read_text(encoding="utf-8"))
    mjcf = Path(spec["mjcf"])
    if not mjcf.is_absolute():
        spec["mjcf"] = str(ROOT / mjcf)
    return create_backend(spec)


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    features = cfg["features"]
    executor = ProbeExecutor(feature_eps=features)
    library = ProbeLibrary.load(ROOT / "configs" / "probes.yaml")

    world = load_backend("oracle_door")
    sim = load_backend("reconstructed_door")
    print(f"world = {world.name} (oracle),  sim = {sim.name} (reconstructed)")

    out_dir = ROOT / "runs" / f"m0_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    curves = {}
    all_pass = True

    for exp_name, template, overrides in EXPERIMENTS:
        spec = library.get(template).instantiate("door_0", overrides=overrides)
        result = {"probe": {"name": spec.name, "params": spec.params}}
        curves[exp_name] = {}

        for role, backend in (("world", world), ("sim", sim)):
            traj_a = backend.run_probe(spec)
            traj_b = backend.run_probe(spec)  # 可重复性

            repeatable = all(
	        np.allclose(
		    traj_a.channels[key],
		    traj_b.channels[key],
		    atol=REPEAT_ATOL,
		    rtol=0,
	        )
	        for key in traj_a.channels
	    )
	    
            evidence = executor.execute(spec, backend)
            result[role] = {
                "door_max": float(traj_a.channels["door_angle"].max()),
                "handle_max": float(traj_a.channels["handle_angle"].max()),
                "repeatable": repeatable,
                "summary": evidence.summary,
            }
            curves[exp_name][role] = {
                "t": traj_a.t,
                "door": traj_a.channels["door_angle"],
                "handle": traj_a.channels["handle_angle"],
            }

        door_world, door_sim = result["world"]["door_max"], result["sim"]["door_max"]
        diff = np.abs(curves[exp_name]["world"]["door"] - curves[exp_name]["sim"]["door"])
        result["diff_max"] = float(diff.max())
        curves[exp_name]["diff"] = {"t": curves[exp_name]["world"]["t"], "door": diff}

        opens = features["door_opens_eps"]
        immobile = features["door_immobile_eps"]
        if exp_name  in (
	    "locked_gentle_pull",
	    "locked_strong_pull",
	    "below_threshold_pull",
	):
            result["acceptance"] = {
                "world_immobile": door_world < immobile,
                "sim_opens": door_sim > opens,
                "clearly_different": result["diff_max"] > opens,
            }
        else:
            result["acceptance"] = {
                "world_opens": door_world > opens,
                "sim_opens": door_sim > opens,
                "handle_reached_target_region": (
		    result["world"]["handle_max"] > 0.6
		    and result["sim"]["handle_max"] > 0.6
		),
            }
            
        result["acceptance"]["repeatable"] = (
            result["world"]["repeatable"] and result["sim"]["repeatable"])
        ok = all(result["acceptance"].values())
        all_pass &= ok
        records[exp_name] = result

        print(f"\n[{exp_name}] {spec.name} {overrides}")
        print(f"  world door_max={door_world:.4f}  sim door_max={door_sim:.4f}  "
              f"diff_max={result['diff_max']:.4f}")
        print(f"  acceptance: {result['acceptance']} -> {'PASS' if ok else 'FAIL'}")

    (out_dir / "baseline.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    _plot(curves, out_dir / "curves.png")
    print(f"\nartifacts -> {out_dir}")
    print("M0 baseline:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


def _plot(curves: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 不可用，跳过曲线绘制")
        return

    fig, axes = plt.subplots(
        len(curves),
        2,
        figsize=(13, 3.1 * len(curves)),
        squeeze=False,
    )

    for row, (name, data) in enumerate(
        curves.items()
    ):
        for col, channel in enumerate(
            ("door", "handle")
        ):
            ax = axes[row, col]

            ax.plot(
                data["world"]["t"],
                data["world"][channel],
                label="World",
            )

            ax.plot(
                data["sim"]["t"],
                data["sim"][channel],
                label="Sim",
            )

            if channel == "door":
                ax.plot(
                    data["diff"]["t"],
                    data["diff"]["door"],
                    "--",
                    label="|World - Sim|",
                )

            ax.set_title(f"{name} / {channel}")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("angle [rad]")
            ax.legend()
            ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    sys.exit(main())
