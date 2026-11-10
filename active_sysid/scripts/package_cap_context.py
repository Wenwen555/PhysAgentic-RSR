
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def package(m2_dir: Path, out: Path):
    config = yaml.safe_load(
        (
            ROOT / "configs/worlds/reconstructed_door.yaml"
        ).read_text()
    )
    config["mjcf"] = "models/repaired_m2.mjcf"

    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        out,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        z.writestr(
            "sim_backend.yaml",
            yaml.safe_dump(config, sort_keys=False),
        )

        for name in (
            "repaired_m2.mjcf",
            "repaired_m2.mjcf.repair.json",
        ):
            z.write(
                m2_dir / name,
                f"models/{name}",
            )

        for path in (
            ROOT / "src/active_sysid"
        ).rglob("*.py"):
            z.write(
                path,
                path.relative_to(ROOT),
            )

        for name in (
            "eval_cap_policy.py",
            "run_m3_fixed.py",
        ):
            z.write(
                ROOT / "scripts" / name,
                f"scripts/{name}",
            )

        z.write(
            ROOT / "pyproject.toml",
            "pyproject.toml",
        )

    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m2-dir", type=Path, required=True
    )
    parser.add_argument(
        "--out", type=Path, required=True
    )
    args = parser.parse_args()

    package(args.m2_dir, args.out)

