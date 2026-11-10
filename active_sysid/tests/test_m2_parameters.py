
"""M2: 参数编辑、参数映射及真实 MuJoCo 集成验收。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
import yaml

from active_sysid.identification.cma_es import CMASysID
from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.simulator.model_editor import ModelEditor


ROOT = Path(__file__).resolve().parents[1]


def test_m2_physical_parameters_are_written(tmp_path):
    source = (
        ROOT / "worlds/legacy_object_space/reconstructed"
        / "reconstructed_door.mjcf"
    )

    model = (
        HypothesisSet.initial_door_hypotheses()
        .get("H3_hidden_latch")
    )

    parameters = {
        "theta_unlock": 0.56,
        "I_d": 2.6,
        "b_d": 0.42,
        "mu_d": 0.11,
        "k_h": 0.51,
        "b_h": 0.081,
    }

    output = ModelEditor().apply(
        source,
        model,
        parameters,
        tmp_path / "m2.mjcf",
    )

    root = ET.parse(output).getroot()

    door = root.find(
        ".//joint[@name='door_hinge']"
    )
    handle = root.find(
        ".//joint[@name='handle_hinge']"
    )
    leaf = root.find(
        ".//geom[@name='door_leaf']"
    )

    assert float(
        door.get("damping")
    ) == pytest.approx(parameters["b_d"])

    assert float(
        door.get("frictionloss")
    ) == pytest.approx(parameters["mu_d"])

    assert float(
        handle.get("stiffness")
    ) == pytest.approx(parameters["k_h"])

    assert float(
        handle.get("damping")
    ) == pytest.approx(parameters["b_h"])

    # 验证写入 density 后，目标惯量确实改变。
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

    assert inertia == pytest.approx(
        parameters["I_d"]
    )

    assert root.find(
        ".//equality/weld[@name='latch_weld']"
    ) is not None

    sidecar = output.with_suffix(
        ".mjcf.repair.json"
    )

    metadata = json.loads(
        sidecar.read_text()
    )

    assert metadata["latch"][
        "unlock_threshold"
    ] == pytest.approx(0.56)


def test_m2_parameter_mapping_round_trip():
    cfg = yaml.safe_load(
        (
            ROOT / "configs/pipeline.yaml"
        ).read_text()
    )

    fitter = CMASysID(
        cfg["cma_es"]
    )

    bounds = (
        HypothesisSet.initial_door_hypotheses()
        .get("H3_hidden_latch")
        .param_bounds
    )

    values = {
        "I_d": 2.6,
        "b_d": 0.4,
        "mu_d": 0.1,
        "k_h": 0.5,
        "b_h": 0.08,
    }

    for key, value in values.items():
        normalized = fitter._encode(
            key, value, bounds[key]
        )

        assert 0 <= normalized <= 1

        restored = fitter._decode(
            key, normalized, bounds[key]
        )

        assert restored == pytest.approx(value)


def test_m2_end_to_end(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("cma")

    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"

    # 自动重跑 M1，再执行 M2。
    for name, args in (
        (
            "run_m1_structure.py",
            ["--output-dir", str(m1)],
        ),
        (
            "run_m2_parameters.py",
            [
                "--m1-dir", str(m1),
                "--output-dir", str(m2),
            ],
        ),
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / name),
                *args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            result.stdout
            + "\n"
            + result.stderr
        )

    report = json.loads(
        (
            m2 / "m2_report.json"
        ).read_text()
    )

    assert report["pass"]

    assert report["training_count"] == 7
    assert report["heldout_count"] == 5

    assert report["fit"]["backend"] == "cma-es"
    assert report["fit"]["n_evals"] > 0

    assert (
        report["structure_plus_parameters"]["traj_err"]
        < report["structure_only"]["traj_err"]
    )

    assert (
        report["structure_plus_parameters"]["event_err"]
        <= report["structure_only"]["event_err"]
    )

    saved = list(
        (m2 / "observations").glob("*.npz")
    )

    assert len(saved) == 12

