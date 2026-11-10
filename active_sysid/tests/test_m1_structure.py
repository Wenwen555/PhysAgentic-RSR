
"""M1: 机构识别、结构修复及 held-out 预测验收。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from active_sysid.identification.mechanism import MechanismIdentifier
from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.simulator.model_editor import ModelEditor
from active_sysid.types import (
    Evidence,
    ProbeSpec,
    ProbeTrajectory,
)


ROOT = Path(__file__).resolve().parents[1]


def _evidence(
    name: str,
    handle: float,
    door: float,
) -> Evidence:
    t = np.arange(10) * 0.05

    trajectory = ProbeTrajectory(
        t=t,
        channels={
            "handle_angle": np.linspace(
                0.0, handle, len(t)
            ),
            "door_angle": np.full(
                len(t), door
            ),
        },
    )

    return Evidence(
        probe=ProbeSpec(
            name=name,
            target="door_0",
            params={"handle_angle": handle},
        ),
        trajectory=trajectory,
        summary={
            "door_immobile": door < 0.02,
            "door_opens": door > 0.2,
        },
    )


def test_mechanism_threshold_is_inferred_from_observed_angles():
    hypotheses = (
        HypothesisSet.initial_door_hypotheses()
    )

    evidences = [
        _evidence("gentle_pull", 0.0, 0.0),
        _evidence("hold_and_pull", 0.40, 0.0),
        _evidence("hold_and_pull", 0.70, 0.5),
    ]

    identified = MechanismIdentifier().update(
        hypotheses,
        evidences,
    )

    assert identified.hid == "H3_hidden_latch"

    assert (
        identified.params["theta_unlock"]
        == pytest.approx(0.55)
    )

    assert (
        identified.param_bounds["theta_unlock"]
        == pytest.approx((0.4, 0.7))
    )


def test_model_editor_inserts_executable_constraint(
    tmp_path,
):
    source = (
        ROOT / "worlds/legacy_object_space/reconstructed"
        / "reconstructed_door.mjcf"
    )

    hypothesis = (
        HypothesisSet.initial_door_hypotheses()
        .get("H3_hidden_latch")
    )

    out = ModelEditor().apply(
        source,
        hypothesis,
        {"theta_unlock": 0.55},
        tmp_path / "repaired.mjcf",
    )

    original_weld = ET.parse(source).find(
        ".//equality/weld"
    )
    assert original_weld is None

    weld = ET.parse(out).find(
        ".//equality/weld[@name='latch_weld']"
    )

    assert weld is not None
    assert weld.attrib["body1"] == "door"
    assert weld.attrib["body2"] == "frame"

    sidecar = out.with_suffix(
        ".mjcf.repair.json"
    )

    metadata = json.loads(
        sidecar.read_text(encoding="utf-8")
    )

    assert (
        metadata["repair_mode"]
        == "functional_latch"
    )

    assert (
        metadata["latch"]["unlock_threshold"]
        == pytest.approx(0.55)
    )


def test_m1_heldout_behavior_and_trajectory_improves(
    tmp_path,
):
    pytest.importorskip("mujoco")

    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT / "scripts"
                / "run_m1_structure.py"
            ),
            "--output-dir",
            str(tmp_path / "m1"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, (
        result.stdout + "\n" + result.stderr
    )

    report = json.loads(
        (
            tmp_path / "m1"
            / "m1_report.json"
        ).read_text(encoding="utf-8")
    )

    assert report["pass"]

    assert (
        report["mechanism"]
        == "H3_hidden_latch"
    )

    assert (
        report["before"]["event_err"]
        > 0
    )

    assert (
        report["after"]["event_err"]
        == 0
    )

    assert (
        report["after"]["traj_err"]
        < report["before"]["traj_err"]
    )

    assert len(report["training_probes"]) == 7
    assert len(report["heldout_probes"]) == 3

