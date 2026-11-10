"""M2 import preserves robot/contact properties and copies fitted mechanics."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from active_sysid.backends.robot_mechanism import apply_mechanism
from active_sysid.planner.hypothesis import HypothesisSet
from active_sysid.simulator.model_editor import ModelEditor


ROOT = Path(__file__).resolve().parents[1]


def test_import_fitted_mechanism_preserves_robot_and_contact_layer(tmp_path):
    hypothesis = HypothesisSet.initial_door_hypotheses().get("H3_hidden_latch")
    source_path = ModelEditor().apply(
        ROOT / "worlds/reconstructed/reconstructed_door.mjcf",
        hypothesis,
        {"I_d": 2.6, "b_d": 0.42, "mu_d": 0.11, "k_h": 0.51,
         "b_h": 0.081, "theta_h0": 0.12, "theta_unlock": 0.59},
        tmp_path / "repaired_m2.mjcf",
    )
    source = mj.MjModel.from_xml_path(str(source_path))
    model = mj.MjModel.from_xml_string("""
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="robot" pos="0 -1 0">
          <joint name="robot_joint" damping="2.3"/>
          <geom name="robot_geom" type="sphere" size="0.1" mass="2"/>
        </body>
        <body name="door">
          <joint name="door_hinge" axis="0 0 1" range="0 1.3"/>
          <geom name="door_leaf" type="box" size=".4 .02 1"
                pos=".4 0 1.05" density="42" friction=".7 .01 .002"/>
          <body name="handle" pos=".75 -.1 1">
            <joint name="handle_hinge" axis="0 -1 0" range="-1 1"/>
            <geom name="handle_bar" type="box" size=".08 .02 .02"
                  mass=".4" friction="1.1 .02 .003"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """)
    robot_mass = model.body("robot").mass.copy()
    robot_damping = model.joint("robot_joint").damping.copy()
    handle_mass = model.body("handle").mass.copy()
    handle_inertia = model.body("handle").inertia.copy()
    handle_axis = model.joint("handle_hinge").axis.copy()
    geometry = {field: getattr(model, field).copy() for field in
                ("geom_size", "geom_pos", "geom_friction", "geom_contype", "geom_conaffinity")}

    result = apply_mechanism(model, source_path)
    data = mj.MjData(model)
    mj.mj_setConst(model, data)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    for name in ("door_hinge", "handle_hinge"):
        for field in ("damping", "frictionloss", "armature", "stiffness", "range", "limited"):
            np.testing.assert_allclose(getattr(model.joint(name), field),
                                       getattr(source.joint(name), field))
        q = int(model.joint(name).qposadr[0])
        source_q = int(source.joint(name).qposadr[0])
        assert model.qpos_spring[q] == pytest.approx(source.qpos_spring[source_q])
    for field in ("mass", "inertia", "ipos", "iquat"):
        np.testing.assert_allclose(getattr(model.body("door"), field),
                                   getattr(source.body("door"), field))
    np.testing.assert_array_equal(model.body("robot").mass, robot_mass)
    np.testing.assert_array_equal(model.joint("robot_joint").damping, robot_damping)
    np.testing.assert_array_equal(model.body("handle").mass, handle_mass)
    np.testing.assert_array_equal(model.body("handle").inertia, handle_inertia)
    np.testing.assert_array_equal(model.joint("handle_hinge").axis, handle_axis)
    for field, original in geometry.items():
        np.testing.assert_array_equal(getattr(model, field), original)
    assert result["latch"]["enabled"]
    assert result["latch"]["unlock_threshold"] == pytest.approx(0.59)
    assert result["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert result["sidecar_sha256"] == hashlib.sha256(
        source_path.with_suffix(".mjcf.repair.json").read_bytes()).hexdigest()
