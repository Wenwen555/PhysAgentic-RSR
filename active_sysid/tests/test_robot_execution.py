"""Core execution integrity: actual contact opens the door; no-contact cannot."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
from active_sysid.robot.fixture_backend import RobotDoorBackend
from active_sysid.robot.controller import ExecutionError
from active_sysid.robot.runtime import RobotRuntime
from active_sysid.robot.fixture_policy import run, reach_and_grasp


def test_robot_contact_execution():
    backend = RobotDoorBackend()
    api = RobotRuntime(backend, observations="privileged")
    result = run(api)
    assert result["door_open_verified"]
    assert backend.sample()["door_angle_rad"] >= 0.30
    audit = backend.model_audit()
    assert audit["robot_only_actuators"] and not audit["robot_object_constraint"]
    assert audit["mocap_bodies"] == 0
    for sample in backend.samples:
        assert np.all(sample["object_applied_torque_nm"] == 0)
        assert np.all(sample["object_actuator_torque_nm"] == 0)
        assert sample["object_xfrc_max"] == 0
        assert np.all(np.isfinite(sample["q"]))
        if sample["phase"] in {"rotate", "pull", "verify"}:
            assert np.all(sample["finger_contact_normal_n"] > 0.5)


def test_contact_disabled_cannot_grasp():
    backend = RobotDoorBackend()
    handle = backend.model.geom("handle_bar").id
    backend.model.geom_contype[handle] = 0
    backend.model.geom_conaffinity[handle] = 0
    api = RobotRuntime(backend, observations="privileged")
    with pytest.raises(ExecutionError, match="bilateral"):
        reach_and_grasp(api)
    assert abs(backend.sample()["door_angle_rad"]) < 1e-5
    assert max(s["total_handle_contact_force_n"] for s in backend.samples) == 0


def test_robot_observation_does_not_expose_object_state():
    api = RobotRuntime(RobotDoorBackend())
    observation = api.observe()
    assert "privileged" not in observation
    assert "door_angle_rad" not in observation
    assert "handle_angle_rad" not in observation
    assert "handle_in_tcp_m" not in observation
    assert not hasattr(api, "step")


def test_pull_without_rotation_stays_latched():
    backend = RobotDoorBackend()
    api = RobotRuntime(backend, observations="privileged")
    reach_and_grasp(api)
    scene = api.observe()["privileged"]
    api.phase("pull_without_rotate")
    api.move_arc(scene["door_pivot_m"], scene["door_axis_world"], 0.45, duration_s=4.2)
    assert backend.latched
    assert abs(backend.sample()["door_angle_rad"]) < 0.01
    assert max(s["handle_angle_rad"] for s in backend.samples) < backend.unlock_threshold
