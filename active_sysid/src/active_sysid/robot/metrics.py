"""Acceptance metrics derived from recorded robot physics, not action status."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _plain(value: Any) -> Any:
    """Produce JSON values, using null for unavailable/non-finite measurements."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize_robot_episode(
    backend,
    *,
    error=None,
    action_count=0,
    policy_result=None,
    goal_angle_rad=0.30,
    contact_force_limit_n=80.0,
    unexpected_contact_limit_n=5.0,
    acceptance="r1",
) -> dict:
    """Evaluate R0, fixed R1, or a generic Robot-Space CaP episode.

    R0 requires measured reach, sustained grasp and robot joint motion. R1
    additionally requires rotation, opening, bilateral contact and limited
    handle pose drift relative to the TCP. Generic ``task`` acceptance uses
    the observed final door angle and common physical execution gates; a CaP
    need not adopt the fixed baseline's phase names or grasping strategy.

    Force thresholds are experiment acceptance limits. The controller is a
    joint position PD servo with bias compensation and measured-force abort.
    Translation/rotation drift measures relative pose change during rotation
    and pulling; it is not an estimate of surface slip velocity.
    """
    if acceptance not in ("r0", "r1", "task"):
        raise ValueError("acceptance must be 'r0', 'r1', or 'task'")

    samples = backend.samples
    count = len(samples)
    dt = float(backend.dt)
    phase = np.asarray([sample["phase"] for sample in samples], dtype=str)
    fields = {
        "time_s": (), "q": (7,), "joint_torque_nm": (7,),
        "finger_contact_normal_n": (2,), "handle_contact_force_world_n": (3,),
        "max_contact_force_n": (), "total_handle_contact_force_n": (),
        "unexpected_contact_force_n": (), "self_contact_force_n": (),
        "door_angle_rad": (), "handle_angle_rad": (), "latch_active": (),
        "handle_in_tcp_m": (3,), "object_applied_torque_nm": (2,),
        "object_actuator_torque_nm": (2,), "object_xfrc_max": (),
    }
    data = {
        name: np.asarray([sample[name] for sample in samples], dtype=float)
        .reshape((count, *shape))
        for name, shape in fields.items()
    }
    relative_quat = (
        np.asarray([sample["handle_in_tcp_quat_wxyz"] for sample in samples], dtype=float)
        if count and "handle_in_tcp_quat_wxyz" in samples[0] else None
    )
    times = data["time_s"]
    elapsed = float(times[-1] - times[0] + dt) if count else 0.0
    bilateral = np.all(data["finger_contact_normal_n"] > 0.5, axis=1)
    grasp = phase == "grasp"
    rotate = phase == "rotate"
    pull = phase == "pull"
    manipulation = rotate | pull
    window_steps = max(1, int(np.ceil(0.2 / dt - 1e-9)))

    def peak(name):
        return float(np.max(data[name])) if count else None

    def vector_peak(name, width):
        return np.max(np.abs(data[name]), axis=0).tolist() if count else [None] * width

    per_phase = {}
    for name in dict.fromkeys(phase.tolist()):
        mask = phase == name
        phase_steps = int(np.count_nonzero(mask))
        contact_steps = int(np.count_nonzero(mask & bilateral))
        per_phase[name] = {
            "steps": phase_steps,
            "duration_s": phase_steps * dt,
            "bilateral_contact_steps": contact_steps,
            "bilateral_contact_duration_s": contact_steps * dt,
            "bilateral_contact_ratio": contact_steps / phase_steps,
        }

    grasp_indices = np.flatnonzero(grasp)
    grasp_tail = grasp_indices[-window_steps:]
    tail_complete = (
        len(grasp_tail) == window_steps and np.all(np.diff(grasp_tail) == 1)
    )
    grasp_tail_ratio = float(np.mean(bilateral[grasp_tail])) if len(grasp_tail) else None

    # Select the last sufficiently long *continuous* bilateral grasp interval,
    # then average its final 0.2 s. Interrupted contact is not a drift reference.
    transitions = np.diff((grasp & bilateral).astype(int), prepend=0, append=0)
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    reference_indices = np.empty(0, dtype=int)
    for start, end in reversed(list(zip(starts, ends))):
        if end - start >= window_steps:
            reference_indices = np.arange(end - window_steps, end)
            break
    reference = None
    translation_drift = rotation_drift = None
    if len(reference_indices):
        reference_position = np.mean(data["handle_in_tcp_m"][reference_indices], axis=0)
        reference = {
            "start_time_s": float(times[reference_indices[0]]),
            "end_time_s": float(times[reference_indices[-1]]),
            "duration_s": len(reference_indices) * dt,
            "sample_count": len(reference_indices),
            "mean_handle_in_tcp_m": reference_position.tolist(),
            "handle_in_tcp_quat_wxyz": None,
        }
        if np.any(manipulation):
            translation_drift = float(np.max(np.linalg.norm(
                data["handle_in_tcp_m"][manipulation] - reference_position, axis=1
            )))
        if relative_quat is not None:
            reference_quat = relative_quat[reference_indices[-1]]
            reference["handle_in_tcp_quat_wxyz"] = reference_quat.tolist()
            measured_quat = relative_quat[manipulation]
            norms = np.linalg.norm(measured_quat, axis=1) * np.linalg.norm(reference_quat)
            if len(norms) and np.all(norms > 1e-12):
                cosine = np.abs(measured_quat @ reference_quat) / norms
                rotation_drift = float(np.max(2 * np.arccos(np.clip(cosine, 0.0, 1.0))))

    manipulation_steps = int(np.count_nonzero(manipulation))
    manipulation_ratio = (
        float(np.mean(bilateral[manipulation])) if manipulation_steps else None
    )
    unexpected = data["unexpected_contact_force_n"] > unexpected_contact_limit_n
    self_contact = data["self_contact_force_n"] > unexpected_contact_limit_n
    harmful = unexpected | self_contact
    joint_limits = np.asarray(
        backend.model.actuator_forcerange[backend.arm_actuators], dtype=float
    )
    torque = data["joint_torque_nm"]
    torque_within_limits = bool(count and np.all(
        (torque >= joint_limits[:, 0] - 1e-6)
        & (torque <= joint_limits[:, 1] + 1e-6)
    ))
    joint_motion = np.ptp(data["q"], axis=0) if count else np.zeros(7)
    final_door = float(data["door_angle_rad"][-1]) if count else None
    final_handle = float(data["handle_angle_rad"][-1]) if count else None
    aggregate_force = peak("total_handle_contact_force_n")
    single_force = peak("max_contact_force_n")
    applied_max = float(np.max(np.abs(data["object_applied_torque_nm"]))) if count else None
    actuator_max = float(np.max(np.abs(data["object_actuator_torque_nm"]))) if count else None
    external_max = peak("object_xfrc_max")
    latch_enabled = bool(backend.latch_enabled)
    active = data["latch_active"].astype(bool)
    released = np.r_[False, active[:-1] & ~active[1:]] if count else np.zeros(0, dtype=bool)
    rotate_threshold = float(backend.unlock_threshold) if latch_enabled else 0.55
    rotate_reached = bool(np.any(rotate & (data["handle_angle_rad"] >= rotate_threshold)))
    rotate_released = bool(np.any(rotate & released))
    audit = backend.model_audit()
    policy_result = policy_result if isinstance(policy_result, Mapping) else {}
    reach_position_error = policy_result.get("reach_position_error_m")
    reach_orientation_error = policy_result.get("reach_orientation_error_rad")

    def within(value, limit):
        return bool(value is not None and np.isfinite(value) and 0 <= value <= limit)

    gates = {
        "samples_recorded": bool(count),
        "finite_telemetry": bool(count and all(
            np.all(np.isfinite(values)) for values in data.values()
        ) and (relative_quat is None or np.all(np.isfinite(relative_quat)))),
        "monotonic_time": bool(count and np.all(np.diff(times) > 0)),
        "execution_completed": error is None,
        "robot_physics": bool(
            audit["robot_only_actuators"] and not audit["robot_object_constraint"]
            and audit["mocap_bodies"] == 0 and audit["handle_collision_enabled"]
        ),
        "no_object_actuation": bool(
            count and applied_max <= 1e-12 and actuator_max <= 1e-12 and external_max <= 1e-12
        ),
        "contact_force_limit": bool(
            count and aggregate_force <= contact_force_limit_n and single_force <= contact_force_limit_n
        ),
        "no_harmful_collision": bool(count and not np.any(harmful)),
        "joint_torque_limits": torque_within_limits,
        "robot_motion": bool(count and np.any(joint_motion > 0.02)),
        "reach": within(reach_position_error, 0.01) and within(reach_orientation_error, 0.10),
        "grasp": bool(tail_complete and grasp_tail_ratio >= 0.95),
        "rotate": rotate_reached and (rotate_released if latch_enabled else True),
        "pull": bool(np.any(pull) and final_door is not None and final_door >= goal_angle_rad),
        "verify": bool(final_door is not None and final_door >= goal_angle_rad),
        "bilateral_manipulation_contact": bool(
            manipulation_ratio is not None and manipulation_ratio >= 0.95
        ),
        "translation_drift": within(translation_drift, 0.01),
        "rotation_drift": within(rotation_drift, 0.15),
    }
    required = [
        "samples_recorded", "finite_telemetry", "monotonic_time", "execution_completed",
        "robot_physics", "no_object_actuation", "contact_force_limit",
        "no_harmful_collision", "joint_torque_limits",
    ]
    if acceptance == "r0":
        required += ["robot_motion", "reach", "grasp"]
    elif acceptance == "r1":
        required += ["reach", "grasp", "rotate", "pull", "verify",
                     "bilateral_manipulation_contact", "translation_drift", "rotation_drift"]
    else:
        required += ["verify"]
    success = all(gates[name] for name in required)

    return _plain({
        "status": "pass" if success else "fail", "success": success,
        "acceptance": acceptance, "execution_error": str(error) if error is not None else None,
        "action_count": int(action_count), "physics_steps": count, "elapsed_s": elapsed,
        "physics_timestep_s": dt, "final_door_angle_rad": final_door,
        "final_handle_angle_rad": final_handle, "max_handle_angle_rad": peak("handle_angle_rad"),
        "max_joint_torque_nm": vector_peak("joint_torque_nm", 7),
        "joint_torque_limits_nm": joint_limits.tolist(),
        "joint_motion_range_rad": joint_motion.tolist() if count else [None] * 7,
        "peak_finger_normal_n": vector_peak("finger_contact_normal_n", 2),
        "peak_aggregate_handle_contact_force_n": aggregate_force,
        "peak_net_handle_force_n": float(np.max(np.linalg.norm(
            data["handle_contact_force_world_n"], axis=1
        ))) if count else None,
        "peak_single_contact_force_n": single_force,
        "peak_unexpected_contact_force_n": peak("unexpected_contact_force_n"),
        "peak_self_contact_force_n": peak("self_contact_force_n"),
        "harmful_contact_steps": int(np.count_nonzero(harmful)),
        "harmful_contact_duration_s": int(np.count_nonzero(harmful)) * dt,
        "unexpected_contact_steps": int(np.count_nonzero(unexpected)),
        "unexpected_contact_duration_s": int(np.count_nonzero(unexpected)) * dt,
        "self_contact_steps": int(np.count_nonzero(self_contact)),
        "self_contact_duration_s": int(np.count_nonzero(self_contact)) * dt,
        "max_abs_object_applied_torque_nm": applied_max,
        "max_abs_object_actuator_torque_nm": actuator_max,
        "max_object_xfrc_component": external_max,
        "per_phase": per_phase, "grasp_final_window_bilateral_ratio": grasp_tail_ratio,
        "grasp_final_window_duration_s": len(grasp_tail) * dt,
        "rotate_pull_bilateral_contact_ratio": manipulation_ratio,
        "rotate_pull_bilateral_contact_duration_s": int(np.count_nonzero(manipulation & bilateral)) * dt,
        "pose_drift_reference": reference,
        "max_translation_drift_m": translation_drift, "max_rotation_drift_rad": rotation_drift,
        "reach_position_error_m": reach_position_error,
        "reach_orientation_error_rad": reach_orientation_error,
        "latch_enabled": latch_enabled, "latch_release_observed": bool(np.any(released)),
        "latch_release_observed_during_rotate": rotate_released,
        "model_audit": audit, "gates": gates, "required_gates": required,
        "failed_required_gates": [name for name in required if not gates[name]],
        "criteria": {
            "door_goal_angle_rad": float(goal_angle_rad),
            "aggregate_handle_force_limit_n": float(contact_force_limit_n),
            "unexpected_or_self_contact_limit_n": float(unexpected_contact_limit_n),
            "bilateral_each_finger_min_normal_n": 0.5,
            "grasp_final_window_s": 0.2, "bilateral_contact_min_ratio": 0.95,
            "reach_position_error_limit_m": 0.01, "reach_orientation_error_limit_rad": 0.10,
            "rotate_handle_threshold_rad": rotate_threshold,
            "translation_drift_limit_m": 0.01, "rotation_drift_limit_rad": 0.15,
            "object_actuation_zero_tolerance": 1e-12,
        },
        "measurement_definitions": {
            "aggregate_handle_force": "Sum of norms of robot-handle contact forces",
            "net_handle_force": "Norm of the summed world-frame force on the handle",
            "pose_drift": "Handle pose change relative to TCP during rotate and pull",
            "object_xfrc_component": "Largest absolute applied object wrench component (N or N m)",
        },
        "control": {
            "joint_servo": "Position PD with bias compensation",
            "contact_limit_response": "Abort on measured force threshold",
            "cartesian_force_regulation": False,
        },
    })
