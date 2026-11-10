"""One robot episode: run a policy, score the observed trace, save the evidence."""
from __future__ import annotations
import hashlib
import inspect
import platform
from pathlib import Path
import mujoco
import numpy as np

from active_sysid.backends.robot_door_backend import RobotDoorBackend
from active_sysid.backends.robot_mechanism import apply_mechanism
from active_sysid.cap.robot_runtime import RobotRuntime
from active_sysid.cap.robot_controller import quat_to_matrix, rotation_vector
from active_sysid.cap.robot_recording import RobotRecorder
from active_sysid.cap.robot_metrics import summarize_robot_episode


def evaluate_robot_policy(policy, *, out_dir, scene=None, mechanism_mjcf=None,
                          observations="robot", acceptance="task", video=False,
                          max_steps=30000, contact_force_limit_n=80.0,
                          unexpected_contact_limit_n=5.0, goal_angle_rad=0.30):
    backend = RobotDoorBackend(scene, mechanism_mjcf=mechanism_mjcf)
    api = RobotRuntime(backend, observations=observations, max_steps=max_steps,
                       contact_force_limit_n=contact_force_limit_n,
                       unexpected_contact_limit_n=unexpected_contact_limit_n)
    error = None
    result = None
    with RobotRecorder(out_dir, backend.model, video=video) as recorder:
        backend.callbacks.append(recorder.capture)
        try:
            result = policy(api)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            api.close()
        # Acceptance recomputes reach from the action target and actual trace.
        # A policy's own success/error claims are never the source of this gate.
        measured_result = {}
        reach_commands = [a for a in api.actions if a["phase"] == "reach" and a["action"] == "move_ee"]
        reach_samples = [s for s in backend.samples if s["phase"] == "reach"]
        if reach_commands and reach_samples:
            goal, actual = reach_commands[-1]["arguments"], reach_samples[-1]
            measured_result["reach_position_error_m"] = float(np.linalg.norm(
                np.asarray(goal["position_m"]) - actual["ee_pos_m"]))
            measured_result["reach_orientation_error_rad"] = float(np.linalg.norm(rotation_vector(
                quat_to_matrix(goal["quaternion_wxyz"]) @ quat_to_matrix(actual["ee_quat_wxyz"]).T)))
        summary = summarize_robot_episode(
            backend, error=error, action_count=len(api.actions), policy_result=measured_result,
            goal_angle_rad=goal_angle_rad, contact_force_limit_n=contact_force_limit_n,
            unexpected_contact_limit_n=unexpected_contact_limit_n, acceptance=acceptance,
        )
        summary["policy_return"] = result
        source_file = inspect.getsourcefile(policy)
        robot_xml = backend.scene.parent / "panda_robot.xml"
        implementation_files = [
            Path(__file__), Path(inspect.getfile(RobotDoorBackend)),
            Path(inspect.getfile(RobotRuntime)),
            Path(inspect.getfile(api._adapter.__class__)),
            Path(inspect.getfile(api._adapter.controller.__class__)),
            Path(inspect.getfile(summarize_robot_episode)),
            Path(inspect.getfile(apply_mechanism)),
            Path(inspect.getfile(RobotRecorder)),
        ]
        summary["provenance"] = {
            "python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
            "scene": str(backend.scene), "scene_sha256": hashlib.sha256(backend.scene.read_bytes()).hexdigest(),
            "robot_xml_sha256": hashlib.sha256(robot_xml.read_bytes()).hexdigest(),
            "policy_file": str(source_file),
            "policy_sha256": hashlib.sha256(Path(source_file).read_bytes()).hexdigest() if source_file else None,
            "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation_files},
            "mechanism": backend.mechanism,
            "observations": observations, "acceptance": acceptance,
            "dt_s": backend.dt, "video_fps": recorder.fps if video else None,
            "video_frame_count": recorder.video_frame_count,
        }
        recorder.finish(summary, api.actions)
    return summary
