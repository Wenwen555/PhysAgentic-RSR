"""R1 privileged fixed program; all manipulation decisions are visible here.

This is an execution-capability baseline, not a Coding-Agent-generated CaP.
It uses known poses to plan explicit approach, pinch, lever rotation and pull.
"""
import numpy as np
from active_sysid.robot.controller import ExecutionError, matrix_to_quat, rotation_vector


def reach_and_grasp(api):
    api.phase("settle")
    api.hold(0.5)
    scene = api.observe()["privileged"]
    position = scene["handle_grasp_pos_m"]
    # Tool z points into the door; finger opening axis is lever-local vertical.
    tool_frame = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation = scene["handle_rotation_world"] @ tool_frame
    orientation = matrix_to_quat(rotation)
    outward = -scene["handle_axis_world"]
    api.phase("pregrasp")
    api.move_ee(position + 0.17 * outward, orientation, duration_s=2.5)
    api.hold(0.4)
    api.phase("reach")
    api.move_ee(position, orientation, duration_s=2.0)
    observation = api.hold(0.3)
    error = np.linalg.norm(observation["ee_pos_m"] - position)
    from active_sysid.robot.controller import quat_to_matrix
    angle_error = np.linalg.norm(rotation_vector(rotation @ quat_to_matrix(observation["ee_quat_wxyz"]).T))
    if error > 0.01 or angle_error > 0.10:
        raise ExecutionError("TCP did not reach the grasp pose")
    api.phase("grasp")
    api.set_gripper(0.006, duration_s=1.0)
    observation = api.hold(0.4)
    if not np.all(observation["finger_contact_normal_n"] > 0.5):
        raise ExecutionError("bilateral finger/handle contact was not established")
    return {"reach_position_error_m": float(error), "reach_orientation_error_rad": float(angle_error)}


def run(api):
    result = reach_and_grasp(api)
    scene = api.observe()["privileged"]
    api.phase("rotate")
    api.move_arc(scene["handle_pivot_m"], scene["handle_axis_world"],
                 0.85 - scene["handle_angle_rad"], duration_s=2.8)
    observation = api.hold(0.3)
    if observation["privileged"]["handle_angle_rad"] < 0.60:
        raise ExecutionError("the lever did not rotate past the planned release region")
    scene = observation["privileged"]
    api.phase("pull")
    api.move_arc(scene["door_pivot_m"], scene["door_axis_world"], 0.45, duration_s=4.2)
    api.hold(0.3)
    api.phase("verify")
    observation = api.hold(0.5)
    result["door_open_verified"] = bool(observation["privileged"]["door_angle_rad"] >= 0.30)
    if not result["door_open_verified"]:
        raise ExecutionError("door did not reach the task goal after pulling")
    return result
