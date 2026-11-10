"""Explicit fixed-program manipulation baseline for the original double-door World.

Robot action sequence: joint-space transit, Cartesian reach, pinch, handle arc,
door arc, visual/proprioceptive verification. No door/handle joint commands.
The frozen pregrasp joints apply ONLY to the exact `unidoor_reference` layout.
"""
import numpy as np
from .controller import ExecutionError, matrix_to_quat, quat_to_matrix, rotation_vector

# Pregrasp pose IK solved once from the original World robot's `home` keyframe.
PREGRASP_Q = [-1.61633587, 0.60866496, -0.74382632,
              -2.13615844, -2.08295941, 2.19003761, -1.00430819]
# Tool local +x is vertical finger opening, +z points into the door (+world X).
GRASP_ROTATION = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
GRASP_QUAT = matrix_to_quat(GRASP_ROTATION)


def reach_and_grasp(api):
    api.phase('settle')
    api.hold(0.6)
    state = api.observe()['privileged']
    # TCP is 8 mm beyond the gripper pad centers along its +z direction.
    grasp_tcp = state['handle_grasp_pos_m'] + np.array([0.009, 0.0, 0.0])
    api.phase('pregrasp')
    api.move_joints(PREGRASP_Q, duration_s=6.0)
    api.phase('reach')
    api.move_ee(grasp_tcp, GRASP_QUAT, duration_s=3.0)
    obs = api.observe()
    pos_err = float(np.linalg.norm(obs['ee_pos_m'] - grasp_tcp))
    rot_err = float(np.linalg.norm(rotation_vector(GRASP_ROTATION @
                     quat_to_matrix(obs['ee_quat_wxyz']).T)))
    if pos_err > 0.01 or rot_err > 0.10:
        raise ExecutionError('robot failed to reach the original World lever handle')
    api.phase('grasp')
    api.set_gripper(0.010, duration_s=1.5)
    obs = api.hold(0.5)
    if not np.all(obs['finger_contact_normal_n'] > 0.5):
        raise ExecutionError('real dual-finger/handle contact not established')
    return {'reach_position_error_m': pos_err,
            'reach_orientation_error_rad': rot_err}


def run(api):
    result = reach_and_grasp(api)
    state = api.observe()['privileged']
    api.phase('rotate')
    api.move_arc(state['handle_pivot_m'], state['handle_axis_world'],
                 0.78 - state['handle_angle_rad'], duration_s=4.0)
    state = api.observe()['privileged']
    if state['handle_angle_rad'] < 0.55 or state['bolt_extension_m'] < 0.020:
        raise ExecutionError('handle rotation did not retract the native door bolt')
    api.phase('pull')
    api.move_arc(state['door_pivot_m'], state['door_axis_world'],
                 0.40, duration_s=5.0)
    api.phase('verify')
    result['final_door_angle_rad'] = api.hold(0.5)['privileged']['door_angle_rad']
    if result['final_door_angle_rad'] < 0.30:
        raise ExecutionError('robot did not pull the native door open')
    return result


def run_without_rotation(api):
    """Ablation: same reach/grasp/pull, explicitly omit turning the handle."""
    result = reach_and_grasp(api)
    state = api.observe()['privileged']
    api.phase('pull')
    api.move_arc(state['door_pivot_m'], state['door_axis_world'],
                 0.40, duration_s=5.0)
    api.phase('verify')
    result['final_door_angle_rad'] = api.hold(0.5)['privileged']['door_angle_rad']
    return result
