"""Independent physics-trace acceptance for the original World robot program."""
from __future__ import annotations

import hashlib
import inspect
import platform
from pathlib import Path

import mujoco
import numpy as np

from .controller import quat_to_matrix, rotation_vector
from .recording import RobotRecorder
from .reference_world_backend import WorldRobotBackend
from .reference_world_runtime import WorldRobotRuntime


def summarize_world_episode(backend, api, error, *, acceptance='r1', goal_angle_rad=0.30,
                            contact_force_limit_n=80.0, unexpected_contact_limit_n=5.0):
    """All task gates use raw MuJoCo samples, not the policy's claimed result."""
    if acceptance not in ('r0', 'r1', 'task'):
        raise ValueError('acceptance must be r0, r1, or task')
    samples=backend.samples
    if not samples:
        return {'status':'fail','success':False,'execution_error':error or 'no samples','physics_steps':0}
    def field(name):return np.asarray([s[name] for s in samples])
    phase=field('phase');time_s=field('time_s')
    bilateral=np.all(field('finger_contact_normal_n')>0.5,axis=1)
    grasp=np.flatnonzero(phase=='grasp')
    rotate=np.flatnonzero(phase=='rotate')
    pull=np.flatnonzero(phase=='pull')
    manipulation=np.r_[rotate,pull]
    nwindow=max(1,int(np.ceil(0.2/backend.dt)))
    grasp_tail=grasp[-nwindow:]
    reach_actions=[a for a in api.actions if a['phase']=='reach' and a['action']=='move_ee']
    reach_indices=np.flatnonzero(phase=='reach')
    reach_pos_err=reach_rot_err=None
    if reach_actions and len(reach_indices):
        target=reach_actions[-1]['arguments'];actual=samples[reach_indices[-1]]
        reach_pos_err=float(np.linalg.norm(np.asarray(target['position_m'])-actual['ee_pos_m']))
        reach_rot_err=float(np.linalg.norm(rotation_vector(quat_to_matrix(target['quaternion_wxyz'])@
                                                     quat_to_matrix(actual['ee_quat_wxyz']).T)))
    relative_positions=field('handle_in_tcp_m')
    relative_quats=field('handle_in_tcp_quat_wxyz')
    slip_trans=slip_rot=None
    if len(grasp_tail)==nwindow and len(manipulation):
        reference=np.mean(relative_positions[grasp_tail],axis=0)
        slip_trans=float(np.max(np.linalg.norm(relative_positions[manipulation]-reference,axis=1)))
        ref_q=relative_quats[grasp_tail[-1]]
        cosine=np.clip(np.abs(relative_quats[manipulation]@ref_q),0,1)
        slip_rot=float(np.max(2*np.arccos(cosine)))
    force_peak=float(field('total_handle_contact_force_n').max())
    unexpected_peak=float(field('unexpected_contact_force_n').max())
    self_peak=float(field('self_contact_force_n').max())
    joint_torque=field('joint_torque_nm')
    lo=backend.model.actuator_forcerange[backend.arm_actuators,0]
    hi=backend.model.actuator_forcerange[backend.arm_actuators,1]
    # In MuJoCo, joint- and object-space external actuations must be audited separately.
    robot_joint_ids=set(backend.arm_jids.tolist()+backend.finger_jids.tolist())
    actuator_joint_ids=[int(x) for x in backend.model.actuator_trnid[:,0]]
    robot_only_actuators=all(i in robot_joint_ids for i in actuator_joint_ids)
    no_external_object_torque=bool(np.max(np.abs(field('object_applied_torque_nm')))<1e-12 and
                                   np.max(field('object_xfrc_max'))<1e-12)
    target_door=field('door_angle_rad')
    target_handle=field('handle_angle_rad')
    bolt=field('bolt_extension_m')
    rgb_times=np.asarray([entry['time_s'] for entry in api.rgb_observations], dtype=float)
    observed_rgb_feedback=bool(len(rgb_times)>=2 and rgb_times[-1]-rgb_times[0]>backend.dt)
    phases={}
    for name in ('settle','pregrasp','reach','grasp','rotate','pull','verify'):
        mask=phase==name
        n=int(mask.sum())
        if n:
            phases[name]={'steps':n,'duration_s':n*backend.dt,
                          'bilateral_contact_ratio':float(np.mean(bilateral[mask])),
                          'max_contact_force_n':float(np.max(field('total_handle_contact_force_n')[mask]))}
    gates={
        'finite_telemetry':bool(np.all(np.isfinite(np.column_stack((time_s,target_door,target_handle,bolt)))) and
                                np.all(np.isfinite(joint_torque))),
        'monotonic_time':bool(np.all(np.diff(time_s)>0)),
        'execution_completed':error is None,
        'robot_only_actuators':robot_only_actuators,
        'native_handle_bolt_constraint':bool(backend.model.eq_active0[backend.equality_id] and
                                             np.all(field('native_constraint_active'))),
        'no_object_actuation':no_external_object_torque,
        'contact_force_limit':force_peak<=contact_force_limit_n,
        'no_unexpected_contact':unexpected_peak<=unexpected_contact_limit_n and self_peak<=unexpected_contact_limit_n,
        'joint_torque_limits':bool(np.all(joint_torque>=lo-1e-6) and np.all(joint_torque<=hi+1e-6)),
        'robot_motion':bool(np.max(np.ptp(field('q'),axis=0))>0.1),
        'reach':bool(reach_pos_err is not None and reach_pos_err<0.01 and reach_rot_err<0.10),
        'grasp':bool(len(grasp_tail)==nwindow and np.mean(bilateral[grasp_tail])>=0.95),
        'rotate':bool(len(rotate) and np.max(target_handle[rotate])>=0.55 and np.max(bolt[rotate])>=0.020),
        'pull':bool(len(pull) and float(target_door[-1])>=goal_angle_rad),
        'verify':bool(float(target_door[-1])>=goal_angle_rad),
        'bilateral_during_manipulation':bool(len(manipulation) and np.mean(bilateral[manipulation])>=0.95),
        'translation_drift':bool(slip_trans is not None and slip_trans<=0.01),
        'rotation_drift':bool(slip_rot is not None and slip_rot<=0.15),
        'physical_handle_contact':bool(np.max(field('total_handle_contact_force_n'))>0.5),
        'physical_unlock':bool(np.max(target_handle)>=0.55 and np.max(bolt)>=0.020),
        'rgb_feedback_observed':observed_rgb_feedback,
    }
    required=['finite_telemetry','monotonic_time','execution_completed','robot_only_actuators',
              'native_handle_bolt_constraint','no_object_actuation','contact_force_limit',
              'no_unexpected_contact','joint_torque_limits','robot_motion']
    if acceptance=='r0':required+=['reach','grasp']
    elif acceptance=='r1':required+=['reach','grasp','rotate','pull','verify',
                                    'bilateral_during_manipulation','translation_drift','rotation_drift']
    else:
        required+=['physical_handle_contact','physical_unlock','verify']
        if api.observations=='visual':
            required+=['rgb_feedback_observed']
    failures=[name for name in required if not gates[name]]
    return {
        'status':'pass' if not failures else 'fail','success':not failures,
        'acceptance':acceptance,'execution_error':error,'physics_steps':len(samples),
        'elapsed_s':len(samples)*backend.dt,'action_count':len(api.actions),
        'final_door_angle_rad':float(target_door[-1]),
        'final_handle_angle_rad':float(target_handle[-1]),
        'max_handle_angle_rad':float(np.max(target_handle)),
        'max_bolt_extension_m':float(np.max(bolt)),
        'final_bolt_extension_m':float(bolt[-1]),
        'peak_aggregate_handle_contact_force_n':force_peak,
        'peak_unexpected_contact_force_n':unexpected_peak,
        'peak_self_contact_force_n':self_peak,
        'max_joint_torque_nm':np.max(np.abs(joint_torque),axis=0).tolist(),
        'joint_torque_limits_nm':backend.model.actuator_forcerange[backend.arm_actuators].tolist(),
        'reach_position_error_m':reach_pos_err,
        'reach_orientation_error_rad':reach_rot_err,
        'max_handle_translation_drift_m':slip_trans,
        'max_handle_rotation_drift_rad':slip_rot,
        'bilateral_manipulation_contact_ratio':float(np.mean(bilateral[manipulation])) if len(manipulation) else None,
        'rgb_observations':api.rgb_observations,
        'rgb_observation_count':len(api.rgb_observations),
        'per_phase':phases,'gates':gates,'required_gates':required,
        'failed_required_gates':failures,
        'criteria':{'goal_door_angle_rad':goal_angle_rad,'handle_unlock_angle_rad':0.55,
                    'minimum_bolt_retraction_m':0.020,
                    'experiment_handle_contact_force_limit_n':contact_force_limit_n,
                    'experiment_unexpected_contact_force_limit_n':unexpected_contact_limit_n,
                    'grasp_bilateral_window_s':nwindow*backend.dt,
                    'bilateral_manipulation_ratio_min':0.95,
                    'translation_drift_max_m':0.01,'rotation_drift_max_rad':0.15},
        'force_measurement':'sum of contact-force vector magnitudes across the two finger-handle collision pairs; not a damage threshold',
    }


def evaluate_world_policy(policy, *, out_dir, binding=None, mechanism='lever',
                          observations='privileged', acceptance='r1', video=False,
                          goal_angle_rad=0.30, max_steps=30000,
                          world_modifier=None, stage_name=None, seed=None,
                          visual_cameras=('overview', 'robot0_eye_in_hand')):
    backend=WorldRobotBackend(mechanism=mechanism, **({'binding':binding} if binding else {}))
    parameter_draw=world_modifier(backend) if world_modifier is not None else {}
    api=WorldRobotRuntime(backend,observations=observations,max_steps=max_steps,
                          contact_force_limit_n=80.0,unexpected_contact_limit_n=5.0,
                          visual_cameras=visual_cameras)
    error=None;policy_return=None
    with RobotRecorder(out_dir,backend.model,video=video,cameras=('overview','closeup'),
                       width=640,height=480,fps=25) as recorder:
        backend.callbacks.append(recorder.capture)
        try:policy_return=policy(api)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        finally:api.close()
        result=summarize_world_episode(backend,api,error,acceptance=acceptance,
                                       goal_angle_rad=goal_angle_rad)
        result['policy_return']=policy_return
        policy_source=inspect.getsourcefile(policy)
        result['provenance']={
            'scene_role':'reference_world','scene':str(backend.scene),
            'scene_sha256':hashlib.sha256(backend.scene.read_bytes()).hexdigest(),
            'binding':str(backend.binding_path),
            'binding_sha256':hashlib.sha256(backend.binding_path.read_bytes()).hexdigest(),
            'policy':policy_source,
            'policy_sha256':hashlib.sha256(Path(policy_source).read_bytes()).hexdigest() if policy_source else None,
            'mechanism':mechanism,'observations':observations,'mujoco':mujoco.__version__,
            'python':platform.python_version(),'dt_s':backend.dt,
            'robot_source_actuator_count':backend.model.nu,
            'native_source_equality_count':backend.model.neq,
            'video_fps':recorder.fps if video else None,'video_frames':recorder.video_frame_count,
            'stage_name':stage_name, 'seed':seed,
            'visual_cameras':list(visual_cameras),
            'parameter_draw_evaluator_private':parameter_draw,
        }
        recorder.finish(result,api.actions)
    return result
