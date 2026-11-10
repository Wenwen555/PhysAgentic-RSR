"""Three separate operations: Sim development, World probing, frozen transfer.

Only Sim accepts privileged observations or randomization. Policy runs in a
filesystem/network-isolated process with a sensor/action-only RPC contract.
"""
from pathlib import Path
import hashlib
import json
import platform
import shutil
import numpy as np
import mujoco
from .fixture_backend import ROOT
from .reconstructed_sim_backend import SimRobotBackend
from .sensors import robot_observation
from .boundary_runtime import BoundaryRuntime
from .policy_rpc import run_policy
from .recording import RobotRecorder, _json_value


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=_json_value)+'\n')


def _score(b,api,error,goal=.30):
    ss=b.samples
    if not ss:return {'success':False,'gates':{'samples_recorded':False},'physics_steps':0,
                     'elapsed_s':0.0,'execution_error':error,'failed_required_gates':['samples_recorded']}
    array=lambda key:np.asarray([s[key] for s in ss])
    peak=float(array('total_handle_contact_force_n').max())
    unexpected=float(array('unexpected_contact_force_n').max())
    self_peak=float(array('self_contact_force_n').max())
    torque=array('joint_torque_nm');limits=b.model.actuator_forcerange[b.arm_actuators]
    rgb=[entry['time_s'] for entry in api.rgb_observations]
    # Model-internal success is a PRIVATE evaluator fact; never sent over RPC.
    # Task acceptance is physical opening, not recovery of a hidden lock state.
    robot_ids=set(b.arm_jids.tolist()+b.finger_jids.tolist())
    object_actuation=max(float(np.abs(array('object_applied_torque_nm')).max()),
                         float(np.abs(array('actual_object_actuator_torque_nm')).max()),
                         float(array('object_xfrc_max').max()))
    gates={'samples_recorded':True,'execution_completed':error is None,
           'finite_telemetry':bool(all(np.isfinite(array(k)).all() for k in
               ('q','dq','joint_torque_nm','ee_pos_m','door_angle_rad','total_handle_contact_force_n'))),
           'robot_only_actuators':all(int(j) in robot_ids for j in b.model.actuator_trnid[:,0]),
           'no_object_actuation':object_actuation<=1e-12,
           'contact_force_limit':peak<=80.0,'no_unexpected_collision':max(unexpected,self_peak)<=5.0,
           'joint_torque_limits':bool(np.all(torque>=limits[:,0]-1e-6) and np.all(torque<=limits[:,1]+1e-6)),
           'robot_motion':bool(np.max(np.ptp(array('q'),axis=0))>.02),
           'physical_handle_contact':peak>.5,
           'door_open':float(ss[-1]['door_angle_rad'])>=goal}
    if api.observations=='visual':gates['rgb_feedback_observed']=len(rgb)>=2 and max(rgb)-min(rgb)>b.dt
    failed=[k for k,v in gates.items() if not v]
    return {'success':not failed,'status':'pass' if not failed else 'fail','gates':gates,
            'failed_required_gates':failed,'physics_steps':len(ss),'elapsed_s':len(ss)*b.dt,
            'execution_error':error,'final_door_angle_rad':float(ss[-1]['door_angle_rad']),
            'peak_handle_contact_force_n':peak,'peak_unexpected_contact_force_n':unexpected,
            'peak_self_contact_force_n':self_peak,'max_object_actuation':object_actuation,
            'criteria':{'goal_angle_rad':goal,'aggregate_contact_limit_n':80.0,
                        'unexpected_contact_limit_n':5.0,'force_limits_are_damage_certified':False},
            'rgb_observations':api.rgb_observations}


def _episode(b,policy,out,*,role,observations,max_steps=30000,video=False,stage=None,seed=None,draw=None,
             goal_angle_rad=.30,cameras=('overview','robot0_eye_in_hand')):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    # Snapshot before import/execution to bind each report to exact executed bytes.
    source=Path(policy).read_bytes();policy_sha=hashlib.sha256(source).hexdigest()
    snapshot=out/'executed_policy.py';snapshot.write_bytes(source)
    private=out/'evaluator_private' if role=='world_evaluation' else out
    private.mkdir(exist_ok=True)
    api=BoundaryRuntime(b,role=role,observations=observations,max_steps=max_steps,
                        image_dir=out/'rgb',cameras=cameras)
    public_samples=[]
    def collect_public(backend,sample):public_samples.append(robot_observation(backend))
    def actual_actuation(backend,sample):
        sample['actual_object_actuator_torque_nm']=backend.data.qfrc_actuator[backend.object_dofs].copy()
    b.callbacks.append(actual_actuation)
    if role=='world_probe':b.callbacks.append(collect_public)
    error=None;result=None
    recorder=None
    try:
        if role!='world_probe':
            recorder=RobotRecorder(private,b.model,video=video,cameras=('overview','robot0_eye_in_hand'),
                                   width=640,height=480)
            b.callbacks.append(recorder.capture)
        try:result=run_policy(snapshot,api.dispatch,out/'worker.log')
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        finally:api.close()
        provenance={'role':role,'scene':str(b.scene),'scene_sha256':digest(b.scene),
            'policy_sha256':policy_sha,'stage_name':stage,'seed':seed,
            'observations':observations,'sim_variant':getattr(b,'variant',None),'sim_parameters':getattr(b,'mechanism_parameters',None),
            'parameter_draw_evaluator_private':draw or {},'mujoco':mujoco.__version__,
            'python':platform.python_version(),'policy_execution':'bubblewrap_sensor_rpc'}
        if role=='world_probe':
            # Do NOT export b.samples, privileged state, object-labelled contacts,
            # bolt/hinge angles, randomized parameters, or evaluator scores.
            keys=tuple(robot_observation(b))
            arrays={key:np.stack([s[key] for s in public_samples]) for key in keys} if public_samples else {}
            np.savez_compressed(out/'observations.npz',**arrays)
            public={'role':role,'policy_sha256':policy_sha,'execution_completed':error is None,
                    'execution_error':error,'physics_steps':len(b.samples),'elapsed_s':len(b.samples)*b.dt,
                    'sensor_keys':list(keys),'observation_contract':'robot_sensors_rgb_no_object_identity',
                    'rgb_observations':api.rgb_observations}
            write_json(out/'probe.json',public);write_json(out/'actions.json',api.actions)
            return public
        metrics=_score(b,api,error,goal_angle_rad);metrics['provenance']=provenance;metrics['policy_return']=result
        recorder.finish(metrics,api.actions)
        if role=='world_evaluation':
            public={'role':role,'success':metrics['success'],'elapsed_s':metrics['elapsed_s'],
                    'execution_error':error,'policy_sha256':policy_sha,
                    'policy_revision_performed':False,'privileged_observations':False}
            write_json(out/'world_summary.json',public)
            return public
        return metrics
    finally:
        api.close()
        if recorder is not None:recorder.close()


def run_sim_episode(policy,out,*,variant='initial',observations='visual',max_steps=30000,
                    stage=None,seed=0,ranges=None,video=False,goal_angle_rad=.30,
                    cameras=('overview','robot0_eye_in_hand')):
    b=SimRobotBackend(variant)
    from .stage_randomization import apply_sim_randomization
    draw=apply_sim_randomization(b,seed,ranges or {})
    return _episode(b,policy,out,role='development_sim',observations=observations,
                    max_steps=max_steps,stage=stage,seed=seed,draw=draw,video=video,
                    goal_angle_rad=goal_angle_rad,cameras=cameras)


def probe_world(policy,out,*,max_steps=8000):
    from .stage_freeze import verify_world_assets
    from .reference_world_backend import WorldRobotBackend
    verify_world_assets()
    if not 0<max_steps<=8000:raise ValueError('World probe budget is 1..8000 physics steps')
    return _episode(WorldRobotBackend(),policy,out,role='world_probe',observations='visual',
                    max_steps=max_steps)


def freeze_policy(policy,sim_summary,out):
    """Snapshot a self-contained policy after a recorded Sim visual evaluation."""
    from .stage_freeze import verify_manifest, MANIFEST
    verify_manifest()
    report=json.loads(Path(sim_summary).read_text())
    if report.get('role')!='development_sim':raise ValueError('Freeze requires a Sim-only development report')
    if report.get('frozen_manifest_sha256')!=digest(MANIFEST):raise ValueError('Sim report belongs to a different execution release')
    if report.get('observations')!='visual':raise ValueError('Privileged-Sim policies cannot receive World truth; develop a visual policy first')
    policy_sha=digest(policy)
    if report['policy_sha256']!=policy_sha:raise ValueError('Policy changed after Sim evaluation')
    for ep in report['episodes']:
        metrics=json.loads(Path(ep['metrics_path']).read_text())
        if metrics['provenance']['role']!='development_sim' or metrics['provenance']['policy_sha256']!=policy_sha:
            raise ValueError('Episode does not match the Sim policy being frozen')
        if digest(ep['metrics_path'])!=ep['metrics_sha256']:raise ValueError('Sim evidence changed')
    if not report['episodes']:raise ValueError('At least one Sim evaluation is required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(policy,out/'policy.py');shutil.copy2(sim_summary,out/'sim_summary.json')
    meta={'role':'frozen_sim_policy','policy_sha256':policy_sha,'source_stage':report['stage'],
          'observations':'visual','sim_variant':report['sim_variant'],
          'sim_summary_sha256':digest(out/'sim_summary.json'),'baseline_sha256':digest(MANIFEST),
          'source_sim_success_rate':report['success_rate'],
          'negative_control':report['success_rate']==0,'entrypoint':'run(api)'}
    write_json(out/'manifest.json',meta);return meta


def evaluate_frozen(bundle,out,*,video=False):
    from .stage_freeze import verify_world_assets,MANIFEST
    from .reference_world_backend import WorldRobotBackend
    verify_world_assets()
    bundle=Path(bundle);meta=json.loads((bundle/'manifest.json').read_text())
    if meta.get('role')!='frozen_sim_policy' or meta.get('observations')!='visual':raise ValueError('Only frozen visual Sim policies may enter World evaluation')
    if digest(bundle/'policy.py')!=meta['policy_sha256'] or digest(bundle/'sim_summary.json')!=meta['sim_summary_sha256']:
        raise ValueError('Frozen policy bundle modified')
    if digest(MANIFEST)!=meta['baseline_sha256']:raise ValueError('Execution baseline changed since policy freeze')
    result=_episode(WorldRobotBackend(),bundle/'policy.py',out,role='world_evaluation',
                    observations='visual',video=video)
    result['frozen_manifest_sha256']=digest(bundle/'manifest.json')
    write_json(Path(out)/'world_summary.json',result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('probe-world');p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-steps',type=int,default=8000)
    p=sub.add_parser('freeze');p.add_argument('--policy',type=Path,required=True);p.add_argument('--sim-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('evaluate-world');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true')
    args=parser.parse_args()
    if args.command=='probe-world':result=probe_world(args.policy,args.out,max_steps=args.max_steps)
    elif args.command=='freeze':result=freeze_policy(args.policy,args.sim_summary,args.out)
    else:result=evaluate_frozen(args.bundle,args.out,video=args.video)
    print(json.dumps(result,indent=2,default=_json_value))

if __name__=='__main__':main()
