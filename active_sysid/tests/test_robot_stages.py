"""Core Sim/World boundary regressions; keep native R1 tests separate."""
import json
from pathlib import Path
import numpy as np
import pytest
from active_sysid.robot.stage_freeze import verify_manifest,verify_world_assets,sha256
from active_sysid.robot.reconstructed_sim_backend import SimRobotBackend
from active_sysid.robot.reference_world_backend import WorldRobotBackend
from active_sysid.robot.boundary_runtime import BoundaryRuntime
from active_sysid.robot.stage_randomization import apply_sim_randomization
from active_sysid.robot.stages import STAGES,load_stage,run_stage,run_all
from active_sysid.robot.sim2sim import probe_world,freeze_policy,evaluate_frozen


def make_policy(tmp_path,body=None):
    p=tmp_path/'policy.py'
    p.write_text(body or 'def run(api):\n    o=api.observe()\n    assert "privileged" not in o\n    api.hold(0.02)\n')
    return p


def test_stage_roles_and_freeze():
    m=verify_manifest()
    assert m['baseline_id']=='unidoor_sim2sim_boundary_v2'
    assert len(m['world_files_sha256'])==111
    for name in STAGES:
        cfg=load_stage(name)
        assert cfg['role']=='development_sim'
        assert cfg['sim_variant']=='robot_m2'
    assert load_stage('privileged_exact')['randomization']=={}
    assert load_stage('visual_exact')['randomization']=={}
    assert load_stage('visual_randomized')['randomization']
    verify_world_assets()


def test_real_reconstructed_assets_and_three_mechanisms():
    models=[SimRobotBackend(v) for v in ('initial','m1','m2')]
    assert all(b.model.njnt==13 and b.model.nu==9 for b in models)
    assert not models[0].latch_enabled
    assert models[1].latch_enabled and models[2].latch_enabled
    assert models[2].unlock_threshold==pytest.approx(.5617619734137613)
    assert models[0].privileged()['source_role']=='development_sim'
    assert models[0].model.joint('LeverDoor_Leaf_joint').id>=0
    assert all('bolt' not in models[0].model.joint(i).name.lower() for i in range(models[0].model.njnt))
    assert 'assets/reconstructed_sim/' in str(models[0].scene)
    assert models[0].scene!=WorldRobotBackend().scene


def test_shared_robot_joint_controls():
    sim,world=SimRobotBackend(),WorldRobotBackend()
    assert np.allclose(sim.model.actuator_gainprm[sim.arm_actuators],world.model.actuator_gainprm[world.arm_actuators])
    assert np.allclose(sim.model.actuator_biasprm[sim.arm_actuators],world.model.actuator_biasprm[world.arm_actuators])
    assert np.allclose(sim.model.jnt_range[sim.arm_jids],world.model.jnt_range[world.arm_jids])
    assert np.allclose(sim.model.dof_armature[sim.arm_dofs],world.model.dof_armature[world.arm_dofs])
    assert np.allclose(sim.tcp_pose()[0],world.tcp_pose()[0])


def test_sim_runner_never_constructs_world(tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Development constructed World')
    monkeypatch.setattr(WorldRobotBackend,'__init__',forbidden)
    p=make_policy(tmp_path,'def run(api):\n    s=api.observe()["privileged"]\n    assert s["source_role"]=="development_sim"\n    api.hold(0.02)\n')
    r=run_stage('privileged_exact',p,tmp_path/'sim',sim_variant='m2')
    assert r['role']=='development_sim' and r['world_evaluations_performed']==0
    ep=json.loads(Path(r['episodes'][0]['metrics_path']).read_text())
    assert ep['execution_error'] is None
    assert ep['provenance']['role']=='development_sim'


def test_world_public_observations_and_privileged_denial():
    b=WorldRobotBackend()
    with pytest.raises(ValueError,match='World truth'):
        BoundaryRuntime(b,role='world_probe',observations='privileged')
    for role in ('world_probe','world_evaluation'):
        api=BoundaryRuntime(b,role=role,observations='visual')
        o=api.observe()
        forbidden={'privileged','door_angle_rad','handle_angle_rad','bolt_extension_m','latch_active',
                   'handle_contact_force_world_n','total_handle_contact_force_n','unexpected_contact_force_n'}
        assert not forbidden.intersection(o)
        assert len(o['q'])==7 and len(o['wrist_wrench_ee'])==6
        with pytest.raises(ValueError,match='Unknown'):api.dispatch('privileged',[],{})
        api.close()


def test_seeded_perturbations_apply_only_to_sim():
    a,b,c=SimRobotBackend('m2'),SimRobotBackend('m2'),SimRobotBackend('m2')
    ranges=load_stage('visual_randomized')['randomization']
    before=sha256(a.scene);gains=a.model.actuator_gainprm.copy()
    da=apply_sim_randomization(a,7,ranges)
    assert da==apply_sim_randomization(b,7,ranges)
    assert da!=apply_sim_randomization(c,3,ranges)
    assert np.array_equal(a.model.actuator_gainprm,gains)
    assert sha256(a.scene)==before
    with pytest.raises(ValueError,match='development_sim'):
        apply_sim_randomization(WorldRobotBackend(),7,ranges)


def test_policy_process_cannot_read_world_or_backend(tmp_path):
    p=make_policy(tmp_path,'''def run(api):
    import os
    assert not os.path.exists('/home/wen/experiment/PhyAgent-RSR/active_sysid/assets/reference_world/unidoor_native/scene.xml')
    assert not hasattr(api, '_backend')
    try:
        from active_sysid.robot.reference_world_backend import WorldRobotBackend
        raise AssertionError('Project code was mounted into policy process')
    except ImportError:
        pass
    api.hold(0.02)
    return {'world_hidden': True}
''')
    r=run_stage('visual_exact',p,tmp_path/'sandbox')
    ep=json.loads(Path(r['episodes'][0]['metrics_path']).read_text())
    assert ep['execution_error'] is None
    assert ep['policy_return']['world_hidden']
    assert ep['provenance']['policy_execution']=='bubblewrap_sensor_rpc'


def test_probe_exports_only_public_sensor_evidence(tmp_path):
    r=probe_world(make_policy(tmp_path),tmp_path/'probe',max_steps=50)
    assert r['execution_completed']
    data=np.load(tmp_path/'probe/observations.npz')
    assert set(data.files)==set(r['sensor_keys'])
    assert not {'door_angle_rad','bolt_extension_m','total_handle_contact_force_n','latch_active'}.intersection(data.files)
    assert not (tmp_path/'probe/metrics.json').exists()
    assert not (tmp_path/'probe/trajectory.npz').exists()
    assert (tmp_path/'probe/actions.json').exists()


def test_freeze_transfer_and_tamper_detection(tmp_path):
    p=make_policy(tmp_path)
    run_stage('visual_exact',p,tmp_path/'sim')
    meta=freeze_policy(p,tmp_path/'sim/summary.json',tmp_path/'frozen')
    assert meta['negative_control']  # only hold(), not an opening CaP
    p.write_text('raise RuntimeError("live policy changed after freeze")\n')
    r=evaluate_frozen(tmp_path/'frozen',tmp_path/'world')
    assert r['role']=='world_evaluation' and r['policy_sha256']==meta['policy_sha256']
    assert r['execution_error'] is None and not r['success']
    assert 'final_door_angle_rad' not in r
    private=json.loads((tmp_path/'world/evaluator_private/metrics.json').read_text())
    assert private['provenance']['observations']=='visual'
    assert private['provenance']['parameter_draw_evaluator_private']=={}
    (tmp_path/'frozen/policy.py').write_text('def run(api): pass\n')
    with pytest.raises(ValueError,match='modified'):
        evaluate_frozen(tmp_path/'frozen',tmp_path/'world_again')


def test_privileged_sim_policy_cannot_be_transferred(tmp_path):
    p=make_policy(tmp_path,'def run(api):\n    assert "privileged" in api.observe()\n    api.hold(0.02)\n')
    run_stage('privileged_exact',p,tmp_path/'sim')
    with pytest.raises(ValueError,match='Privileged-Sim'):
        freeze_policy(p,tmp_path/'sim/summary.json',tmp_path/'frozen')


def test_run_all_is_sim_only_and_keeps_visual_sha(tmp_path):
    p=make_policy(tmp_path,'def run(api):\n    api.observe()\n    api.hold(0.02)\n')
    r=run_all(p,p,tmp_path/'all',randomized_seeds=[3,7],sim_variant='m1')
    assert r['role']=='development_sim' and r['world_evaluations_performed']==0
    assert r['visual_policy_unchanged_in_randomized']
    assert r['stages']['visual_randomized']['total_episodes']==2
    assert all(s['sim_variant']=='m1' for s in r['stages'].values())


def test_modified_freeze_fails(monkeypatch):
    from active_sysid.robot import stage_freeze as module
    real=module.sha256;target=module.ROOT/'src/active_sysid/robot/controller.py'
    monkeypatch.setattr(module,'sha256',lambda p:'0'*64 if p==target else real(p))
    with pytest.raises(RuntimeError,match='controller.py'):module.verify_manifest()


def test_legacy_direct_world_evaluation_cli_is_blocked(tmp_path):
    import subprocess,sys
    p=make_policy(tmp_path)
    proc=subprocess.run([sys.executable,'-m','active_sysid.robot','world-evaluate',
        '--policy',str(p),'--observations','privileged'],capture_output=True,text=True)
    assert proc.returncode!=0
    assert 'Direct/privileged World policy development is disabled' in proc.stderr
