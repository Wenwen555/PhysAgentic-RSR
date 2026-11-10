"""R1 checks on the *source* double-door World, never the Panda fixture."""
from pathlib import Path
import mujoco as mj
import numpy as np
import pytest

from active_sysid.robot.reference_world_backend import WorldRobotBackend
from active_sysid.robot.reference_world_evaluator import evaluate_world_policy
from active_sysid.robot.reference_oracle_policy import run, run_without_rotation
from active_sysid.robot.reference_world_runtime import WorldRobotRuntime
from active_sysid.robot.controller import ExecutionError


def test_world_binding_and_native_contact():
    b=WorldRobotBackend()
    assert (b.model.nbody,b.model.njnt,b.model.nu,b.model.neq)==(29,15,9,2)
    assert b.model.ncam==11
    assert b.model.eq_active0[b.equality_id]
    assert any((b.model.geom_contype[b.handle_geom] & b.model.geom_conaffinity[g]) and
               (b.model.geom_conaffinity[b.handle_geom] & b.model.geom_contype[g])
               for g in b.finger_geom_sets[0])
    robot_joint_ids=set(b.arm_jids.tolist()+b.finger_jids.tolist())
    assert all(j in robot_joint_ids for j in b.model.actuator_trnid[:,0])
    assert b.model.joint('round_door_hinge').id != b.door_jid
    assert np.allclose(b.data.qpos[[b.door_qpos,b.handle_qpos,b.bolt_qpos]],0)
    assert len(b.finger_geom_sets[0])>0 and len(b.finger_geom_sets[1])>0
    api=WorldRobotRuntime(b,observations='robot')
    assert 'privileged' not in api.observe()
    api.close()


def test_original_world_fixed_robot_r1(tmp_path):
    before=WorldRobotBackend().scene.read_bytes()
    r=evaluate_world_policy(run,out_dir=tmp_path/'r1',video=False)
    assert r['success'],r['failed_required_gates']
    assert r['final_door_angle_rad']>=0.30
    assert r['max_handle_angle_rad']>=0.55
    assert r['max_bolt_extension_m']>=0.020
    assert r['bilateral_manipulation_contact_ratio']>=0.95
    assert r['max_handle_translation_drift_m']<0.010
    assert r['gates']['no_object_actuation']
    assert r['provenance']['scene_role']=='reference_world'
    assert (tmp_path/'r1'/'trajectory.npz').exists()
    assert WorldRobotBackend().scene.read_bytes()==before


def test_native_bolt_lock_requires_handle_rotation(tmp_path):
    r=evaluate_world_policy(run_without_rotation,out_dir=tmp_path/'no_rotate',video=False)
    assert not r['success']
    assert r['execution_error'] is None
    assert r['final_door_angle_rad']<0.05
    assert r['max_handle_angle_rad']<0.55
    assert not r['gates']['rotate']
    assert not r['gates']['pull']


def test_contact_disabled_in_memory_blocks_grasp():
    b=WorldRobotBackend()
    b.model.geom_contype[b.handle_geom]=0
    b.model.geom_conaffinity[b.handle_geom]=0
    mj.mj_forward(b.model,b.data)
    api=WorldRobotRuntime(b,observations='privileged',contact_force_limit_n=80,
                          unexpected_contact_limit_n=5)
    # The same program cannot pass the grasp check if finger-handle collision is removed.
    with pytest.raises(ExecutionError,match='dual-finger/handle contact'):
        from active_sysid.robot.reference_oracle_policy import reach_and_grasp
        reach_and_grasp(api)
    assert b.sample()['door_angle_rad']<0.05
    assert np.max(np.abs(b.data.qfrc_applied[b.object_dofs]))==0
    api.close()
