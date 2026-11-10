"""Core completeness checks for task-driven robot grounding."""
from pathlib import Path
import json
import numpy as np
from active_sysid.robot.fixture_backend import ROOT
from active_sysid.robot.reconstructed_sim_backend import SimRobotBackend
from active_sysid.robot.sim2sim import run_sim_episode


def test_robot_grounded_model_versions():
    models=[SimRobotBackend(v) for v in ('robot_m0','robot_m1','robot_m2')]
    assert [b.latch_enabled for b in models]==[False,True,True]
    assert models[1].unlock_threshold==.125
    assert models[2].mechanism_parameters['door_radius_m']==.70
    assert all(b.model.nu==9 for b in models)
    assert all('bolt' not in b.model.joint(i).name for b in models for i in range(b.model.njnt))
    assert all(np.array_equal(b.model.actuator_gainprm,models[0].model.actuator_gainprm) for b in models)
    provenance=json.loads((ROOT/'assets/reconstructed_sim/task_grounded_v1/grounding_provenance.json').read_text())
    assert provenance['old_object_space_M1_M2_parameters_used'] is False
    assert provenance['source_world_XML_read_by_builder'] is False


def test_grounded_reference_completes_task_in_sim(tmp_path):
    source=(ROOT/'policies/robot_space/reference_grounded_turn_pull.py').read_text()
    # Keep physical regression headless; real RGB is checked by the stage CLI.
    source=source.replace("rgb=True,camera='overview'",'').replace("rgb=True,camera='robot0_eye_in_hand'",'')
    policy=tmp_path/'reference.py';policy.write_text(source)
    result=run_sim_episode(policy,tmp_path/'sim',variant='robot_m2',observations='robot')
    assert result['success'],result.get('failed_required_gates')
    assert result['final_door_angle_rad']>=.30
    assert result['max_object_actuation']==0
    assert 'mechanism_released' not in result['gates']
    assert result['provenance']['sim_parameters']['model_type']=='task_condition_contact_surrogate'
