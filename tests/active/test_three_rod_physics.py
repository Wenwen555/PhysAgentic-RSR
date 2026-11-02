"""Compile and step a real shared MuJoCo scene; no LLM, IK or hardware needed."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from physrsr.active.core.contracts import Hypothesis, IDENTIFICATION_API_VERSION
from physrsr.active.core.policy import Policy
from physrsr.active.core.measurements import SCENE_FEATURE_SCHEMA
from physrsr.active.core.paths import PACKAGE_ROOT
from physrsr.active.runtime.predictive import PredictiveSimulator
from physrsr.active.tasks.rod_scene import (
    OBJECT_IDS, SCENE_SUCCESS, SCENE_MECHANISM, SequentialLiftScore, ThreeRodAdapter)
from physrsr.active.tasks.rod_scene_world import ThreeRodWorld


def config():
    return json.loads((PACKAGE_ROOT/'examples/three_rod_world.private.example.json').read_text())


def model_for(value):
    return Hypothesis('scene',{f'{key}__{name}':v for key,rod in value['objects'].items()
        for name,v in rod['physics'].items()},1.,SCENE_MECHANISM)


@pytest.fixture(scope='module')
def world():
    value=config()
    # Deliberately distinct values test that parameters do not all bind to rod_a.
    value['objects']['rod_b']['physics']['mass_kg']=.21
    value['objects']['rod_c']['physics']['friction']=.7
    instance=ThreeRodWorld(value)
    yield instance
    instance.close()


def test_all_three_free_bodies_compile_in_one_scene_with_independent_physics(world):
    compiled=world.call('world.inspect_compilation',{})
    assert compiled['world_count']==compiled['robot_count']==1
    assert set(compiled['objects'])==set(OBJECT_IDS)
    assert len({o['body_id'] for o in compiled['objects'].values()})==3
    assert len({o['free_joint'] for o in compiled['objects'].values()})==3
    for key,row in compiled['objects'].items():
        expected=world.backend.config['objects'][key]['physics']
        assert row['mass_kg']==pytest.approx(expected['mass_kg'])
        assert row['com_body_m'][0]==pytest.approx(expected['com_offset'])
        assert row['friction'][0]==pytest.approx(expected['friction'])
    np.testing.assert_allclose(compiled['gripper_actuator_force_ranges_n'],[[-2.,2.],[-2.,2.]])


def test_scene_reset_is_repeatable_and_all_observations_share_robot_time(world):
    first=world.call('target.prepare',{})
    again=world.call('target.prepare',{})
    assert len(first['objects'])==3
    for key in OBJECT_IDS:
        np.testing.assert_allclose(first['objects'][key]['position'],again['objects'][key]['position'],atol=1e-9)
        assert abs(first['objects'][key]['support_clearance_m'])<.003
        assert not {'physics','mass_kg','com_offset','friction'} & set(first['objects'][key])
    assert 'wrench' not in first['objects']['rod_a'] and len(first['wrench']['force_n'])==3
    env,_=world.backend.ensure()
    assert first['time_s']==pytest.approx(env.robosuite_env.sim.data.time)
    assert world.call('target.describe',{})['workspace']['min_position_m'][0]<first['objects']['rod_a']['position'][0]


def test_rods_generate_cross_object_contacts_in_shared_dynamics(world):
    world.call('target.prepare',{})
    env,_=world.backend.ensure();suite=env.robosuite_env;sim=suite.sim
    a=sim.data.get_joint_qpos(suite.rods['rod_a'].joints[0]).copy()
    b=a.copy();b[1]+=.025  # overlap their 3 cm width by 5 mm
    sim.data.set_joint_qpos(suite.rods['rod_b'].joints[0],b)
    sim.forward()
    bodies=set(suite.rod_body_ids[key] for key in ('rod_a','rod_b'))
    contacts=[c for c in sim.data.contact[:sim.data.ncon] if
        {sim.model.geom_bodyid[c.geom1],sim.model.geom_bodyid[c.geom2]}==bodies]
    assert contacts, 'The rods must collide in the same MuJoCo simulation'
    before=sim.data.get_joint_qpos(suite.rods['rod_b'].joints[0]).copy()
    for _ in range(10): sim.step()
    after=sim.data.get_joint_qpos(suite.rods['rod_b'].joints[0]).copy()
    assert np.linalg.norm(after[:3]-before[:3])>1e-6


def test_claimed_success_without_physical_lifts_does_not_pass_scene_task(world):
    policy=Policy("def run(robot):\n    robot.hold(0.4)\n    return {'status':'success','completed':['rod_a','rod_b','rod_c']}\n",
        api_version=IDENTIFICATION_API_VERSION)
    result=world.call('world.evaluate_policy',dict(policy=policy.wire()))
    assert result['status']=='completed' and result['record']['status']=='fail'
    assert result['task_result']['success'] is False
    assert not any(row['held'] for row in result['task_result']['objects'].values())
    assert result['feature_schema']==SCENE_FEATURE_SCHEMA and len(result['features'])==48


def test_predictive_scene_replays_public_geometry_in_one_48_dimensional_response(world):
    observed=world.call('target.prepare',{})
    scene={k:observed[k] for k in ('time_s','coordinate_frame','objects','tcp')}
    value=config();model=model_for(value)
    sim=PredictiveSimulator([ThreeRodAdapter()])
    task=dict(id='three_rod_lift_return',success=SCENE_SUCCESS,constraints=dict(grip_force_per_finger_n=2.),scene=scene)
    sim.call('sim.configure',dict(session_id='one',object_ids=list(OBJECT_IDS),task=task,models=[asdict(model)]))
    policy=Policy('def run(robot):\n    robot.hold(0.2)\n    return {}\n',api_version=IDENTIFICATION_API_VERSION)
    result=sim.call('sim.predict_probe',dict(session_id='one',model_version=0,seeds=[1],probe=dict(policy=policy.wire())))
    assert len(sim.sessions)==1 and len(result['predictions']['scene']['features'])==48
    assert result['feature_schema']==SCENE_FEATURE_SCHEMA
    evaluation=sim.call('sim.evaluate',dict(session_id='one',model_version=0,seeds=[1],policies=[policy.wire()]))
    row=evaluation['rows'][0]
    assert row['success_rate']==0. and row['margin']<0.
    assert set(row['feedback'][0]['task_result']['objects'])==set(OBJECT_IDS)


def state(**changes):
    value=dict(lifted=False,grasped=False,level=True,at_rest=True,lift_margin=.02)
    value.update(changes)
    return value


def test_temporal_score_requires_every_lift_hold_and_return_and_ignores_repeat_reads():
    score=SequentialLiftScore(0.)
    baseline={key:state() for key in OBJECT_IDS}
    now=0.
    for index,key in enumerate(OBJECT_IDS):
        lifted=deepcopy(baseline);lifted[key].update(lifted=True,grasped=True,at_rest=False)
        for _ in range(10):
            now+=.05;score.update(now,lifted)
            score.update(now,lifted)  # repeated reward/observe at the same tick
        assert score.progress[key]['held']
        assert not score.report()['success']
        for _ in range(5): now+=.05;score.update(now,baseline)
        assert score.report()['success'] is (index==2)
    disturbed=deepcopy(baseline);disturbed['rod_a']['at_rest']=False
    score.update(now+.05,disturbed)
    assert not score.report()['success']


def test_partial_hold_and_ungrasped_lift_cannot_be_accumulated():
    score=SequentialLiftScore(0.)
    states={key:state() for key in OBJECT_IDS}
    states['rod_a'].update(lifted=True,grasped=True,at_rest=False)
    for tick in range(1,5): score.update(tick*.05,states)
    states['rod_a']['grasped']=False
    score.update(.25,states)
    states['rod_a']['grasped']=True
    for tick in range(6,10): score.update(tick*.05,states)
    assert not score.progress['rod_a']['held']
    assert not score.report()['success']
