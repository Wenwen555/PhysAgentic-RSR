"""Single-World diagnostic calls reuse live contracts without another loop."""
from copy import deepcopy
import json

import pytest

from physrsr.active.core.artifacts import save_json
from physrsr.active.core.belief import merge_identification
from physrsr.active.core.programs import source_policy
from physrsr.active.runtime.instrumented import InstrumentedTarget
from physrsr.active.tools.stages import execute_program
from test_exploration import PROBE, Robot
from test_single_world_scene import SceneRobot, FIT_SCENE, RESPONSES


def bundle(path,source,kind='probe'):
    path.mkdir()
    policy=source_policy(source)
    (path/'program.py').write_text(source)
    save_json(path/'manifest.json',dict(kind=kind,api_version=policy.api_version,sha256=policy.sha256,
        status='authored_not_executed',debug_only=True,experiment_id='fixture'))
    return path


class RecordedTarget:
    def __init__(self,robot):self.target=InstrumentedTarget(robot);self.calls=[]
    def call(self,method,params):
        self.calls.append((method,deepcopy(params)))
        return self.target.call(method,params)


def test_debug_probe_continues_frozen_source_without_restarting_search(tmp_path):
    source=bundle(tmp_path/'probe',PROBE)
    robot=Robot(.04);target=RecordedTarget(robot)
    first=execute_program(tmp_path/'first',kind='probe',bundle=source,target=target)
    assert first['resumable'] and robot.moves==[0.]
    memory=json.loads((tmp_path/'first/belief.json').read_text())
    with pytest.raises(ValueError,match='unfinished progress'):
        execute_program(tmp_path/'unrequested_restart',kind='probe',bundle=source,target=target,belief=memory)
    assert sum(m=='target.execute_probe' for m,_ in target.calls)==1
    second=execute_program(tmp_path/'second',kind='probe',bundle=source,target=target,belief=memory,resume=True)
    assert not second['resumable'] and robot.moves==[0.,.01]
    assert [p['policy']['source'] for m,p in target.calls if m=='target.execute_probe']==[PROBE,PROBE]
    acquired=json.loads((tmp_path/'second/belief.json').read_text())
    assert len(acquired['trial_history'])==2 and acquired['previous_evidence_retained']
    assert not list(tmp_path.rglob('final'))


def test_debug_source_hash_mismatch_fails_before_touching_target(tmp_path):
    source=bundle(tmp_path/'probe',PROBE)
    (source/'program.py').write_text(PROBE+'\n# changed\n')
    class NoCalls:
        def call(self,*args):pytest.fail('Changed source must not reach the robot')
    with pytest.raises(ValueError,match='frozen hash'):
        execute_program(tmp_path/'execution',kind='probe',bundle=source,target=NoCalls())
    assert not (tmp_path/'execution').exists()


def test_debug_resume_requires_an_acknowledged_checkpoint(tmp_path):
    source=bundle(tmp_path/'probe',PROBE)
    target=RecordedTarget(Robot(.04))
    with pytest.raises(ValueError,match='acknowledged'):
        execute_program(tmp_path/'execution',kind='probe',bundle=source,target=target,resume=True)
    assert not any(method=='target.execute_probe' for method,_ in target.calls)


def test_debug_measures_all_objects_in_one_world_and_preserves_estimate_identity(tmp_path):
    source=bundle(tmp_path/'probe',FIT_SCENE)
    target=RecordedTarget(SceneRobot())
    result=execute_program(tmp_path/'execution',kind='probe',bundle=source,target=target)
    assert result['status']=='completed' and result['debug_only']
    memory=json.loads((tmp_path/'execution/belief.json').read_text())
    assert memory['object_ids']==sorted(RESPONSES)
    for key,value in RESPONSES.items():assert memory['objects'][key]['estimates'][0]['value']==value
    assert sum(method=='target.execute_probe' for method,_ in target.calls)==1
    assert not (tmp_path/'execution/evaluator_only_cases.json').exists()


def test_debug_policy_execution_does_not_certify_or_run_other_stages(tmp_path):
    source=bundle(tmp_path/'policy','def run(robot):\n    robot.close_gripper()\n    robot.hold(0.2)\n','policy')
    target=RecordedTarget(Robot(.04))
    result=execute_program(tmp_path/'execution',kind='policy',bundle=source,target=target)
    assert result['status']=='completed' and result['debug_only'] and result['task_result'] is None
    assert sum(method=='target.execute_policy' for method,_ in target.calls)==1
    assert not (tmp_path/'execution/manifest.json').exists()
    assert not (tmp_path/'execution/belief.json').exists()


def test_failed_recovery_keeps_valid_object_measurements(tmp_path):
    source=FIT_SCENE.replace("'status':'identified'","'status':'recovery_failed'")
    frozen=bundle(tmp_path/'probe',source)
    target=RecordedTarget(SceneRobot())
    result=execute_program(tmp_path/'execution',kind='probe',bundle=frozen,target=target)
    memory=json.loads((tmp_path/'execution/belief.json').read_text())
    assert result['program_status']=='recovery_failed' and len(memory['estimates'])==3
    assert memory['status']=='recovery_failed' and not result['resumable']


def test_belief_retention_keeps_objects_separate_and_does_not_duplicate_echoed_measurements():
    previous=dict(instance_id='current',object_ids=sorted(RESPONSES),estimates=[])
    first=merge_identification(previous,dict(status='identified',estimates=[dict(
        name='response',value=value,object_id=key,identifiable=True) for key,value in RESPONSES.items()]),
        source_probe='p1',evidence_id='measured-1')
    second=merge_identification(first,dict(status='recovery_failed',estimates=first['estimates']),
        source_probe='p2',evidence_id='measured-2')
    assert second['estimates']==first['estimates'] and second['status']=='recovery_failed'
    for key,value in RESPONSES.items():assert second['objects'][key]['estimates'][0]['value']==value


def test_debug_cli_requires_only_the_selected_agent_and_does_not_need_a_simulator(tmp_path,monkeypatch):
    import physrsr.paths as paths
    from physrsr.active import __main__ as cli
    from physrsr.active.tools import stages
    monkeypatch.setattr(paths,'OUTPUT_ROOT',tmp_path)
    config=tmp_path/'config.json';context=tmp_path/'public.json'
    save_json(config,dict(task=dict(id='fixture',instruction='Manipulate',success={},constraints={}),
        endpoints=dict(physic='http://127.0.0.1:1')))
    save_json(context,dict(initial_evidence=[]))
    calls=[]
    monkeypatch.setattr(stages,'author_program',lambda output,**kw:calls.append((output,kw)) or dict(status='authored_not_executed'))
    assert cli.main(['debug','design-probe','--config',str(config),'--context',str(context),'--out','debug_probe'])==0
    assert len(calls)==1 and calls[0][1]['kind']=='probe'
    assert calls[0][1]['task'].id=='fixture' and calls[0][0]==tmp_path/'debug_probe'
