"""Offline exploration scheduling with executable numerical robot fixtures.

No LLM, physical simulator, experiment or hardware is invoked here.
"""
import json

import pytest

from physrsr.active.core.exploration import CONTRACT,validate_plan,resumable
from physrsr.active.core.programs import source_policy,resumable_probe_source
from physrsr.active.agents.physic import PhysicAgent
from physrsr.active.runtime.instrumented import InstrumentedTarget,Dispatcher
from physrsr.active.runtime.robot import RobotContractError
from test_unified_loop import FixtureRobot
from agent_fixtures import design_answer


PROBE="""def run(robot):
    execution=robot.describe()['probe_execution']
    state=execution['resume_state'] or {'next':0,'tried':[]}
    estimates=robot.describe()['identification'].get('estimates',[])
    while state['next']<2:
        execution=robot.describe()['probe_execution']
        if execution['remaining_rpc_calls']<14:
            robot.open_gripper()
            robot.hold(0.2)
            robot.observe()
            robot.checkpoint(state)
            return {'status':'budget_exhausted','estimates':estimates,'trials':state['tried']}
        first=robot.observe()
        setting=state['next']
        p=list(first['tcp']['position'])
        p[0]=0.01*setting
        robot.goto_pose(p,first['tcp']['quaternion_wxyz'],speed=0.1)
        robot.close_gripper()
        robot.hold(0.2)
        value=robot.observe()['wrench']['force_n'][0]
        robot.open_gripper()
        robot.hold(0.2)
        robot.observe()
        estimates.append({'name':'response','value':value,'interval':[value-.001,value+.001],
            'unit':'N','frame':'robot_base_x','identifiable':True,'assumptions':['fixture only'],'setting':setting})
        state['tried'].append({'setting':setting})
        state['next']+=1
    return {'status':'identified','estimates':estimates,'trials':state['tried']}
"""
POLICY="""def run(robot):
    memory=robot.describe()['identification']['estimates']
    robot.close_gripper()
    robot.hold(0.2)
    return {'value':memory[-1]['value']}
"""


class Robot(FixtureRobot):
    def __init__(self,response):
        self.moves=[]
        super().__init__(response)
    def describe(self):
        cap=super().describe()
        cap['gripper']['max_opening_m']=.04
        cap['experiment_budget']={'max_rpc_calls':20}
        return cap
    def action(self,method,params):
        super().action(method,params)
        if method=='open_gripper':self.state['gripper']['opening_m']=.04
        if method=='close_gripper':self.state['gripper']['opening_m']=.02
        if method=='goto_pose':self.moves.append(params['position'][0])


def plan(context,action,targets=(),ready=()):
    rows=[];interactions=[]
    for instance in context['instances']:
        eid=instance['episode_id'];evidence=instance['available_evidence_ids']
        rows.append(dict(episode_id=eid,sufficient=eid in ready,reason='Observed fixture requirement',
            evidence_ids=evidence,requirements=[dict(quantity_or_condition='Two independent settings',
                operation_decision='Choose the operating configuration',met=eid in ready,evidence_ids=evidence)]))
        if eid in targets:
            interactions.append(dict(episode_id=eid,uncertainty='Setting-dependent response',
                why_existing_evidence_insufficient='The contrast has not been observed',evidence_ids=evidence,
                outcomes=[dict(condition='Response positive',operation_decision='Choose configuration one'),
                    dict(condition='Response negative',operation_decision='Choose configuration two')],estimated_cost_s=4.))
    return dict(action=action,rationale='Fixture scientific decision',
        value_basis='none' if not targets else 'validation' if action=='validate' or context.get('current_probe',{}).get('purpose')=='validation' else 'bootstrap' if not context.get('model_version') else 'new_structure',
        instance_assessments=rows,
        target_instance_ids=list(targets),interactions=interactions,
        expected_decision_benefit=.5 if targets else 0.,estimated_probe_cost_s=4.*len(targets))


@pytest.mark.parametrize('defect',['same_decision','cross_instance','already_sufficient','zero_cost'])
def test_physical_interaction_requires_scoped_decision_information(tmp_path,monkeypatch,defect):
    context=dict(instances=[dict(episode_id=e,available_evidence_ids=[f'{e}:initial_visual']) for e in ['a','b']])
    answer=plan(context,'execute_probe',['b'])
    if defect=='same_decision':answer['interactions'][0]['outcomes'][1]['operation_decision']='Choose configuration one'
    if defect=='cross_instance':answer['interactions'][0]['evidence_ids']=['a:initial_visual']
    if defect=='already_sufficient':
        answer['instance_assessments'][1]['sufficient']=True
        answer['instance_assessments'][1]['requirements'][0]['met']=True
    if defect=='zero_cost':answer['estimated_probe_cost_s']=0.
    with pytest.raises(ValueError):validate_plan(answer,context)


def test_checkpoint_cannot_be_accepted_while_gripping_or_after_more_motion():
    robot=Robot(.04);robot.belief={'instance_id':'a'};dispatch=Dispatcher(robot,robot.observe(),program_sha256='frozen')
    dispatch('hold',dict(duration_s=.2))
    dispatch('close_gripper',{})
    with pytest.raises(RobotContractError,match='open gripper'):dispatch('checkpoint',dict(state={'next':1}))
    dispatch('open_gripper',{});dispatch('hold',dict(duration_s=.2))
    dispatch('checkpoint',dict(state={'next':1}))
    assert dispatch.checkpoint_result('completed',dict(status='paused'))['resumable']
    dispatch('hold',dict(duration_s=.2))
    assert not dispatch.checkpoint_result('completed',dict(status='paused'))['resumable']


def test_missing_continuation_is_a_code_only_authoring_failure(tmp_path):
    class Agent:
        calls=0
        def generate(self,*args,**kwargs):
            self.calls+=1
            return design_answer(dict(goal={'instruction':'Manipulate'}),POLICY)
    agent=Agent()
    from physrsr.active.api.client import APIError
    with pytest.raises(APIError,match='authoring failed'):
        PhysicAgent(tmp_path/'physic',agent=agent).design_probe(dict(goal={'instruction':'Manipulate'}))
    assert agent.calls==3
    resumable_probe_source(PROBE)


def test_predictive_loop_repairs_program_error_without_automatic_reexecution(tmp_path):
    from test_unified_loop import build,StructuredAnswers,FIT
    from physrsr.active.core.contracts import LoopConfig
    broken="""def run(robot):
    state=robot.describe()['probe_execution']['resume_state']
    robot.observe()
    raise RuntimeError('fixture program defect')
    robot.checkpoint(state or {})
"""
    class Answers(StructuredAnswers):
        designs=0
        def generate(self,messages,destination,*,schema):
            props=schema['properties']
            if 'action' in props:
                context=json.loads(messages[0]['content'].split('PUBLIC CONTEXT\n')[1])
                failed=bool(context['instances'][0]['latest_execution'])
                return plan(context,'repair_probe' if failed else 'probe',[] if failed else ['current'])
            answer=super().generate(messages,destination,schema=schema)
            if 'experimental_design' in props:
                self.designs+=1;answer['code']=broken if self.designs==1 else FIT
            return answer
    agent=Answers();native,events=build(tmp_path,answers=agent,cfg=LoopConfig(max_rounds=2))
    result=native.run()
    assert result['probe_count']==1 and result['probe_designs']==result['probe_repairs']==1
    assert sum(m=='target.execute_probe' for m,p in events)==1
    assert not any(m=='sim.configure' for m,p in events)
    assert (native.output/'active_probe_2/probe.py').read_text()==FIT
    assert not (native.output/'active_probe_2/world_execution').exists()


def test_predictive_loop_resumes_the_same_source_and_replays_both_measured_segments(tmp_path,monkeypatch):
    import test_unified_loop as fixture
    from physrsr.active.core.contracts import LoopConfig
    monkeypatch.setattr(fixture,'FixtureRobot',Robot)
    class Answers(fixture.StructuredAnswers):
        actions=0
        def generate(self,messages,destination,*,schema):
            props=schema['properties'];content=messages[0]['content']
            if 'action' in props:
                context=json.loads(content.split('PUBLIC CONTEXT\n')[1]);self.actions+=1
                return plan(context,'probe' if self.actions==1 else 'resume_probe',['current'])
            answer=super().generate(messages,destination,schema=schema)
            if 'experimental_design' in props:answer['code']=PROBE
            if 'models' in props:
                context=json.loads(content.split('Context:\n')[1])
                answer['evidence_ids']=[e['evidence_id'] for e in context['fit_evidence']]
            return answer
    native,events=fixture.build(tmp_path,answers=Answers(),cfg=LoopConfig(max_rounds=2))
    result=native.run()
    calls=[p['policy']['source'] for m,p in events if m=='target.execute_probe']
    assert calls==[PROBE,PROBE]
    assert result['probe_count']==2 and result['probe_program_versions']==1
    assert len(native.fit_data)==2 and native.model_version==1
    assert (native.output/'active_probe_1/world_execution/attempt_002/result.json').exists()
    assert not resumable(native.belief,source_policy(PROBE).sha256)


def test_predictive_loop_can_continue_identification_before_a_task_candidate(tmp_path,monkeypatch):
    import test_unified_loop as fixture
    from physrsr.active.core.contracts import LoopConfig
    monkeypatch.setattr(fixture,'FixtureRobot',Robot)
    class Answers(fixture.StructuredAnswers):
        actions=0;designs=0
        def generate(self,messages,destination,*,schema):
            props=schema['properties'];content=messages[0]['content']
            if 'action' in props:
                context=json.loads(content.split('PUBLIC CONTEXT\n')[1]);self.actions+=1
                return plan(context,'resume_probe' if self.actions==3 else 'probe',['current'])
            answer=super().generate(messages,destination,schema=schema)
            if 'experimental_design' in props:
                self.designs+=1
                # The one-shot fixture uses ten calls and fits the declared
                # twenty-call budget; only the second experiment must pause.
                answer['code']=PROBE if self.designs==2 else fixture.FIT.replace('<20:','<12:')
            if 'models' in props:
                context=json.loads(content.split('Context:\n')[1])
                answer['evidence_ids']=[e['evidence_id'] for e in context['fit_evidence']]
            return answer
    native,events=fixture.build(tmp_path,answers=Answers(),cfg=LoopConfig(max_rounds=3,max_heuristic_probes=3))
    result=native.run()
    assert result['probe_count']==3 and len(native.fit_data)==3 and native.model_version==2
    assert not any(m=='agent.synthesize_policy' for m,p in events)
    resumed=[p for m,p in events if m=='sim.predict_probe' and p['probe']['policy']['source']==PROBE]
    assert any(resumable(p['belief'],source_policy(PROBE).sha256) for p in resumed)


@pytest.mark.parametrize('mismatch_first',[False,True])
def test_paused_independent_validation_needs_all_segments_and_never_fits_them(tmp_path,monkeypatch,mismatch_first):
    import test_unified_loop as fixture
    from physrsr.active.core.contracts import LoopConfig
    monkeypatch.setattr(fixture,'FixtureRobot',Robot)
    actions=iter(['probe','revise_policy','validate','resume_probe','stop'])
    class Answers(fixture.StructuredAnswers):
        def generate(self,messages,destination,*,schema):
            props=schema['properties'];content=messages[0]['content']
            if 'action' in props:
                context=json.loads(content.split('PUBLIC CONTEXT\n')[1]);action=next(actions)
                if action=='resume_probe':
                    assert not context['validation_reports'], 'A paused check cannot certify independent validation'
                return plan(context,action,['current'] if action in ('probe','validate','resume_probe') else [],
                    [] if action in ('probe','validate','resume_probe') else ['current'])
            answer=super().generate(messages,destination,schema=schema)
            if 'experimental_design' in props:
                context=json.loads(content.split('PUBLIC CONTEXT\n')[1])
                answer['code']=PROBE if context['purpose']=='validation' else fixture.FIT.replace('<20:','<12:')
            return answer
    class Target(InstrumentedTarget):
        segments=0
        def call(self,method,params):
            if method=='target.execute_probe' and params['policy']['source']==PROBE:
                self.segments+=1
                self.robot.response=.9 if mismatch_first and self.segments==1 else .04
            return super().call(method,params)
    native,events=fixture.build(tmp_path,answers=Answers(),cfg=LoopConfig(max_rounds=5),target=Target(Robot(.04)))
    result=native.run()
    assert len(native.validation_data)==1 and len(native.validation_data[0]['segment_evidence_ids'])==2
    assert native.validation_data[0]['passed'] is (not mismatch_first)
    assert len(native.fit_data)==1 and not native.belief.get('probe_progress')
    model_calls=[p for m,p in events if m=='physic.revise_physical_model']
    assert len(model_calls)==1 and len(model_calls[0]['fit_evidence'])==1
    assert result['status']==('exploration_stopped' if mismatch_first else 'task_sufficient')
