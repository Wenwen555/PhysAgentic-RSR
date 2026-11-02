"""Offline exploration scheduling with executable numerical robot fixtures.

No LLM, physical simulator, experiment or hardware is invoked here.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json

import pytest

from physrsr.active.core.artifacts import save_json
from physrsr.active.core.exploration import CONTRACT,validate_plan,resumable
from physrsr.active.agents.identification import source_policy,resumable_probe_source,generate_design
from physrsr.active.runtime.instrumented import InstrumentedTarget,Dispatcher
from physrsr.active.runtime.robot import RobotContractError
from physrsr.active.workflows import identification as workflow,scoped_loop as loop
from test_unified_loop import FixtureRobot


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
            'unit':'N','frame':'robot_base_x','identifiable':True,'setting':setting})
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


class Endpoint:
    def __init__(self,robot):self.robot=robot;self.target=InstrumentedTarget(robot);self.calls=[]
    def call(self,method,params):
        self.calls.append((method,deepcopy(params)))
        if method!='world.evaluate_policy':return self.target.call(method,params)
        result=self.target.call('target.execute_policy',params)
        passed=result['status']=='completed' and abs(result['program_return']['value']-self.robot.response)<.001
        return dict(policy_sha256=result['policy_sha256'],observations=result['observations'],
            program_return=result['program_return'],error_kind=None,
            record=dict(status='success' if passed else 'fail',reason='' if passed else 'fixture_error'))


def fixture_run(tmp_path,monkeypatch,goal='Move the observed object'):
    cases=[dict(episode_id=e,endpoint='fixture://'+e,inspect_compilation=False) for e in ['a','b']]
    save_json(tmp_path/'evaluator_only_cases.json',cases)
    robots={'a':Robot(.04),'b':Robot(-.07)};endpoints={k:Endpoint(v) for k,v in robots.items()}
    @contextmanager
    def world(case,folder,port):yield endpoints[case['episode_id']]
    monkeypatch.setattr(workflow,'world_endpoint',world)
    stage=tmp_path/'active_probe_1';stage.mkdir()
    policy=source_policy(PROBE)
    (stage/'probe.py').write_text(PROBE)
    save_json(stage/'manifest.json',dict(probe_sha256=policy.sha256,status='authored_not_executed',
        experiment_id='first-question',change_kind='new_design'))
    context=dict(goal=dict(instruction=goal),capabilities=robots['a'].describe(),robot_api='public fixture API',
        initial_evidence=[],initial_visual_observations={e:{} for e in robots},experiment_budget={'max_rpc_calls':20})
    save_json(stage/'context.json',context);save_json(stage/'analysis.json',dict(selected_factors=['response']))
    coding=[];repairs=[]
    class Agents:
        def call(self,method,params):
            if method=='physic.design_identification':
                repairs.append(params)
                return dict(code=PROBE+'\n# repaired source\n',selected_factors=['response'],
                    experimental_design=dict(manipulated_variables=['contact setting'],minimum_distinct_settings=2))
            assert method=='agent.synthesize_identified_policy'
            coding.append(params)
            return dict(code=POLICY+f'\n# candidate {len(coding)}\n',rationale='fixture synthesis')
    @contextmanager
    def agents(*args,**kwargs):yield Agents()
    monkeypatch.setattr(workflow,'service',agents)
    return endpoints,coding,repairs


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
    return dict(action=action,rationale='Fixture scientific decision',instance_assessments=rows,
        target_instance_ids=list(targets),interactions=interactions,
        expected_decision_benefit=.5 if targets else 0.,estimated_probe_cost_s=4.*len(targets))


@pytest.mark.parametrize('goal',['Move the observed object','Reorient the observed object'])
def test_scoped_repair_resume_and_partial_candidate_do_not_repeat_sufficient_instance(tmp_path,monkeypatch,goal):
    endpoints,coding,repairs=fixture_run(tmp_path,monkeypatch,goal)
    actions=iter([('execute_probe',['a'],[]),('resume_probe',['a'],[]),
        ('repair_probe',[],['a']),('revise_policy',[],['a']),
        ('execute_probe',['b'],['a']),('resume_probe',['b'],['a']),('revise_policy',[],['a','b'])])
    contexts=[]
    def decide(root,context,index,port):
        contexts.append(deepcopy(context));action,targets,ready=next(actions)
        return plan(context,action,targets,ready)
    monkeypatch.setattr(loop,'decide',decide)
    result=loop.continue_probe_first(tmp_path,max_iterations=7)
    assert result['status']=='accepted' and result['policy_round']==2
    assert len(repairs)==1 and repairs[0]['authoring_mode']=='repair_probe'
    assert len(coding)==2 and coding[1]['previous_task_candidate'].startswith(POLICY)
    assert [x['episode_id'] for x in coding[1]['actual_task_feedback']]==['a']
    assert len(json.loads((tmp_path/'identified_cap_1/summary.json').read_text()))==1
    assert not (tmp_path/'identified_cap_1/evaluation/b').exists()
    for eid,endpoint in endpoints.items():
        executed=[p['policy']['source'] for m,p in endpoint.calls if m=='target.execute_probe']
        assert len(executed)==2 and executed[0]==executed[1]
        assert endpoint.robot.moves==[0.,.01], 'A resumed source must not restart the first setting'
    assert [m for m,p in endpoints['a'].calls].count('world.evaluate_policy')==2
    assert [m for m,p in endpoints['b'].calls].count('world.evaluate_policy')==1
    before=contexts[2]['measured_interactions'];after=contexts[3]['measured_interactions']
    assert before==after,'Repair must not spend physical interaction or create evidence'
    manifest=json.loads((tmp_path/'active_probe_2/manifest.json').read_text())
    assert manifest['experiment_id']=='first-question' and manifest['change_kind']=='repair'
    history=json.loads((tmp_path/'progress.json').read_text())
    assert history[2]['physical_interactions']==0
    assert contexts[2]['instances'][0]['retained_identification']['probe_progress']
    assert 'rod_lift' not in json.dumps(contexts)


def test_repair_never_implicitly_executes_new_source(tmp_path,monkeypatch):
    endpoints,_,_=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(loop,'decide',lambda root,ctx,*args:plan(ctx,'repair_probe'))
    result=loop.continue_probe_first(tmp_path,max_iterations=1)
    assert result['status']=='iteration_budget_exhausted'
    assert all(not e.calls for e in endpoints.values())
    assert not (tmp_path/'active_probe_2/world_execution').exists()


def test_controller_restart_retains_same_source_budget_pause(tmp_path,monkeypatch):
    endpoints,_,_=fixture_run(tmp_path,monkeypatch)
    calls=iter(['execute_probe','resume_probe'])
    monkeypatch.setattr(loop,'decide',lambda root,ctx,*args:plan(ctx,next(calls),['a']))
    first=loop.continue_probe_first(tmp_path,max_iterations=1)
    second=loop.continue_probe_first(tmp_path,max_iterations=1)
    assert first['status']==second['status']=='iteration_budget_exhausted'
    executed=[p['policy']['source'] for m,p in endpoints['a'].calls if m=='target.execute_probe']
    assert executed==[PROBE,PROBE] and endpoints['a'].robot.moves==[0.,.01]
    assert not endpoints['b'].calls
    assert second['authoring']['scientific_designs']==1 and second['measured_interactions']['executions']==2


def test_unacknowledged_attempt_cannot_be_repeated_as_a_fresh_probe(tmp_path,monkeypatch):
    endpoints,_,_=fixture_run(tmp_path,monkeypatch)
    save_json(tmp_path/'active_probe_1/world_execution/execution_index.json',[
        dict(attempt=1,status='running',instance_ids=['a'],results=[])])
    monkeypatch.setattr(loop,'decide',lambda *args:pytest.fail('Unknown interaction must be reconciled first'))
    result=loop.continue_probe_first(tmp_path,max_iterations=1)
    assert result['status']=='cost_unknown'
    assert all(not e.calls for e in endpoints.values())


@pytest.mark.parametrize('defect',['same_decision','cross_instance','already_sufficient','zero_cost'])
def test_physical_interaction_requires_scoped_decision_information(tmp_path,monkeypatch,defect):
    fixture_run(tmp_path,monkeypatch)
    context=loop.review_context(tmp_path,1)
    answer=plan(context,'execute_probe',['b'])
    if defect=='same_decision':answer['interactions'][0]['outcomes'][1]['operation_decision']='Choose configuration one'
    if defect=='cross_instance':answer['interactions'][0]['evidence_ids']=['a:initial_visual']
    if defect=='already_sufficient':
        answer['instance_assessments'][1]['sufficient']=True
        answer['instance_assessments'][1]['requirements'][0]['met']=True
    if defect=='zero_cost':answer['estimated_probe_cost_s']=0.
    with pytest.raises(ValueError):validate_plan(answer,context)


def test_resume_rejects_changed_source_and_failed_recovery(tmp_path,monkeypatch):
    fixture_run(tmp_path,monkeypatch)
    workflow.execute_identification(tmp_path,instance_ids=['a'])
    memory=workflow.instance_memory(tmp_path,'a',2)
    original=source_policy(PROBE)
    assert resumable(memory,original.sha256)
    assert not resumable(dict(memory,instance_id='another'),original.sha256)
    assert not resumable(memory,source_policy(PROBE+'\n# different\n').sha256)
    result=tmp_path/'active_probe_1/world_execution/a/result.json';raw=json.loads(result.read_text())
    raw['experiment_checkpoint']['resumable']=False;save_json(result,raw)
    with pytest.raises(ValueError,match='checkpoint'):
        workflow.execute_identification(tmp_path,instance_ids=['a'],resume=True)


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
            return dict(code=POLICY,experimental_design=dict(minimum_distinct_settings=2))
    agent=Agent()
    with pytest.raises(ValueError,match='authoring failed'):
        generate_design(agent,dict(goal={'instruction':'Manipulate'}),tmp_path)
    assert agent.calls==3
    resumable_probe_source(PROBE)


@pytest.mark.parametrize('budget,expected',[
    (loop.ExplorationBudget(max_probe_executions=1),'probe_budget_exhausted'),
    (loop.ExplorationBudget(probe_cost_weight=1.),'low_decision_value')])
def test_unaffordable_interaction_cannot_touch_robot(tmp_path,monkeypatch,budget,expected):
    endpoints,_,_=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(loop,'decide',lambda root,ctx,*args:plan(ctx,'execute_probe',['a','b']))
    result=loop.continue_probe_first(tmp_path,max_iterations=1,budget=budget)
    assert result['status']==expected
    assert all(not e.calls for e in endpoints.values())


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
    native,events=fixture.build(tmp_path,answers=Answers(),cfg=LoopConfig(max_rounds=3))
    result=native.run()
    assert result['probe_count']==3 and len(native.fit_data)==3 and native.model_version==2
    assert not any(m=='agent.synthesize_identified_policy' for m,p in events)
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
    model_calls=[p for m,p in events if m=='physic.revise_predictive_model']
    assert len(model_calls)==1 and len(model_calls[0]['fit_evidence'])==1
    assert result['status']==('exploration_stopped' if mismatch_first else 'task_sufficient')


def test_cli_exposes_scoped_same_source_resume_and_code_only_repair(tmp_path,monkeypatch):
    import physrsr.active.__main__ as cli
    monkeypatch.setattr(cli,'output_path',Path)
    calls=[]
    monkeypatch.setattr(workflow,'execute_identification',lambda root,**kwargs:calls.append(('execute',kwargs)) or [])
    assert cli.main(['identification','probe','--experiment',str(tmp_path),'--instances','b','--resume'])==0
    assert calls==[('execute',dict(round_index=1,instance_ids=['b'],resume=True))]
    review=tmp_path/'review.json';save_json(review,dict(finding='program defect'))
    monkeypatch.setattr(workflow,'redesign_identification',lambda root,review,**kwargs:calls.append(('repair',review,kwargs)) or {})
    cli.main(['identification','redesign','--experiment',str(tmp_path),'--round','2','--review',str(review),'--repair'])
    assert calls[-1][0]=='repair' and calls[-1][1]['mode']=='repair_probe' and len(calls)==2


def test_cli_passes_global_interaction_budgets_to_batch(tmp_path,monkeypatch):
    import physrsr.active.__main__ as cli
    from physrsr.active.workflows import batch
    monkeypatch.setattr(cli,'output_path',Path)
    config=tmp_path/'config.json';cases=tmp_path/'cases.json'
    save_json(config,dict(task=dict(id='fixture',instruction='Manipulate',success={},constraints={}),
        loop=dict(max_probes=7,max_probe_cost=40.,probe_cost_weight=.02)))
    save_json(cases,[dict(episode_id='a'),dict(episode_id='b')])
    calls=[]
    monkeypatch.setattr(batch,'run_probe_first_batch',lambda *args,**kwargs:calls.append(kwargs) or dict(status='iteration_budget_exhausted'))
    assert cli.main(['run','--config',str(config),'--cases',str(cases),'--out',str(tmp_path/'out')])==2
    budget=calls[0]['budget']
    assert budget.max_probe_executions==7 and budget.max_probe_cost_s==40. and budget.probe_cost_weight==.02
