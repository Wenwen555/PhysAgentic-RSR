"""Adversarial regressions from architecture review; no LLM or hardware calls."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from physrsr.active.core.bundle import execute_bundle
from physrsr.active.core.contracts import Hypothesis, LoopConfig, IDENTIFICATION_API_VERSION, read_config
from physrsr.active.core.measurements import response_features, scene_scales, FEATURE_SCALES
from physrsr.active.core.policy import Policy
from physrsr.active.core.reasoning import joint_fit, posterior
from physrsr.active.core.validation import check_executed, validate_contract
from physrsr.active.runtime.predictive import PredictiveSimulator
from physrsr.active.runtime.instrumented import InstrumentedTarget, Dispatcher
from physrsr.active.runtime.robot import RobotContractError
from physrsr.active.tasks.rod_lift import RodMeasuredAdapter, ROD_SUCCESS, FIXED_MASS_MECHANISM
from physrsr.active.core.belief import merge_identification
from test_unified_loop import build, sensor_packet, FixtureRobot, StructuredAnswers, VALIDATE


def prediction(features):return dict(features=features,risk=0.,duration_s=1.)


def test_two_incompatible_models_cannot_explain_different_fit_segments():
    models=[Hypothesis('a',{'x':0.},.5,'fixture'),Hypothesis('b',{'x':1.},.5,'fixture')]
    data=[dict(evidence_id='first',actual=[0.],predictions={'a':prediction([0.]),'b':prediction([5.])}),
          dict(evidence_id='second',actual=[0.],predictions={'a':prediction([5.]),'b':prediction([0.])})]
    fitted,report=joint_fit(models,data,[1.],3.)
    assert fitted is None and report['accepted_model_ids']==[]
    assert report['errors']['a']['first']==0. and report['errors']['a']['second']==5.


def test_full_dataset_refit_is_idempotent_and_ignores_agent_posterior_weights():
    models=[Hypothesis('a',{'x':0.},.9,'fixture'),Hypothesis('b',{'x':1.},.1,'fixture')]
    data=[dict(evidence_id='one',actual=[0.],predictions={'a':prediction([0.]),'b':prediction([1.])})]
    first,_=joint_fit(models,data,[1.],3.)
    second,_=joint_fit(first,data,[1.],3.)
    assert [m.weight for m in first]==[m.weight for m in second]
    assert first[0].weight==pytest.approx(.6224593312)


def test_shared_wrist_channels_do_not_get_stronger_when_object_tracks_are_added():
    low=sensor_packet();high=deepcopy(low);high['time_s']=1.;high['wrench']['force_n'][0]=1.
    single=[dict(observation=low),dict(observation=high)]
    scene=[]
    for sample in single:
        obs=deepcopy(sample['observation'])
        obs['objects']={key:deepcopy(obs['object']) for key in ['a','b','c']}
        scene.append(dict(observation=obs))
    one=response_features(single);three=response_features(scene)
    assert len(one)==16 and len(three)==28
    models=[Hypothesis('a',{'x':0.},.5,'fixture'),Hypothesis('b',{'x':1.},.5,'fixture')]
    one_p={'a':prediction(one),'b':prediction([0.]*len(one))}
    scene_p={'a':prediction(three),'b':prediction([0.]*len(three))}
    a=posterior(models,[0.]*16,one_p,FEATURE_SCALES)
    b=posterior(models,[0.]*28,scene_p,scene_scales(FEATURE_SCALES,3))
    assert [m.weight for m in a]==[m.weight for m in b]


def test_repeated_same_timestamp_observation_does_not_change_time_mean():
    first=sensor_packet();last=deepcopy(first);last['time_s']=1.;last['wrench']['force_n'][0]=1.
    baseline=[dict(observation=first),dict(observation=last)]
    repeated=[dict(observation=deepcopy(first)) for _ in range(11)]+[dict(observation=last)]
    assert response_features(baseline)==response_features(repeated)
    assert response_features(baseline)[10]==.5


def test_rod_program_error_keeps_duration_and_reaches_predictive_feedback(monkeypatch):
    import physrsr.active.tasks.rod_lift_world as module
    class World:
        def __init__(self,*args):pass
        def call(self,method,params):
            if method=='target.update_identification':return dict(updated=True)
            first=sensor_packet();last=deepcopy(first);last['time_s']=.4
            return dict(record=dict(status='error',reason='NameError: generated typo',sim_duration_s=None),
                observations=[dict(observation=first),dict(observation=last)],program_return=None,error_kind='program')
        def close(self):pass
    monkeypatch.setattr(module,'InstrumentedWorld',World)
    sim=PredictiveSimulator([RodMeasuredAdapter()])
    model=Hypothesis('m',dict(mass_kg=.2,com_offset=.02,friction=1.),1.,FIXED_MASS_MECHANISM)
    task=dict(id='rod_lift',success=ROD_SUCCESS,constraints=dict(grip_force_per_finger_n=2.))
    sim.call('sim.configure',dict(session_id='one',task=task,models=[asdict(model)]))
    policy=Policy('def run(robot):\n    robot.close_gripper()\n',api_version=IDENTIFICATION_API_VERSION)
    result=sim.call('sim.evaluate',dict(session_id='one',model_version=0,seeds=[1,2],policies=[policy.wire()]))
    row=result['rows'][0]
    assert row['duration_s']==.4 and row['success_rate']==0.
    assert row['feedback'][0]['error_kind']=='program' and 'generated typo' in row['feedback'][0]['reason']


def test_later_validation_pass_cannot_clear_same_version_contradiction(tmp_path):
    class Answers(StructuredAnswers):
        validations=0
        def generate(self,messages,destination,*,schema):
            answer=super().generate(messages,destination,schema=schema)
            if 'experimental_design' in schema['properties']:
                context=json.loads(messages[0]['content'].split('PUBLIC CONTEXT\n')[1])
                if context['purpose']=='validation':
                    self.validations+=1
                    answer['code']=VALIDATE.replace('hold(0.3)',f'hold({.3+self.validations*.1})')
            return answer
    class Changed(InstrumentedTarget):
        changed=False
        def call(self,method,params):
            if method=='target.execute_probe' and 'validation' in params['policy']['source']:
                self.robot.response=.9 if not self.changed else .04
                self.changed=True
            return super().call(method,params)
    loop,_=build(tmp_path,answers=Answers(),target=Changed(FixtureRobot(.04)),
        cfg=LoopConfig(max_rounds=5,max_probes=2,max_prediction_error=.1))
    assert loop.run()['final_policy'] is None
    assert loop.needs_identification and len(loop.unresolved_failures())==1
    loop.cfg=replace(loop.cfg,max_probes=4)
    ctx=loop.context();decision=loop.physic.call('physic.assess_task',ctx)
    candidates=list(loop.store.policies.values())
    loop.probe('validation',ctx,decision,candidates)
    assert loop.validation_data[-1]['passed'] and loop.needs_identification
    assert len(loop.unresolved_failures())==1 and loop.model_version==1
    stop=dict(expected_decision_benefit=0.,estimated_probe_cost_s=0.)
    assert loop.handle_stop(ctx,stop,candidates,{})['status']=='exploration_stopped'
    assert not (loop.output/'final').exists()


def test_source_alpha_renaming_is_only_duplicate_detection_not_certification():
    a=Policy('def run(robot):\n    x=robot.observe()\n    robot.hold(.1)\n')
    b=Policy('def run(robot):\n    y=robot.observe()\n    robot.hold(.1)\n')
    assert a.sha256!=b.sha256 and a.behavior_sha256==b.behavior_sha256


def test_observe_only_execution_cannot_satisfy_a_declared_validation_intervention():
    packet=sensor_packet();last=deepcopy(packet);last['time_s']=1.
    contract=dict(settings=[dict(object_id='object',tcp_offset_body_m=[0.,0.,0.],setting_tolerance_m=.003,
        min_duration_s=.1,min_translation_m=0.,min_rise_m=0.,min_force_change_n=0.)])
    report=check_executed(contract,[dict(event='observe',observation=packet),dict(event='observe',observation=last)])
    assert not report['passed'] and 'no measured contact intervention' in report['errors'][0]


def test_wrong_candidate_binding_or_weaker_excitation_is_rejected_before_execution():
    context=dict(candidate_sha256='current',models=[dict(id='m',weight=1.)],max_prediction_error=1.,
        validation_requirements={'object':dict(min_duration_s=.2,min_translation_m=.02,min_rise_m=.01,min_force_change_n=.2)})
    contract=dict(alternative_explanation='Unmodeled contact',task_operation='Loaded contact',candidate_sha256='old',object_ids=['object'],hypothesis_ids=['m'],settings=[dict(object_id='object',
        min_duration_s=.1,min_translation_m=0.,min_rise_m=0.,min_force_change_n=0.)],max_prediction_error=1.)
    with pytest.raises(ValueError,match='current candidate'):validate_contract(contract,context)
    contract['candidate_sha256']='current'
    with pytest.raises(ValueError,match='weaker'):validate_contract(contract,context)


@pytest.mark.parametrize('score,expected',[(True,0),(False,2),(None,3)])
def test_formal_execute_restores_matching_memory_and_distinguishes_task_failure(tmp_path,monkeypatch,score,expected):
    class Scoring(InstrumentedTarget):
        task_contract=dict(id='translate',success={},constraints={})
        def call(self,method,params):
            result=super().call(method,params)
            if method=='target.execute_policy':result['task_result']=dict(success=score) if score is not None else None
            return result
    target=Scoring(FixtureRobot(.04));loop,_=build(tmp_path,target=target)
    assert loop.run()['status']=='task_sufficient'
    target.call('target.update_identification',dict(belief=dict(estimates=[],instance_id='current')))
    from physrsr.active.__main__ import main
    monkeypatch.setattr('physrsr.active.__main__.HTTPClient',lambda url:target)
    assert main(['execute','--bundle',str(loop.output/'final'),'--target-url','http://fixture'])==expected
    assert target.robot.belief['estimates'][0]['value']==.04
    wrong=Scoring(FixtureRobot(.04))
    with pytest.raises(ValueError,match='different World'):execute_bundle(loop.output/'final',wrong)
    assert wrong.robot.belief['estimates']==[]


def test_frozen_memory_tampering_is_rejected_before_target_calls(tmp_path):
    loop,_=build(tmp_path);loop.run()
    path=loop.output/'final/identification.json'
    path.write_text('{"estimates":[]}')
    class NoCalls:
        def call(self,*args):pytest.fail('Tampering must be rejected before reaching target')
    with pytest.raises(ValueError,match='hash mismatch'):execute_bundle(loop.output/'final',NoCalls())


@pytest.mark.parametrize('field,value',[('value',float('nan')),('interval',[2.,1.]),('unit',''),('frame','')])
def test_invalid_estimate_cannot_enter_policy_memory(field,value):
    estimate=dict(name='mass',value=1.,interval=[.9,1.1],unit='kg',frame='scalar',identifiable=True,assumptions=['measured'])
    estimate[field]=value
    with pytest.raises(ValueError):merge_identification(dict(estimates=[]),dict(estimates=[estimate]),source_probe='p',evidence_id='e')


def test_execution_hard_duration_and_contact_count_are_enforced_before_action():
    robot=FixtureRobot(.04)
    dispatcher=Dispatcher(robot,robot.observe(),budget=dict(max_duration_s=.1))
    with pytest.raises(RobotContractError,match='remaining execution duration'):dispatcher('hold',dict(duration_s=.2))
    assert robot.state['time_s']==0.
    dispatcher=Dispatcher(robot,robot.observe(),budget=dict(max_distinct_settings=1))
    dispatcher('close_gripper',{})
    robot.state['tcp']['position'][0]+=.01
    with pytest.raises(RobotContractError,match='setting budget'):dispatcher('close_gripper',{})


@pytest.mark.parametrize('defect',['unknown_task','empty_success','wrong_dimensions'])
def test_invalid_config_is_rejected_offline(defect):
    config=json.loads(Path('physrsr/active/examples/rod_lift.json').read_text())
    if defect=='unknown_task':config['task']['id']='missing'
    if defect=='empty_success':config['task']['success']={}
    if defect=='wrong_dimensions':config['loop']['feature_scales']=[1.,1.]
    with pytest.raises((ValueError,RuntimeError)):read_config(config)


def test_actual_loop_joint_fit_rejects_patchwork_without_committing(tmp_path,monkeypatch):
    from physrsr.active.core.reasoning import Probe
    loop,events=build(tmp_path,cfg=LoopConfig(feature_scales=(1.,)*16))
    models=(Hypothesis('a',{'response':.01},.5,'fixture_response_v2'),Hypothesis('b',{'response':.02},.5,'fixture_response_v2'))
    loop.models=models;loop.sim_configured=True
    loop.simulator.call('sim.configure',dict(session_id=loop.session_id,
        task=loop.task.public(IDENTIFICATION_API_VERSION),models=[asdict(m) for m in models]))
    policy=Policy('def run(robot):\n    robot.observe()\n',api_version=IDENTIFICATION_API_VERSION)
    loop.fit_data=[dict(probe=Probe(key,policy,1.,('response',),'contrast'),result=dict(evidence_id=key,features=[0.]*16),belief_before={})
        for key in ['first','second']]
    def predict(probe,**kwargs):
        return {m.id:prediction([0. if (probe.id=='first')==(m.id=='a') else 5.]*16) for m in models}
    monkeypatch.setattr(loop,'predict',predict)
    assert not loop.update_model()
    assert loop.model_version==0 and not any(method=='sim.update_model' for method,_ in events)
    assert loop.model_diagnostics['accepted_model_ids']==[]


def test_actual_loop_does_not_condition_twice_on_unchanged_fit_evidence(tmp_path):
    loop,_=build(tmp_path);assert loop.run()['status']=='task_sufficient'
    loop.models=(Hypothesis('a',{'response':.04},.5,'fixture_response_v2'),Hypothesis('b',{'response':.03},.5,'fixture_response_v2'))
    # Keep the same physical support; bypass another parameter proposal.
    loop.belief=dict(loop.belief,estimates=[])
    assert loop.update_model()
    first=[m.weight for m in loop.models]
    assert loop.update_model()
    assert [m.weight for m in loop.models]==first and first[0]>.5


def test_failed_validation_cannot_be_retired_by_version_bump_without_new_fit(tmp_path):
    loop,events=build(tmp_path);loop.run()
    fit_ids=[d['result']['evidence_id'] for d in loop.fit_data]
    loop.validation_failures.append(dict(evidence_id='heldout',model_version=loop.model_version,fit_evidence_ids=fit_ids))
    before=loop.model_version;calls=sum(method=='sim.update_model' for method,_ in events)
    assert not loop.update_model() and loop.model_version==before
    assert sum(method=='sim.update_model' for method,_ in events)==calls


def test_executable_probe_comparison_selects_decision_information(tmp_path,monkeypatch):
    from physrsr.active.core.reasoning import Probe
    loop,events=build(tmp_path,cfg=LoopConfig(feature_scales=(1.,)*16))
    loop.models=(Hypothesis('a',{'response':.01},.5,'fixture_response_v2'),Hypothesis('b',{'response':.02},.5,'fixture_response_v2'))
    loop.sim_configured=True
    policies=[Policy('def run(robot):\n    robot.hold(.1)\n'),Policy('def run(robot):\n    robot.hold(.2)\n')]
    loop.rows=[dict(model_id=m.id,policy_sha256=p.sha256,success_rate=float((m.id=='a')==(i==0)),
        risk=0.,duration_s=0.,margin=.1) for m in loop.models for i,p in enumerate(policies)]
    programs=[Probe(name,Policy('def run(robot):\n    robot.hold('+str(duration)+')\n'),1.,('response',),'contrast')
        for name,duration in [('unhelpful',.1),('discriminating',.2)]]
    remaining=list(programs);predicted=[]
    for probe in programs:loop.recorder.write_json(probe.id+'/manifest.json',dict(experiment_id=probe.id))
    monkeypatch.setattr(loop,'author_probe',lambda *args:remaining.pop(0))
    def predict(probe,**kwargs):
        predicted.append(probe.id)
        return {m.id:prediction([0.]*16 if probe.id=='unhelpful' else [(-10. if m.id=='a' else 10.)]*16) for m in loop.models}
    monkeypatch.setattr(loop,'predict',predict)
    chosen=loop.select_probe('fit',{},dict(value_basis='existing_candidates'),policies)
    assert chosen.id=='discriminating' and predicted==['unhelpful','discriminating']
    assert not any(method=='target.execute_probe' for method,_ in events)


def test_agent_optimism_cannot_override_low_computed_probe_value(tmp_path):
    from test_exploration import plan
    class Distinct(StructuredAnswers):
        def generate(self,messages,destination,*,schema):
            answer=super().generate(messages,destination,schema=schema)
            if 'experimental_design' in schema['properties']:
                context=json.loads(messages[0]['content'].split('PUBLIC CONTEXT\n')[1])
                if context.get('executable_candidate_index')==1:answer['code']=answer['code'].replace('hold(0.1)','hold(0.2)')
            return answer
    loop,events=build(tmp_path,answers=Distinct());loop.run()
    before=loop.probe_count;ctx=loop.context();decision=plan(ctx,'probe',['current'])
    decision.update(value_basis='existing_candidates',expected_decision_benefit=1e9)
    assert loop.probe('fit',ctx,decision,list(loop.store.policies.values()))=='low_value'
    assert loop.probe_count==before
    comparisons=sorted((loop.output/'probe_comparisons').glob('*.json'))
    assert len(json.loads(comparisons[-1].read_text())['candidates'])==2
    decision['value_basis']='new_structure'
    loop.cfg=replace(loop.cfg,max_heuristic_probes=0)
    assert loop.probe('fit',ctx,decision,list(loop.store.policies.values()))=='heuristic_budget_exhausted'
    assert loop.probe_count==before


def test_rollout_worker_error_records_measured_physics_time(monkeypatch):
    from types import SimpleNamespace as NS
    import physrsr.worker as module
    from physrsr.physics_spec import PhysicsSpec
    from physrsr.execution import ProgramError
    data=NS(time=0.)
    suite=NS(sim=NS(data=data),seed_episode=lambda seed:None)
    env=NS(robosuite_env=suite,reset=lambda seed:None)
    tracker=NS(stage='contact',grasp_established=False,record_initial=lambda:None,as_dicts=lambda:[])
    monkeypatch.setattr(module,'PhaseTracker',lambda *args:tracker)
    worker=module.RolloutWorker(PhysicsSpec())
    monkeypatch.setattr(worker,'_ensure_env',lambda:(env,NS(cfg=None)))
    class Broken:
        id='broken'
        def run(self,*args):
            data.time=1.25
            raise ProgramError('generated defect')
    record=worker.rollout(Broken(),1)
    assert record.status=='error' and record.sim_duration_s==1.25 and record.duration_s>=0.


def test_open_gripper_resets_phase_tracker_before_regrasp():
    from types import SimpleNamespace as NS
    import numpy as np
    from physrsr.phase_tracker import PhaseTracker
    tracker=PhaseTracker(None,NS(open_gripper=lambda:None),state_fn=lambda env:{})
    tracker.closed=True;tracker.stage='transport';tracker._last_command=np.array([.1,0.,.1])
    tracker.open_gripper()
    assert not tracker.closed and tracker.stage=='pre_grasp'
    assert tracker._classify_move(np.array([.2,0.,.1]))=='pre_grasp'


def test_declared_setting_cannot_borrow_excitation_from_another_setting():
    packet=sensor_packet();samples=[]
    def add(event,t,x):
        obs=deepcopy(packet);obs['time_s']=t;obs['tcp']['position'][0]=x
        samples.append(dict(event=event,observation=obs,params={}))
    add('initial',0.,0.);add('close_gripper',0.,0.);add('hold',.01,0.);add('open_gripper',.01,0.)
    add('close_gripper',.01,.02);add('hold',1.01,.02);add('open_gripper',1.01,.02)
    setting=dict(object_id='object',tcp_offset_body_m=[0.,0.,0.],setting_tolerance_m=.003,
        min_duration_s=.2,min_translation_m=0.,min_rise_m=0.,min_force_change_n=0.)
    assert not check_executed(dict(settings=[setting]),samples)['passed']


def test_run_manifest_contains_real_sources_controller_and_agent_bindings(tmp_path):
    import hashlib
    loop,_=build(tmp_path);loop.run()
    manifest=json.loads((loop.output/'run_manifest.json').read_text())
    source='active/workflows/loop.py'
    payload=(loop.output/'task_context/source/physrsr'/source).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==manifest['source_sha256'][source]
    assert manifest['capx']['commit'] and manifest['controller_sha256'] and manifest['agents']['physic']['service_id']
    assert manifest['artifacts']['probes'][0]['response_sha256']
    assert json.loads((loop.output/'final/manifest.json').read_text())['run_manifest_sha256']


def test_configuration_or_task_preflight_error_cannot_prepare_the_world(tmp_path):
    loop,events=build(tmp_path,task_id='not_registered')
    with pytest.raises(RuntimeError,match='No predictive task adapter'):loop.run()
    assert not any(method=='target.prepare' for method,_ in events)


def test_package_resource_locator_honors_an_installed_capx_tree(tmp_path,monkeypatch):
    import physrsr.paths as paths
    from types import SimpleNamespace as NS
    installed=tmp_path/'installed';(installed/'capx').mkdir(parents=True)
    monkeypatch.delenv('PHYSRSR_CAPX_ROOT',raising=False)
    monkeypatch.setattr(paths.importlib.util,'find_spec',lambda name:NS(submodule_search_locations=[str(installed/'capx')]))
    assert paths.capx_root()==installed


def test_preparation_consumes_the_same_hard_budget_as_program_execution():
    target=InstrumentedTarget(FixtureRobot(.04))
    policy=Policy('def run(robot):\n    robot.hold(2.)\n',api_version=IDENTIFICATION_API_VERSION)
    result=target.call('target.execute_probe',dict(policy=policy.wire(),budget=dict(max_duration_s=2.)))
    assert result['status']=='program_error' and result['setup_cost_s']==1.
    assert result['cost_s']==0. and 'remaining execution duration' in result['error']


def test_unknown_setup_cost_prevents_program_execution():
    class Unknown(FixtureRobot):
        def prepare(self):
            super().prepare();self.setup_cost_s=None
    target=InstrumentedTarget(Unknown(.04))
    policy=Policy('def run(robot):\n    robot.close_gripper()\n',api_version=IDENTIFICATION_API_VERSION)
    result=target.call('target.execute_probe',dict(policy=policy.wire()))
    assert result['status']=='cost_unknown' and result['rpc_calls']==0
    assert result['program_return'] is None


def test_initial_preparation_cost_is_not_omitted_or_allowed_to_start_probe(tmp_path):
    class SetupCost(FixtureRobot):
        def prepare(self):super().prepare();self.setup_cost_s=2.
    loop,events=build(tmp_path,target=InstrumentedTarget(SetupCost(.04)),cfg=LoopConfig(max_probe_cost=2.))
    result=loop.run()
    assert result['status']=='cost_budget_exhausted' and result['probe_cost']==2.
    assert not any(method=='target.execute_probe' for method,_ in events)


def test_overlapping_setting_tolerances_cannot_double_count_one_intervention():
    context=dict(candidate_sha256='current',models=[dict(id='m',weight=1.)],max_prediction_error=1.,
        validation_requirements={'object':dict(min_duration_s=.1,min_translation_m=0.,min_rise_m=0.,min_force_change_n=0.)})
    setting=dict(object_id='object',tcp_offset_body_m=[0.,0.,0.],setting_tolerance_m=.003,
        min_duration_s=.1,min_translation_m=0.,min_rise_m=0.,min_force_change_n=0.)
    other=dict(setting,tcp_offset_body_m=[.005,0.,0.])
    contract=dict(candidate_sha256='current',object_ids=['object'],hypothesis_ids=['m'],
        alternative_explanation='Unmodeled contact',task_operation='Loaded contact',settings=[setting,other],max_prediction_error=1.)
    with pytest.raises(ValueError,match='distinguishable'):validate_contract(contract,context)


def test_each_setting_uses_its_own_pose_and_unloaded_wrench_baseline():
    packet=sensor_packet();samples=[]
    def add(event,t,object_x,tcp_x,force):
        obs=deepcopy(packet);obs['time_s']=t;obs['object']['position'][0]=object_x
        obs['tcp']['position'][0]=tcp_x;obs['wrench']['force_n'][0]=force
        samples.append(dict(event=event,observation=obs,params={}))
    add('initial',0.,0.,0.,0.);add('close_gripper',0.,0.,0.,0.)
    add('hold',1.,.1,.1,1.);add('open_gripper',1.,.1,.1,1.)
    add('close_gripper',1.,.1,.12,1.);add('hold',2.,.1,.12,1.)
    setting=dict(object_id='object',tcp_offset_body_m=[.02,0.,0.],setting_tolerance_m=.003,
        min_duration_s=.2,min_translation_m=0.,min_rise_m=0.,min_force_change_n=.5,min_object_translation_m=.05)
    assert not check_executed(dict(settings=[setting]),samples)['passed']


def test_joint_likelihood_does_not_underflow_between_balanced_segments():
    models=[Hypothesis('a',{'x':0.},.5,'fixture'),Hypothesis('b',{'x':1.},.5,'fixture')]
    data=[]
    for index in range(14):
        a_bad=index>=7
        data.append(dict(evidence_id=str(index),actual=[0.]*28,predictions={
            'a':prediction([3. if a_bad else 0.]*28),'b':prediction([0. if a_bad else 3.]*28)}))
    fitted,_=joint_fit(models,data,[1.]*28,3.)
    assert [m.weight for m in fitted]==[.5,.5]
    reverse,_=joint_fit(models,list(reversed(data)),[1.]*28,3.)
    assert fitted==reverse


def test_initialized_but_unfitted_session_cannot_generate_a_task_policy(tmp_path,monkeypatch):
    loop,_=build(tmp_path)
    loop.sim_configured=True;loop.fit_data=[dict(result={'evidence_id':'fit'})]
    monkeypatch.setattr(loop,'update_model',lambda:False)
    monkeypatch.setattr(loop,'revise_policy',lambda context:pytest.fail('No policy before a grounded commit'))
    entry={}
    result=loop.handle_policy({},dict(instance_assessments=[dict(sufficient=True)]),[],entry)
    assert result is None and entry['outcome']=='model_inadequate'


def test_simulator_replays_the_actual_program_budget_not_a_fresh_default():
    from test_unified_loop import FixtureAdapter
    source='''def run(robot):
    limit=robot.describe()['probe_execution']['max_duration_s']
    first=robot.observe()
    point=list(first['tcp']['position'])
    point[0]+=0.001 if limit<0.25 else 0.02
    robot.goto_pose(point,first['tcp']['quaternion_wxyz'],speed=0.1)
    robot.hold(0.1)
    return {'limit':limit}
'''
    policy=Policy(source,api_version=IDENTIFICATION_API_VERSION)
    actual=InstrumentedTarget(FixtureRobot(.04)).call('target.execute_probe',
        dict(policy=policy.wire(),budget=dict(max_duration_s=1.2)))
    assert actual['status']=='completed' and actual['execution_budget']['max_duration_s']<.25
    simulator=PredictiveSimulator([FixtureAdapter('translate')])
    simulator.call('sim.configure',dict(session_id='one',task=dict(id='translate'),models=[dict(
        id='m',mechanism='fixture_response_v2',parameters=dict(response=.04),weight=1.)]))
    request=dict(session_id='one',model_version=0,seeds=[1],probe=dict(policy=policy.wire()))
    wrong=simulator.call('sim.predict_probe',request)
    replayed=simulator.call('sim.predict_probe',dict(request,execution_budget=actual['execution_budget']))
    assert wrong['predictions']['m']['features']!=actual['features']
    assert replayed['predictions']['m']['features']==actual['features']


def test_wrong_real_gripper_cap_is_rejected_before_preparation(tmp_path):
    from physrsr.active.core.contracts import Task
    class WrongCap(FixtureRobot):
        def describe(self):
            caps=super().describe();caps['gripper']['normal_force_limit_per_finger_n']=20.
            return caps
    loop,events=build(tmp_path,target=InstrumentedTarget(WrongCap(.04)))
    loop.task=Task('translate','Fixture task',{},dict(grip_force_per_finger_n=2.))
    with pytest.raises(ValueError,match='force cap differs'):loop.run()
    assert not any(method=='target.prepare' for method,_ in events)


def test_common_tcp_tracking_signal_is_not_recounted_for_stationary_objects():
    first=sensor_packet();last=deepcopy(first);last['time_s']=1.;last['tcp']['position'][0]=.001
    single=[dict(observation=first),dict(observation=last)]
    scene=[]
    for sample in single:
        obs=deepcopy(sample['observation']);obs['objects']={key:deepcopy(obs['object']) for key in ['a','b','c']}
        scene.append(dict(observation=obs))
    one=response_features(single);three=response_features(scene)
    models=[Hypothesis('a',{'x':0.},.5,'fixture'),Hypothesis('b',{'x':1.},.5,'fixture')]
    a=posterior(models,[0.]*16,{'a':prediction(one),'b':prediction([0.]*16)},FEATURE_SCALES)
    b=posterior(models,[0.]*28,{'a':prediction(three),'b':prediction([0.]*28)},scene_scales(FEATURE_SCALES,3))
    assert [m.weight for m in a]==[m.weight for m in b]
