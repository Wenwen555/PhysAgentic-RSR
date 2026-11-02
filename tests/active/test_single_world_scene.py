"""One-loop, three-object numerical integration, without LLM/physics/hardware.

The fixture exercises the scene contracts. It is not a three-rod physics result.
"""
from copy import deepcopy
import json

import pytest

from physrsr.active.agents.coding import CodingService
from physrsr.active.agents.physic import PhysicService
from physrsr.active.core.belief import merge_identification
from physrsr.active.core.contracts import Task, LoopConfig, IDENTIFICATION_API_VERSION
from physrsr.active.core.policy import Policy
from physrsr.active.core.measurements import SCENE_FEATURE_SCHEMA, response_features
from physrsr.active.runtime.instrumented import InstrumentedTarget, observation, Dispatcher
from physrsr.active.runtime.predictive import PredictiveSimulator
from physrsr.active.runtime.robot import RobotContractError
from physrsr.active.workflows.loop import ActiveLoop
from test_unified_loop import FixtureRobot, StructuredAnswers, Recording
from agent_fixtures import design_answer


RESPONSES = dict(rod_a=.02, rod_b=.04, rod_c=.06)

FIT_SCENE = '''def run(robot):
    execution=robot.describe()['probe_execution']
    progress=execution['resume_state'] or {'next':0,'estimates':[]}
    first=robot.observe()
    estimates=progress['estimates']
    keys=sorted(first['objects'])
    for index in range(progress['next'],len(keys)):
        if robot.describe()['probe_execution']['remaining_rpc_calls']<25:
            robot.open_gripper()
            robot.hold(0.2)
            robot.observe()
            robot.checkpoint(progress)
            return {'status':'paused','estimates':estimates}
        key=keys[index]
        obj=first['objects'][key]
        robot.goto_pose(obj['position'],obj['grasp_quaternion_wxyz'],speed=0.1)
        robot.close_gripper()
        robot.hold(0.1)
        value=robot.observe()['wrench']['force_n'][0]
        robot.open_gripper()
        estimates.append({'object_id':key,'name':'response','value':value,'unit':'N',
            'frame':'robot_base_x','identifiable':True,'interval':[value-.001,value+.001]})
        progress={'next':index+1,'estimates':estimates}
    return {'status':'identified','estimates':estimates}
'''

POLICY_SCENE = '''def run(robot):
    memory=robot.describe()['identification']['objects']
    scene=robot.observe()['objects']
    completed={}
    for key in sorted(scene):
        robot.goto_pose(scene[key]['position'],scene[key]['grasp_quaternion_wxyz'],speed=0.1)
        robot.close_gripper()
        robot.hold(0.1)
        robot.open_gripper()
        completed[key]=memory[key]['estimates'][0]['value']
    return {'completed':completed}
'''

VALIDATE_SCENE = '''def run(robot):
    scene=robot.observe()['objects']
    for key in sorted(scene):
        robot.goto_pose(scene[key]['position'],scene[key]['grasp_quaternion_wxyz'],speed=0.1)
        robot.close_gripper()
        robot.hold(0.4)
        robot.observe()
        robot.open_gripper()
    return {'status':'validation'}
'''


class SceneRobot(FixtureRobot):
    def __init__(self, responses=None):
        self.responses = dict(RESPONSES if responses is None else responses)
        super().__init__(0.)

    def prepare(self):
        super().prepare()
        self.state['objects'] = {}
        for index,key in enumerate(sorted(self.responses)):
            obj = deepcopy(self.state['object'])
            obj['position'][0] = index*.15
            self.state['objects'][key] = obj
        self.state = observation(self.state)

    def describe(self):
        return dict(super().describe(), workspace=dict(min_position_m=[-.2,-.2,0.], max_position_m=[.6,.2,.3]))

    def action(self, method, params):
        super().action(method, params)
        if method == 'close_gripper':
            key = min(self.state['objects'], key=lambda k: abs(
                self.state['tcp']['position'][0]-self.state['objects'][k]['position'][0]))
            self.state['wrench']['force_n'][0] = self.responses[key]


class SceneAdapter:
    task_id = 'scene_fixture'
    mechanisms = ('fixture_scene',)
    parameter_contracts = {'fixture_scene': {key: dict(unit='N', frame='robot_base_x', minimum=-1., maximum=1.)
        for key in RESPONSES}}

    def validate(self, task, model):
        if model.mechanism != 'fixture_scene' or set(model.parameters) != set(RESPONSES):
            raise ValueError('Fixture must model the whole scene')

    def execute(self, task, model, policy, seed, belief, purpose):
        target = InstrumentedTarget(SceneRobot(model.parameters))
        target.call('target.update_identification', dict(belief=belief))
        result = target.call('target.execute_probe' if purpose == 'probe' else 'target.execute_policy',
            dict(policy=policy.wire()))
        if purpose == 'probe': return dict(result, risk=0.)
        completed = result['program_return'].get('completed', {})
        ok = set(completed)==set(RESPONSES) and all(abs(completed[k]-v)<.002 for k,v in model.parameters.items())
        return dict(status='success' if ok else 'fail', success=ok, risk=0., margin=.1 if ok else -.1,
            duration_s=result['cost_s'], observations=result['observations'], program_return=result['program_return'])


class SceneAnswers(StructuredAnswers):
    def generate(self, messages, destination, *, schema):
        properties=schema['properties']
        if 'experimental_design' in properties:
            context=json.loads(messages[0]['content'].split('PUBLIC CONTEXT\n')[1])
            return design_answer(context,FIT_SCENE if context['purpose']=='fit' else VALIDATE_SCENE)
        if 'models' in properties:
            context=json.loads(messages[0]['content'].split('Context:\n')[1])
            measured={e['object_id']:e['value'] for e in context['belief']['estimates']}
            return dict(rationale='Numerical scene fixture',
                evidence_ids=[e['evidence_id'] for e in context['fit_evidence']],
                models=[dict(id='scene',mechanism='fixture_scene',parameters=measured,weight=1.)])
        if 'code' in properties:
            return dict(code=POLICY_SCENE,rationale='Whole scene fixture',changes=['scene'],
                embodiment_checks=['fixture'],remaining_uncertainties=['not physics'])
        return super().generate(messages,destination,schema=schema)


def build_scene(tmp_path, answers=None):
    events=[];answers=answers or SceneAnswers();sim=PredictiveSimulator([SceneAdapter()])
    target=InstrumentedTarget(SceneRobot())
    loop=ActiveLoop(Task('scene_fixture','Manipulate all three perceived rods',{},{}),LoopConfig(max_rounds=6),[],
        agent=Recording(CodingService(tmp_path/'coder',agent=answers),events),
        physic=Recording(PhysicService(tmp_path/'physic',agent=answers),events),
        simulator=Recording(sim,events),target=Recording(target,events),output=tmp_path/'run')
    return loop,events,sim


def test_three_objects_share_one_world_session_policy_and_predictive_loop(tmp_path):
    loop,events,sim=build_scene(tmp_path)
    result=loop.run()
    assert result['status']=='task_sufficient'
    assert result['world_count']==1 and result['object_ids']==sorted(RESPONSES)
    assert result['feature_schema']==SCENE_FEATURE_SCHEMA
    assert len(sim.sessions)==1 and len(loop.store.policies)==1
    assert loop.models[0].parameters==RESPONSES and loop.model_version==1
    assert loop.belief['instance_id']=='current'
    for key,value in RESPONSES.items():
        assert loop.belief['objects'][key]['estimates'][0]['value']==value
    assert len(loop.fit_data)==len(loop.validation_data)==1
    assert len(loop.fit_data[0]['result']['features'])==16*3
    context=next(params for method,params in events if method=='physic.assess_task')
    assert len(context['instances'])==1 and len(context['current_observation']['objects'])==3
    assert 'wrench' not in context['current_observation']
    methods=[method for method,_ in events]
    assert methods.index('target.execute_probe')<methods.index('sim.configure')<methods.index('sim.update_model')
    assert methods.index('sim.update_model')<methods.index('agent.synthesize_policy')<methods.index('sim.evaluate')
    update=next(params for method,params in events if method=='sim.update_model')
    assert loop.validation_data[0]['evidence_id'] not in update['fit_evidence_ids']
    assert not any(method in ('world.evaluate_policy','target.execute_policy') for method in methods)


def test_partial_object_policy_is_not_a_whole_world_certificate(tmp_path):
    class PartialAnswers(SceneAnswers):
        def generate(self,messages,destination,*,schema):
            answer=super().generate(messages,destination,schema=schema)
            if 'code' in schema['properties'] and 'experimental_design' not in schema['properties']:
                answer['code']=POLICY_SCENE.replace('for key in sorted(scene):','for key in sorted(scene)[:2]:')
            return answer
    loop,_,_=build_scene(tmp_path,PartialAnswers())
    result=loop.run()
    assert result['status']!='task_sufficient' and result['final_policy'] is None
    assert not (loop.output/'final').exists()


def test_predictive_adapter_cannot_drop_an_object_from_the_scene():
    class MissingObject(SceneAdapter):
        def execute(self,*args):
            result=super().execute(*args)
            for sample in result['observations']: sample['observation']['objects'].pop('rod_c')
            return result
    sim=PredictiveSimulator([MissingObject()])
    sim.call('sim.configure',dict(session_id='world',object_ids=sorted(RESPONSES),
        task=dict(id='scene_fixture',success={},constraints={}),
        models=[dict(id='scene',mechanism='fixture_scene',parameters=RESPONSES,weight=1.)]))
    with pytest.raises(ValueError,match='identities changed'):
        sim.call('sim.predict_probe',dict(session_id='world',model_version=0,seeds=[1],
            probe=dict(policy=Policy(VALIDATE_SCENE,api_version=IDENTIFICATION_API_VERSION).wire())))


def test_scene_projection_preserves_nonfirst_objects_and_rejects_track_substitution():
    first=SceneRobot().observe();last=deepcopy(first)
    last['objects']['rod_c']['position'][0]+=.01
    features=response_features([dict(observation=first),dict(observation=last)])
    assert features[0]==0. and features[32]==pytest.approx(.01)
    last['objects']['replacement']=last['objects'].pop('rod_c')
    with pytest.raises(ValueError,match='identities changed'):
        response_features([dict(observation=first),dict(observation=last)])


@pytest.mark.parametrize('object_id', [None,'unseen_rod'])
def test_scene_belief_cannot_pool_or_misattribute_object_estimates(object_id):
    previous=dict(estimates=[],instance_id='current',object_ids=sorted(RESPONSES))
    estimate=dict(name='mass',value=1.,identifiable=True)
    if object_id is not None: estimate['object_id']=object_id
    with pytest.raises(ValueError,match='known object_id'):
        merge_identification(previous,dict(estimates=[estimate]),source_probe='p',evidence_id='e')


def test_scene_observation_whitelists_geometry_and_recovery_checks_all_objects():
    robot=SceneRobot();raw=robot.observe()
    raw['objects']['rod_b']['private_mass']=9.
    observed=observation(raw)
    assert 'private_mass' not in observed['objects']['rod_b']
    dispatcher=Dispatcher(robot,robot.observe())
    robot.state['objects']['rod_c']['support_clearance_m']=.02
    with pytest.raises(RobotContractError,match='support'):
        dispatcher('open_gripper',{})
