"""Agent ownership, common Probe protocol and evidence reviews; no real LLM."""
import ast
from copy import deepcopy
import json

import pytest

from physrsr.active.agents.base import validate_schema
from physrsr.active.agents.coding import CodingService
from physrsr.active.agents.physic import PhysicAgent, DESIGN_SCHEMA, DECISION_SCHEMA, validate_probe_design
from physrsr.active.api.client import APIError
from physrsr.active.core.exploration import CONTRACT
from physrsr.active.core.paths import PACKAGE_ROOT
from physrsr.active.tools.stages import author_program
from physrsr.active.core.contracts import Task
from agent_fixtures import design_answer
from test_unified_loop import FIT, VALIDATE, StructuredAnswers


def context():
    return dict(goal=dict(instruction="Manipulate the observed scene"),capabilities=dict(api="fixture"),
        initial_evidence=[dict(evidence_id="visual-1",source="numerical fixture")],
        decision_contract=CONTRACT,predictive_validation=True,world=dict(id="current"),
        instances=[dict(episode_id="current",available_evidence_ids=["task_goal","visual-1"])],
        purpose="fit",model_version=0,assessment=None,validation_reports=[])


@pytest.mark.parametrize('defect',['missing_field','fake_evidence','extra_claim','unsupported_api','missing_checkpoint','wrong_factor'])
def test_one_probe_contract_rejects_bad_schema_evidence_and_source_before_return(tmp_path,defect):
    ctx=context()
    class Backend:
        requests=[]
        def generate(self,messages,destination,*,schema):
            self.requests.append(deepcopy(messages))
            assert schema==DESIGN_SCHEMA
            answer=design_answer(ctx,FIT)
            if len(self.requests)==1:
                if defect=='missing_field': del answer['measurement_model']
                if defect=='fake_evidence': answer['observations'][0]['evidence_ids']=['not-recorded']
                if defect=='extra_claim': answer['status']='physically_verified'
                if defect=='unsupported_api': answer['code']='def run(robot):\n    robot.apply_force(10)\n'
                if defect=='missing_checkpoint': answer['code']='def run(robot):\n    robot.observe()\n'
                if defect=='wrong_factor': answer['selected_factors']=['undeclared']
            return answer
    backend=Backend();answer=PhysicAgent(tmp_path/'physic',agent=backend).design_probe(ctx)
    assert answer==design_answer(ctx,FIT) and len(backend.requests)==2
    assert 'no probe was executed' in backend.requests[1][-1]['content']


def test_manual_design_and_repair_use_the_exact_same_schema_prompt_and_rpc(tmp_path,monkeypatch):
    ctx=context();calls=[];schemas=[]
    class Backend:
        def generate(self,messages,destination,*,schema):
            schemas.append(deepcopy(schema))
            supplied=json.loads(messages[0]['content'].split('PUBLIC CONTEXT\n')[1])
            assert 'IDENTIFICATION' in messages[0]['content']
            return design_answer(supplied,FIT)
    physic=PhysicAgent(tmp_path/'physic',agent=Backend())
    class Endpoint:
        def call(self,method,params):
            calls.append(method)
            return physic.call(method,params)
    root=tmp_path/'manual'
    task=Task('fixture','Manipulate the observed scene',{}, {})
    author_program(root/'probe_1',kind='probe',task=task,context=ctx,agent=Endpoint())
    author_program(root/'probe_2',kind='probe',task=task,context=dict(ctx,design_review='source review only'),
        agent=Endpoint(),previous=root/'probe_1')
    assert calls==['physic.design_probe','physic.design_probe'] and schemas==[DESIGN_SCHEMA,DESIGN_SCHEMA]
    for index in (1,2):
        assert json.loads((root/f'probe_{index}/schema.json').read_text())==DESIGN_SCHEMA
        assert (root/f'probe_{index}/prompt.txt').read_text()==(
            physic.output/f'design_{index:04d}/prompt.txt').read_text()
    first=json.loads((root/'probe_1/manifest.json').read_text())
    second=json.loads((root/'probe_2/manifest.json').read_text())
    assert first['status']==second['status']=='authored_not_executed'
    assert first['experiment_id']==second['experiment_id'] and second['parent_sha256']==first['sha256']
    assert not (root/'probe_1/world_execution').exists()
    assert not (root/'probe_2/world_execution').exists()


def test_assessment_reassessment_and_stop_share_one_decision_contract(tmp_path):
    answers=StructuredAnswers();physic=PhysicAgent(tmp_path/'physic',agent=answers)
    ctx=context()
    assert physic.assess_task(ctx)['action']=='probe'
    ctx.update(model_version=1,assessment=dict(robust=True),validation_reports=[dict(passed=True,model_version=1)])
    assert physic.assess_task(ctx)['action']=='stop'
    assert [schema for _,schema in answers.requests]==[DECISION_SCHEMA,DECISION_SCHEMA]


def test_model_revision_rejects_validation_evidence_and_only_cites_fit_interactions(tmp_path):
    ctx=dict(parameter_contracts=dict(fixture=dict(response=dict(type='number',minimum=-1.,maximum=1.))),
        fit_evidence=[dict(evidence_id='fit-1')],validation_reports=[dict(evidence_id='heldout-1')])
    class Backend:
        requests=[]
        def generate(self,messages,destination,*,schema):
            self.requests.append(deepcopy(messages))
            assert 'heldout-1' not in messages[0]['content']
            return dict(rationale='fixture only',evidence_ids=['heldout-1' if len(self.requests)==1 else 'fit-1'],
                models=[dict(id='m',mechanism='fixture',weight=1.,parameters=dict(response=.04))])
    backend=Backend()
    answer=PhysicAgent(tmp_path/'physic',agent=backend).revise_physical_model(ctx)
    assert answer['evidence_ids']==['fit-1'] and len(backend.requests)==2


@pytest.mark.parametrize('method',['agent.propose','agent.revise_model','physic.design_probe',
    'agent.generate_initial_cap','agent.revise_cap_from_probe'])
def test_coding_service_cannot_design_probes_or_revise_physical_models(tmp_path,method):
    class NoInference:
        def generate(self,*args,**kwargs):pytest.fail('Unsupported role must be rejected before inference')
    with pytest.raises(APIError,match='Unknown coding operation'):
        CodingService(tmp_path/'coder',agent=NoInference()).call(method,{})


@pytest.mark.parametrize('method',['physic.propose_probe','physic.revise_probe','physic.design_identification',
    'physic.revise_predictive_model','agent.synthesize_policy'])
def test_removed_physic_protocols_and_coding_requests_are_rejected(tmp_path,method):
    class NoInference:
        def generate(self,*args,**kwargs):pytest.fail('Removed protocol must not trigger inference')
    with pytest.raises(APIError,match='Unknown Physic Agent operation'):
        PhysicAgent(tmp_path/'physic',agent=NoInference()).call(method,{})


def test_validation_design_uses_the_same_schema_with_a_fresh_source():
    ctx=dict(context(),purpose='validation',candidate_sha256='fixture',models=[dict(id='m',weight=1.)],
        max_prediction_error=3.,validation_requirements={'object':dict(min_duration_s=.05,min_translation_m=0.,min_rise_m=0.,min_force_change_n=0.)})
    payload=design_answer(ctx,VALIDATE)
    validate_schema(payload,DESIGN_SCHEMA)
    assert validate_probe_design(payload,ctx).source==VALIDATE


def test_coding_agent_repairs_noop_without_delegating_to_physical_reasoning(tmp_path):
    class Backend:
        calls=0
        def generate(self,messages,destination,*,schema):
            self.calls+=1
            assert 'experimental_design' not in schema['properties'] and 'models' not in schema['properties']
            return dict(code='def run(robot):\n    robot.observe()\n' if self.calls==1 else
                'def run(robot):\n    robot.close_gripper()\n',rationale='fixture',changes=['fixture'],
                embodiment_checks=['fixture'],remaining_uncertainties=['fixture only'])
    backend=Backend();coder=CodingService(tmp_path/'coding',agent=backend)
    answer=coder.synthesize_policy(context())
    assert 'close_gripper' in answer['code'] and backend.calls==2


def test_removed_author_probe_cli_does_not_invoke_any_agent(capsys):
    from physrsr.active.__main__ import main
    with pytest.raises(SystemExit) as exc: main(['author-probe'])
    assert exc.value.code==2 and 'invalid choice' in capsys.readouterr().err


def test_agents_do_not_import_workflow_or_each_others_domain_implementation():
    for name in ('physic','coding','base'):
        tree=ast.parse((PACKAGE_ROOT/'agents'/f'{name}.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert 'workflows' not in (node.module or '')
                if name=='physic': assert node.module not in ('coding','identification')
                if name=='coding': assert node.module not in ('physic','identification')
    assert not (PACKAGE_ROOT/'agents/identification.py').exists()
    assert not (PACKAGE_ROOT/'workflows/probe_authoring.py').exists()
