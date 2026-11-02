"""Typed contracts, bounded repair, persistence injection and model version errors."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from physrsr.active.agents.coding import CodingService
from physrsr.active.agents.physic import PhysicAgent
from physrsr.active.api.client import APIError, ModelVersionError, RecordedCalls
from physrsr.active.core.agent_models import ProbeDesign, TaskDecision, physical_revision_model
from physrsr.active.core.contracts import LoopConfig, Task
from physrsr.active.runtime.instrumented import InstrumentedTarget
from physrsr.active.runtime.predictive import PredictiveSimulator
from physrsr.active.workflows.loop import ActiveLoop
from agent_fixtures import design_answer
from test_agent_contracts import context
from test_unified_loop import FIT, FixtureAdapter, FixtureRobot, REVISED, StructuredAnswers, build


class MemoryRecorder:
    """Test implementation of Recorder: paths are labels, never filesystem I/O."""
    def __init__(self, root, records=None):
        self.root=Path(root)
        self.records=records if records is not None else {}

    def scope(self, name, *, exclusive=False):
        return MemoryRecorder(self.root/name,self.records)

    def write_json(self, name, value):self.records[str(self.root/name)]=deepcopy(value)
    def read_json(self, name):return deepcopy(self.records[str(self.root/name)])
    def write_text(self, name, value):self.records[str(self.root/name)]=value
    def read_text(self, name):return self.records[str(self.root/name)]
    def event(self, kind, **value):
        self.records.setdefault(str(self.root/'events.jsonl'),[]).append(deepcopy(dict(event=kind,**value)))


def test_malformed_generated_json_is_repaired_with_error_and_archived(tmp_path):
    class Backend:
        requests=[]
        def generate(self,messages,destination,*,schema):
            self.requests.append(deepcopy(messages))
            if len(self.requests)==1:json.loads('{"observations":')
            return design_answer(context(),FIT)
    backend=Backend()
    assert PhysicAgent(tmp_path/'agent',agent=backend).design_probe(context())['code']==FIT
    assert len(backend.requests)==2
    correction=backend.requests[1][-1]['content']
    assert 'Malformed generated JSON' in correction and 'no probe was executed' in correction
    errors=json.loads((tmp_path/'agent/design_0001/validation_errors.json').read_text())
    assert len(errors)==1 and errors[0]['attempt']==1


def test_exhausted_repair_retains_last_failure_and_all_attempts(tmp_path):
    class Backend:
        calls=0
        def generate(self,*args,**kwargs):
            self.calls+=1
            answer=design_answer(context(),FIT)
            answer['selected_factors']=['invented']
            return answer
    backend=Backend()
    with pytest.raises(APIError,match='three attempts; last validation error:.*') as exc:
        PhysicAgent(tmp_path/'agent',agent=backend).design_probe(context())
    assert backend.calls==3 and 'Selected factors' in str(exc.value)
    assert isinstance(exc.value.__cause__,ValidationError)
    errors=json.loads((tmp_path/'agent/design_0001/validation_errors.json').read_text())
    assert [row['attempt'] for row in errors]==[1,2,3]
    assert not (tmp_path/'agent/design_0001/response.json').exists()


@pytest.mark.parametrize('error',[APIError('transport failed'),RuntimeError('backend unavailable'),ValueError('backend configuration')])
def test_backend_failures_are_not_misclassified_as_invalid_model_json(tmp_path,error):
    class Backend:
        calls=0
        def generate(self,*args,**kwargs):
            self.calls+=1
            raise error
    backend=Backend()
    with pytest.raises(type(error),match=str(error)):
        PhysicAgent(tmp_path/'agent',agent=backend).design_probe(context())
    assert backend.calls==1


@pytest.mark.parametrize('value',[0.,2.1,True,'1',float('nan')])
def test_adapter_parameter_bounds_and_numeric_types_are_enforced(value):
    model=physical_revision_model({'fixture':{'mass':{'exclusiveMinimum':0.,'maximum':2.}}})
    answer=dict(rationale='fixture',evidence_ids=['fit-1'],models=[dict(id='a',mechanism='fixture',
        weight=1.,parameters=dict(mass=value))])
    with pytest.raises(ValidationError):
        model.model_validate(answer,context={'fit_evidence':[{'evidence_id':'fit-1'}]})


def test_typed_models_enforce_semantics_without_agent_or_jsonschema():
    ctx=context();answer=design_answer(ctx,FIT)
    answer['observations'][0]['evidence_ids']=['invented']
    with pytest.raises(ValidationError,match='unavailable evidence'):
        ProbeDesign.model_validate(answer,context=ctx)
    decision=StructuredAnswers().generate([{'content':'PUBLIC CONTEXT\n'+json.dumps(ctx)}],None,
        schema=TaskDecision.model_json_schema())
    decision['interactions'][0]['outcomes'][1]['operation_decision']=decision['interactions'][0]['outcomes'][0]['operation_decision']
    with pytest.raises(ValidationError,match='Indistinguishable'):
        TaskDecision.model_validate(decision,context=ctx)


def test_model_revision_repairs_parameter_bound_using_the_same_retry_entry(tmp_path):
    ctx=dict(parameter_contracts={'fixture':{'mass':{'exclusiveMinimum':0.,'maximum':2.}}},
        fit_evidence=[{'evidence_id':'fit-1'}])
    class Backend:
        requests=[]
        def generate(self,messages,destination,*,schema):
            self.requests.append(deepcopy(messages))
            return dict(rationale='fixture',evidence_ids=['fit-1'],models=[dict(id='a',mechanism='fixture',
                weight=1.,parameters=dict(mass=0. if len(self.requests)==1 else .1))])
    backend=Backend()
    answer=PhysicAgent(tmp_path/'agent',agent=backend).revise_physical_model(ctx)
    assert answer['models'][0]['parameters']['mass']==.1 and len(backend.requests)==2
    assert '0.0' in backend.requests[1][-1]['content']


def test_dynamic_models_keep_each_mechanisms_parameter_contract_separate():
    model=physical_revision_model({'mass':{'mass':{'exclusiveMinimum':0.}},
        'contact':{'friction':{'exclusiveMinimum':0.,'maximum':3.}}})
    answer=dict(rationale='fixture',evidence_ids=['fit-1'],models=[
        dict(id='m',mechanism='mass',weight=.5,parameters={'mass':.1}),
        dict(id='c',mechanism='contact',weight=.5,parameters={'friction':.7})])
    assert len(model.model_validate(answer,context={'fit_evidence':[{'evidence_id':'fit-1'}]}).model_dump()['models'])==2
    answer['models'][1]['parameters']={'mass':.1}
    with pytest.raises(ValidationError):
        model.model_validate(answer,context={'fit_evidence':[{'evidence_id':'fit-1'}]})


def test_single_loop_can_complete_and_freeze_with_no_artifact_filesystem(tmp_path):
    output=tmp_path/'virtual';recorder=MemoryRecorder(output)
    answers=StructuredAnswers()
    loop=ActiveLoop(Task('fixture','Manipulate the scene',{},{}),LoopConfig(max_rounds=6),[],
        agent=CodingService(output/'coding',agent=answers,recorder=recorder.scope('coding')),
        physic=PhysicAgent(output/'physic',agent=answers,recorder=recorder.scope('physic')),
        simulator=PredictiveSimulator([FixtureAdapter('fixture')]),
        target=InstrumentedTarget(FixtureRobot(.04)),output=output,recorder=recorder)
    result=loop.run()
    assert result['status']=='task_sufficient' and not output.exists()
    assert recorder.read_text('final/policy.py')==REVISED
    assert recorder.read_json('result.json')==result
    designs=[value for name,value in recorder.records.items()
        if '/physic/design_' in name and name.endswith('/effective_context.json')]
    assert [ctx['purpose'] for ctx in designs]==['fit','validation']
    assert recorder.records[str(output/'events.jsonl')][-1]['event']=='loop_stopped'


@pytest.mark.parametrize('received',[None,True,0,2])
def test_version_mismatch_has_session_method_expected_and_actual_without_resync(tmp_path,received):
    recorder=MemoryRecorder(tmp_path/'virtual');calls=RecordedCalls(recorder)
    with pytest.raises(ModelVersionError) as exc:
        calls.checked_version({'model_version':received},session_id='one-world',method='sim.evaluate',expected=1)
    assert exc.value.session_id=='one-world' and exc.value.expected==1 and exc.value.received==received
    assert 'sim.evaluate' in str(exc.value)
    assert recorder.records[str(recorder.root/'events.jsonl')][-1]['event']=='model_version_mismatch'


def test_failed_rpc_is_recorded_and_never_repeated(tmp_path):
    recorder=MemoryRecorder(tmp_path);calls=RecordedCalls(recorder)
    class Uncertain:
        attempts=0
        def call(self,method,params):
            self.attempts+=1
            raise APIError('acknowledgement lost')
    target=Uncertain()
    with pytest.raises(APIError,match='acknowledgement lost'):
        calls.call(target,'target.execute_probe',{})
    assert target.attempts==1
    assert [row['event'] for row in recorder.records[str(tmp_path/'events.jsonl')]]==['api_request','api_error']


@pytest.mark.parametrize('method',['sim.update_model','sim.evaluate'])
def test_unified_loop_stops_on_wrong_version_acknowledgement_before_certification(tmp_path,method):
    loop,events=build(tmp_path)
    service=loop.simulator.service
    original=service.call
    def wrong_ack(name,params):
        result=original(name,params)
        if name==method:result['model_version']=999
        return result
    service.call=wrong_ack
    with pytest.raises(ModelVersionError) as exc:loop.run()
    assert exc.value.session_id==loop.session_id and exc.value.method==method
    assert exc.value.expected==1 and exc.value.received==999
    assert sum(name==method for name,_ in events)==1
    assert json.loads((loop.output/'result.json').read_text())['status']=='infrastructure_error'
    assert not (loop.output/'final').exists()
