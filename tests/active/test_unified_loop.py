"""Offline integration: real RPC services/executor, supplied numerical fixtures.

No LLM, MuJoCo, task experiment or hardware runs. Fixture responses are not
claimed physical identification or benchmark results.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from physrsr.active.agents.coding import CodingService
from physrsr.active.agents.physic import PhysicService
from physrsr.active.core.contracts import Task,Hypothesis,LoopConfig,IDENTIFICATION_API_VERSION
from physrsr.active.core.measurements import FEATURE_SCHEMA
from physrsr.active.core.policy import Policy
from physrsr.active.runtime.instrumented import InstrumentedTarget
from physrsr.active.runtime.predictive import PredictiveSimulator
from physrsr.active.tasks.rod_lift import RodMeasuredAdapter
from physrsr.active.workflows.loop import ActiveLoop


def sensor_packet():
    return dict(time_s=0.,coordinate_frame="robot_base",
        object=dict(position=[0.,0.,.1],quaternion_wxyz=[1.,0.,0.,0.],axes=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
            dimensions_m=[.2,.04,.04],grasp_quaternion_wxyz=[1.,0.,0.,0.],support_clearance_m=0.),
        tcp=dict(position=[0.,0.,.1],quaternion_wxyz=[1.,0.,0.,0.]),
        wrench=dict(force_n=[0.,0.,0.],torque_nm=[0.,0.,0.],valid=True,frame="robot_base",about="tcp",provenance="Numerical test fixture"),
        joints=dict(position_rad=[0.]*7,actuator_torque_nm=[0.]*7),gripper=dict(opening_m=.04))


class FixtureRobot:
    def __init__(self,response):
        self.response,self.belief=response,dict(estimates=[])
        self.prepare()
    def prepare(self): self.state=sensor_packet(); self.setup_cost_s=0.
    def describe(self):
        return dict(api_version=IDENTIFICATION_API_VERSION,identification=deepcopy(self.belief),
            gripper=dict(normal_force_limit_per_finger_n=2.),arm=dict(joint_torque_limits_nm=[80.]*7))
    def observe(self): return deepcopy(self.state)
    def action(self,method,params):
        if method=="close_gripper": self.state["wrench"]["force_n"][0]=self.response
        if method=="open_gripper": self.state["wrench"]["force_n"][0]=0.
        if method=="hold": self.state["time_s"]+=params["duration_s"]
        if method=="goto_pose": self.state["tcp"]["position"]=params["position"]


INITIAL="def run(robot):\n    robot.close_gripper()\n    robot.hold(0.1)\n    return {'kind':'initial'}\n"
REVISED="""def run(robot):
    memory=robot.describe()['identification']['estimates']
    o=robot.observe()
    robot.close_gripper()
    robot.goto_pose(o['tcp']['position'],o['tcp']['quaternion_wxyz'],speed=0.1)
    robot.hold(0.1)
    return {'kind':'revised','estimate':memory[0]['value']}
"""
CONSERVATIVE="def run(robot):\n    robot.close_gripper()\n    robot.hold(0.1)\n    return {'kind':'conservative'}\n"
FIT="""def run(robot):
    progress=robot.describe()['probe_execution']
    state=progress['resume_state'] or {}
    if progress['remaining_rpc_calls']<20:
        robot.open_gripper()
        robot.hold(0.2)
        robot.observe()
        robot.checkpoint(state)
        return {'status':'paused','estimates':[]}
    first=robot.observe()
    robot.close_gripper()
    robot.hold(0.1)
    value=robot.observe()['wrench']['force_n'][0]
    robot.open_gripper()
    robot.goto_pose([first['tcp']['position'][0]+0.002,0.,0.1],[1.,0.,0.,0.],speed=0.1)
    robot.close_gripper()
    robot.hold(0.1)
    robot.open_gripper()
    return {'status':'identified','estimates':[{'name':'response','value':value,'interval':[value-0.001,value+0.001],
        'unit':'N','frame':'robot_base_x','identifiable':True,'assumptions':['fixture only']}],
        'embodiment_assessment':{'limits':'from measured API'}}
"""
VALIDATE="""def run(robot):
    robot.close_gripper()
    robot.hold(0.3)
    robot.observe()
    robot.open_gripper()
    return {'status':'validation'}
"""


class FixtureAdapter:
    mechanisms=("fixture_response_v2",)
    parameter_contracts={"fixture_response_v2":dict(response=dict(unit="N",frame="robot_base_x",minimum=-1.,maximum=1.))}
    def __init__(self,task_id): self.task_id=task_id
    def validate(self,task,model):
        if model.mechanism not in self.mechanisms or set(model.parameters)!={"response"}:
            raise ValueError("Fixture model mismatch")
    def execute(self,task,model,policy,seed,belief,purpose):
        target=InstrumentedTarget(FixtureRobot(model.parameters["response"]))
        target.call("target.update_identification",dict(belief=belief))
        result=target.call("target.execute_probe" if purpose=="probe" else "target.execute_policy",dict(policy=policy.wire()))
        if purpose=="probe": return dict(result,risk=0.)
        returned=result["program_return"]
        ok=(returned.get("kind")=="conservative" or
            (returned.get("kind")=="revised" and abs(returned["estimate"]-model.parameters["response"])<.002))
        return dict(status="success" if ok else "fail",success=ok,risk=0.,margin=.1 if ok else -.1,
            duration_s=result["cost_s"],observations=result["observations"],program_return=returned)


class StructuredAnswers:
    def __init__(self,conservative=False,model_value=.04):
        self.requests=[];self.conservative=conservative;self.model_value=model_value
    def generate(self,messages,destination,*,schema):
        content=messages[0]["content"]
        self.requests.append((content,schema))
        properties=schema["properties"]
        if "mode" in properties:
            return dict(mode="policy",command="",files=[dict(path="policy.py",source=INITIAL)])
        if "action" in properties:
            context=json.loads(content.split("PUBLIC CONTEXT\n")[1])
            if context["assessment"] is None:
                action="revise_policy" if context['model_version'] else "probe"
            elif not context["assessment"]["robust"]:
                action="revise_policy" if self.conservative else "probe"
            elif any(v["passed"] and v["model_version"]==context["model_version"] for v in context["validation_reports"]):
                action="stop"
            else: action="validate"
            physical=action in ('probe','validate')
            evidence=context['instances'][0]['available_evidence_ids']
            return dict(action=action,rationale="Numerical fixture decision",
                instance_assessments=[dict(episode_id='current',sufficient=not physical,reason='Fixture decision evidence',
                    evidence_ids=evidence,requirements=[dict(quantity_or_condition='Response or independent check',
                        operation_decision='Select or commit operation',met=not physical,evidence_ids=evidence)])],
                target_instance_ids=['current'] if physical else [],
                interactions=[dict(episode_id='current',uncertainty='Response or prediction ambiguity',
                    why_existing_evidence_insufficient='An independent response is unavailable',evidence_ids=evidence,
                    outcomes=[dict(condition='Prediction agrees',operation_decision='Commit candidate'),
                        dict(condition='Prediction disagrees',operation_decision='Change model and action')],
                    estimated_cost_s=1.)] if physical else [],
                expected_decision_benefit=1. if physical else 0.,estimated_probe_cost_s=1. if physical else 0.)
        if "experimental_design" in properties:
            context=json.loads(content.split("PUBLIC CONTEXT\n")[1])
            return dict(code=FIT if context["purpose"]=="fit" else VALIDATE,selected_factors=["response"],
                experimental_design=dict(manipulated_variables=["contact configuration"],minimum_distinct_settings=2),
                rationale="Numerical fixture design")
        if "models" in properties:
            context=json.loads(content.split("Context:\n")[1])
            return dict(rationale="Use actual fixture measurement",evidence_ids=[context["fit_evidence"][0]["evidence_id"]],
                models=[dict(id="measured",parameters=dict(response=self.model_value),weight=1.,mechanism="fixture_response_v2")])
        return dict(code=CONSERVATIVE if self.conservative else REVISED,rationale="Use current API memory",changes=["revise"],
            embodiment_checks=["measured limits"],remaining_uncertainties=["fixture only"])


class Recording:
    def __init__(self,service,events): self.service,self.events=service,events
    def call(self,method,params):
        self.events.append((method,deepcopy(params)))
        return self.service.call(method,params)


def build(tmp_path,task_id="translate",*,answers=None,cfg=None,target=None):
    events=[];answers=answers or StructuredAnswers()
    endpoints=dict(agent=Recording(CodingService(tmp_path/"coding",agent=answers),events),
        physic=Recording(PhysicService(tmp_path/"physic",agent=answers),events),
        simulator=Recording(PredictiveSimulator([FixtureAdapter("translate"),FixtureAdapter("reorient")]),events),
        target=Recording(target or InstrumentedTarget(FixtureRobot(.04)),events))
    models=[]  # No physical model is known at bootstrap.
    loop=ActiveLoop(Task(task_id,"Manipulate the perceived object",{},{}),cfg or LoopConfig(max_rounds=6),models,
        output=tmp_path/"run",**endpoints)
    return loop,events


@pytest.mark.parametrize("task_id",["translate","reorient"])
def test_full_v2_chain_uses_physic_updates_real_predictive_parameters_and_freezes_same_source(tmp_path,task_id):
    loop,events=build(tmp_path,task_id)
    result=loop.run()
    assert result["status"]=="task_sufficient"
    assert loop.model_version==1 and loop.models[0].parameters==dict(response=.04)
    assert loop.probe_count==2 and len(loop.fit_data)==len(loop.validation_data)==1
    methods=[m for m,p in events]
    order=["physic.assess_task","physic.design_identification","target.execute_probe",
        "physic.revise_predictive_model","sim.configure","sim.predict_probe","sim.update_model",
        "agent.synthesize_identified_policy","sim.evaluate"]
    positions=[methods.index(m) for m in order]
    assert positions==sorted(positions)
    assert "agent.generate_initial_cap" not in methods
    assert not (loop.output/"initial_cap").exists()
    first_context=next(p for m,p in events if m=="physic.design_identification")
    assert "wrench" not in first_context["current_observation"]
    assert first_context["assessment"] is None
    assert "agent.propose" not in methods and "world.evaluate_policy" not in methods and "target.execute_policy" not in methods
    assert (loop.output/"final/policy.py").read_text()==REVISED
    assert result["final_policy"]["sha256"]==Policy(REVISED).sha256
    updates=[p for m,p in events if m=="sim.update_model"]
    assert updates[0]["models"][0]["parameters"]["response"]==.04
    for method,context in events:
        if method.startswith("physic."):
            assert "task" not in context and "rod_lift" not in json.dumps(context)
    assert (loop.output/"active_probe_1/prompt.txt").exists()
    assert (loop.output/"active_probe_1/world_execution/result.json").exists()


def test_sufficient_information_revises_control_without_additional_identification(tmp_path):
    loop,events=build(tmp_path,answers=StructuredAnswers(conservative=True))
    assert loop.run()["status"]=="task_sufficient"
    assert len(loop.fit_data)==1 and loop.probe_count==2 and loop.model_version==1
    assert sum(m=="sim.update_model" for m,p in events)==1


@pytest.mark.parametrize("model_value",[-.9,-.04])
def test_model_that_cannot_replay_measured_fit_is_not_committed(tmp_path,model_value):
    loop,events=build(tmp_path,answers=StructuredAnswers(model_value=model_value),cfg=LoopConfig(max_rounds=5,max_prediction_error=.1))
    result=loop.run()
    assert result["status"]=="model_inadequate"
    assert loop.model_version==0 and result["final_policy"] is None
    assert not any(m=="sim.update_model" for m,p in events)


def test_validation_mismatch_requires_new_fit_and_cannot_count_as_sufficiency(tmp_path):
    class Changed(InstrumentedTarget):
        def call(self,method,params):
            if method=="target.execute_probe" and params["policy"]["source"]==VALIDATE:
                self.robot.response=.9
            return super().call(method,params)
    loop,events=build(tmp_path,target=Changed(FixtureRobot(.04)),
        cfg=LoopConfig(max_rounds=5,max_probes=2,max_prediction_error=.1))
    result=loop.run()
    assert result["status"]=="probe_budget_exhausted" and result["final_policy"] is None
    assert len(loop.fit_data)==1 and len(loop.validation_data)==1 and not loop.validation_data[0]["passed"]
    revisions=[p for m,p in events if m=="physic.revise_predictive_model"]
    assert len(revisions)==1 and len(revisions[0]["fit_evidence"])==1


def test_simulator_sessions_reject_stale_versions(tmp_path):
    sim=PredictiveSimulator([FixtureAdapter("translate")])
    task=Task("translate","Manipulate",{},{}).public(IDENTIFICATION_API_VERSION)
    model=dict(id="m",parameters=dict(response=.04),weight=1.,mechanism="fixture_response_v2")
    sim.call("sim.configure",dict(session_id="x",task=task,models=[model]))
    request=dict(session_id="x",model_version=0,models=[model],belief=dict(estimates=[]),fit_evidence_ids=["measured-1"])
    assert sim.call("sim.update_model",request)["model_version"]==1
    with pytest.raises(Exception,match="Stale"):
        sim.call("sim.update_model",request)


def test_fixed_mass_adapter_preserves_physical_protocol_without_loading_world():
    adapter=RodMeasuredAdapter()
    model=Hypothesis("m",dict(com_offset=-.04,friction=1.5,mass_kg=.27),1.,"rod_fixed_mass_com_v1")
    adapter.validate(dict(success=dict(lift_above_table_m=.04,max_abs_axis_z=.25,hold_steps=10),
        constraints=dict(grip_force_per_finger_n=2.)),model)
    assert set(adapter.parameter_contracts[model.mechanism])==set(model.parameters)


def test_unknown_hardware_setup_cost_preserves_measurement_without_delivery(tmp_path):
    class UnknownCost(InstrumentedTarget):
        def call(self,method,params):
            result=super().call(method,params)
            if method=="target.execute_probe": result["setup_cost_s"]=None
            return result
    loop,events=build(tmp_path,target=UnknownCost(FixtureRobot(.04)))
    result=loop.run()
    assert result["status"]=="cost_unknown" and result["final_policy"] is None
    assert result["probe_cost"]>0 and result["probe_cost_complete"] is False
    assert json.loads((loop.output/"active_probe_1/belief.json").read_text())["estimates"][0]["value"]==.04
    assert not any(m=="sim.update_model" for m,p in events)


def test_measured_cost_overrun_stops_before_model_or_policy_revision(tmp_path):
    class Costly(InstrumentedTarget):
        def call(self,method,params):
            result=super().call(method,params)
            if method=="target.execute_probe": result["setup_cost_s"]=10.
            return result
    loop,events=build(tmp_path,target=Costly(FixtureRobot(.04)),cfg=LoopConfig(max_probe_cost=2.))
    result=loop.run()
    assert result["status"]=="cost_budget_exhausted" and result["final_policy"] is None
    assert result["probe_cost"]>10.
    assert (loop.output/"active_probe_1/belief.json").exists()
    assert not any(m=="sim.update_model" for m,p in events)
