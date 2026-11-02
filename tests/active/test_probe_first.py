"""Cold-start readiness and per-instance evidence retention, no physical rollout."""
import copy
import json
import pytest

from physrsr.active.runtime.instrumented import settle_ready_state
from physrsr.active.workflows.identification import instance_memory, merge_identification
from physrsr.active.workflows.batch import complete_cases


def test_ready_waits_for_gripper_motion_instead_of_only_static_object_pose():
    class Robot:
        setup_cost_s=0.
        def __init__(self):self.time=0.;self.opening=0.;self.steps=0
        def observe(self):
            pose=dict(position=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.])
            return dict(time_s=self.time,object=copy.deepcopy(pose),tcp=copy.deepcopy(pose),
                gripper=dict(opening_m=self.opening))
        def action(self,method,params):
            assert method=="hold"
            self.time+=params["duration_s"];self.steps+=1
            self.opening=min(.08,self.opening+.02)
    robot=Robot();ready=settle_ready_state(robot)
    assert ready["gripper"]["opening_m"]==.08
    assert robot.steps>=6 and robot.setup_cost_s>=3.


def test_failed_new_probe_retains_only_the_same_instances_valid_evidence(tmp_path):
    for index,values in [(1,[("a",.03),("b",-.07)]),(2,[("a",None),("b",None)])]:
        for episode,value in values:
            folder=tmp_path/f"active_probe_{index}/world_execution"/episode
            folder.mkdir(parents=True)
            estimates=[] if value is None else [dict(name="response",value=value,unit="N",frame="base_x",identifiable=True)]
            (folder/"result.json").write_text(json.dumps(dict(evidence_id=f"{episode}-{index}",
                program_return=dict(status="recovery_failed",estimates=estimates))))
    memory=instance_memory(tmp_path,"a",3)
    assert [e["value"] for e in memory["estimates"]]==[.03]
    assert memory["estimates"][0]["source_evidence_id"]=="a-1"
    assert memory["status"]=="recovery_failed"


def test_echoing_measured_history_does_not_create_a_new_measurement():
    estimate=dict(name="response",value=.03,unit="N",frame="base_x",identifiable=True,
        source_probe="active_probe_1",source_evidence_id="measured-1")
    previous=dict(estimates=[estimate])
    merged=merge_identification(previous,dict(status="insufficient_evidence",estimates=[estimate]),
        source_probe="active_probe_2",evidence_id="measured-2")
    assert merged["estimates"]==[estimate]
    assert merged["estimates"][0]["source_evidence_id"]=="measured-1"


def test_batch_acceptance_requires_every_distinct_instance():
    cases=[dict(episode_id=e) for e in ["a","b","c"]]
    assert not complete_cases([dict(episode_id="a",status="success")],cases)
    assert not complete_cases([dict(episode_id=e,status="success") for e in ["a","a","c"]],cases)
    assert complete_cases([dict(episode_id=e,status="success") for e in ["c","a","b"]],cases)


@pytest.mark.parametrize('kind',['program','infrastructure'])
def test_resume_keeps_completed_error_and_only_program_errors_reach_revision(tmp_path,monkeypatch,kind):
    from physrsr.active.workflows import identification as workflow
    from physrsr.active.core.artifacts import save_json
    from physrsr.active.agents.identification import source_policy
    policy=source_policy('def run(robot):\n    robot.close_gripper()\n')
    stage=tmp_path/'identified_cap_1';stage.mkdir()
    (stage/'policy.py').write_text(policy.source)
    save_json(stage/'manifest.json',dict(sha256=policy.sha256,probe_round=1))
    save_json(tmp_path/'evaluator_only_cases.json',[dict(episode_id='a')])
    save_json(tmp_path/'active_probe_1/world_execution/a/belief.json',dict(estimates=[]))
    save_json(stage/'evaluation/a/result.json',dict(policy_sha256=policy.sha256,error_kind=kind,
        record=dict(status='error',reason='retained original error'),program_return=None))
    monkeypatch.setattr(workflow,'world_endpoint',lambda *args:pytest.fail('completed result must not rerun'))
    if kind=='infrastructure':
        with pytest.raises(RuntimeError,match='infrastructure'):
            workflow.evaluate_identified_policy(tmp_path,resume=True)
    else:
        rows=workflow.evaluate_identified_policy(tmp_path,resume=True)
        assert rows[0]['status']=='error' and rows[0]['error_kind']=='program'
