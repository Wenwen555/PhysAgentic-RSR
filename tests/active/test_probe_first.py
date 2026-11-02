"""Cold-start readiness and per-instance evidence retention, no physical rollout."""
import copy

from physrsr.active.runtime.instrumented import settle_ready_state
from physrsr.active.core.belief import merge_identification


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




def test_echoing_measured_history_does_not_create_a_new_measurement():
    estimate=dict(name="response",value=.03,unit="N",frame="base_x",identifiable=True,
        source_probe="active_probe_1",source_evidence_id="measured-1",interval=[.02,.04],assumptions=["fixture only"])
    previous=dict(estimates=[estimate])
    merged=merge_identification(previous,dict(status="insufficient_evidence",estimates=[estimate]),
        source_probe="active_probe_2",evidence_id="measured-2")
    assert merged["estimates"]==[estimate]
    assert merged["estimates"][0]["source_evidence_id"]=="measured-1"
