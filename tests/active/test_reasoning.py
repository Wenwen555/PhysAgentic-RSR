"""Decision-value rules retained independently of any retired controller."""
import pytest

from physrsr.active.core.contracts import Hypothesis, LoopConfig
from physrsr.active.core.measurements import FEATURE_SCALES
from physrsr.active.core.policy import Policy
from physrsr.active.core.reasoning import assess, posterior, probe_value, Probe


MODELS=(Hypothesis('left',dict(response=-.04),.5),Hypothesis('right',dict(response=.04),.5))
SLOW=Policy("def run(robot):\n    return 'slow'\n")
FAST=Policy("def run(robot):\n    return 'fast'\n")
PROBE=Probe('contrast',Policy("def run(robot):\n    return 'probe'\n"),1.,('response',),'contrast')
PREDICTIONS={model.id:dict(features=[model.parameters['response'],*([0.]*15)],risk=0.) for model in MODELS}


def test_parameter_information_without_operation_value_does_not_justify_probe():
    rows=[dict(model_id=model.id,policy_sha256=policy.sha256,success_rate=1.,risk=0.,duration_s=10.,margin=.1)
        for model in MODELS for policy in (SLOW,FAST)]
    cfg=LoopConfig(feature_scales=FEATURE_SCALES)
    value=probe_value(PROBE,PREDICTIONS,MODELS,[SLOW,FAST],rows,cfg)
    assert value['expected_decision_value']==pytest.approx(0.) and value['net_value']<0
    fitted=posterior(MODELS,[-.04,*([0.]*15)],PREDICTIONS,cfg.feature_scales)
    assert fitted[0].weight>.99


def test_faster_operation_can_justify_probe_even_with_a_robust_slow_candidate():
    rows=[dict(model_id=model.id,policy_sha256=policy.sha256,
        success_rate=1. if policy==SLOW or model.id=='left' else 0.,risk=0.,margin=.1,
        duration_s=100. if policy==SLOW else 1.) for model in MODELS for policy in (SLOW,FAST)]
    cfg=LoopConfig(time_weight=.01,feature_scales=FEATURE_SCALES)
    assert assess(MODELS,[SLOW,FAST],rows,cfg)['selected']==SLOW.sha256
    assert probe_value(PROBE,PREDICTIONS,MODELS,[SLOW,FAST],rows,cfg)['net_value']>.4
