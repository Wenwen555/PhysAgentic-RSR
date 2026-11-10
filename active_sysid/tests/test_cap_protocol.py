"""Revision/freeze/independent-seed protocol on the robot-space Sim runner."""
from pathlib import Path
import json
import pytest

from active_sysid.robot.cap_protocol import develop, freeze, accept


def test_robot_cap_revision_freeze_acceptance(tmp_path):
    cfg=tmp_path/'protocol.yaml'
    cfg.write_text('''stage: visual_randomized
sim_variant: robot_m2
revisions_max: 2
development_seeds: [12001, 12002]
independent_acceptance_seeds: [91201, 91202]
''')
    policy=tmp_path/'cap.py'
    policy.write_text('''def run(api):
    s=api.observe(rgb=True,camera='overview')
    api.hold(.04)
    api.observe(rgb=True,camera='robot0_eye_in_hand')
''')
    experiment=tmp_path/'experiment'
    development=develop(policy, 0, experiment, config=cfg)
    assert development['development_success_rate']==0
    with pytest.raises(FileExistsError):
        develop(policy,0,experiment,config=cfg)
    frozen=freeze(experiment,0,config=cfg)
    assert frozen['policy_sha256']==development['policy_sha256']
    with pytest.raises(ValueError,match='already frozen'):
        develop(policy,1,experiment,config=cfg)
    acceptance=accept(experiment,config=cfg)
    assert acceptance['independent_acceptance']
    assert acceptance['accepted_seed_count']==2
    assert acceptance['policy_sha256']==frozen['policy_sha256']
    assert json.loads((experiment/'acceptance/summary.json').read_text())['world_evaluations_performed']==0
    with pytest.raises(FileExistsError):
        accept(experiment,config=cfg)
