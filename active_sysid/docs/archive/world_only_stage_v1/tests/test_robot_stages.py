"""Reusable M3-R stage harness and reference baseline contracts."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from active_sysid.robot.stage_freeze import verify_manifest, sha256
from active_sysid.robot.stage_randomization import apply_world_randomization
from active_sysid.robot.stages import STAGES, load_stage, run_stage
from active_sysid.robot.world_backend import WorldRobotBackend
from active_sysid.robot.world_runtime import WorldRobotRuntime


def test_frozen_native_world_manifest():
    frozen = verify_manifest()
    assert frozen['baseline_id'] == 'unidoor_r1_native_robot_v1'
    assert len(frozen['files_sha256']) >= 120
    assert frozen['baseline_reference']['bilateral_manipulation_contact_ratio'] == 1
    assert frozen['baseline_reference']['final_door_angle_rad'] >= 0.30


def test_stage_configs_preserve_equal_exact_worlds():
    assert STAGES == ('privileged_exact', 'visual_exact', 'visual_randomized')
    privileged, visual, randomized = [load_stage(stage) for stage in STAGES]
    assert privileged['randomization'] == visual['randomization'] == {}
    assert privileged['seeds'] == visual['seeds'] == [0]
    assert privileged['world'] == visual['world'] == randomized['world'] == 'reference_world'
    assert privileged['mechanism'] == visual['mechanism'] == randomized['mechanism'] == 'lever'
    assert privileged['observations'] == 'privileged'
    assert visual['observations'] == randomized['observations'] == 'visual'
    assert visual['visual_cameras'] == randomized['visual_cameras']
    assert visual['max_steps'] == randomized['max_steps'] == privileged['max_steps']
    assert randomized['randomization'] and len(randomized['seeds']) > 1


def test_visual_policy_observations_are_robot_only():
    b = WorldRobotBackend()
    api = WorldRobotRuntime(b, observations='visual')
    observation = api.observe()
    assert 'privileged' not in observation
    assert 'rgb' not in observation
    assert len(observation['q']) == 7
    assert len(observation['wrist_wrench_ee']) == 6
    assert 'door_angle_rad' not in observation
    with pytest.raises(ValueError, match='visual stage camera not allowed'):
        api.observe(rgb=True, camera='closeup')
    api.close()


def test_seeded_randomization_does_not_edit_assets_or_robot():
    cfg = load_stage('visual_randomized')
    a = WorldRobotBackend()
    source_sha = sha256(a.scene)
    robot_actuator_gains = a.model.actuator_gainprm.copy()
    nominal_handle_stiffness = float(a.model.jnt_stiffness[a.handle_jid])
    b = WorldRobotBackend()
    c = WorldRobotBackend()
    draw_a = apply_world_randomization(a, 7, cfg['randomization'])
    draw_b = apply_world_randomization(b, 7, cfg['randomization'])
    draw_c = apply_world_randomization(c, 3, cfg['randomization'])
    assert draw_a == draw_b
    assert draw_a != draw_c
    assert np.allclose(a.model.dof_damping, b.model.dof_damping)
    assert not np.allclose(a.model.dof_damping, c.model.dof_damping)
    assert np.array_equal(a.model.actuator_gainprm, robot_actuator_gains)
    assert a.model.jnt_stiffness[a.handle_jid] == pytest.approx(
        nominal_handle_stiffness * draw_a['handle_stiffness'])
    assert sha256(a.scene) == source_sha
    assert a.model.neq == b.model.neq == c.model.neq == 2


def test_stage_runner_can_evaluate_new_policy_entry_without_codegen(tmp_path):
    policy = tmp_path / 'minimal_observation_probe.py'
    policy.write_text('''def run(api):
    observation = api.observe()
    assert "privileged" not in observation
    assert "door_angle_rad" not in observation
    api.hold(0.02)
    return {"observation_checked": True}
''')
    result = run_stage('visual_randomized', policy, tmp_path / 'run', seeds=[3, 7])
    assert result['total_episodes'] == 2
    assert result['successful_episodes'] == 0  # No CaP was developed.
    assert result['observations'] == 'visual'
    assert [e['seed'] for e in result['episodes']] == [3, 7]
    draws = []
    for entry in result['episodes']:
        metrics = json.loads(Path(entry['metrics_path']).read_text())
        assert metrics['execution_error'] is None
        assert not metrics['success']
        assert metrics['gates']['rgb_feedback_observed'] is False
        assert metrics['provenance']['stage_name'] == 'visual_randomized'
        assert metrics['provenance']['policy_sha256'] == sha256(policy)
        draws.append(metrics['provenance']['parameter_draw_evaluator_private'])
    assert draws[0] != draws[1]
    assert json.loads((tmp_path / 'run/summary.json').read_text())['success_rate'] == 0


def test_run_all_reuses_exact_visual_policy_by_default(tmp_path):
    from active_sysid.robot.stages import run_all
    privileged = tmp_path / 'privileged.py'
    visual = tmp_path / 'visual.py'
    privileged.write_text('''def run(api):
    assert "privileged" in api.observe()
    api.hold(0.02)
''')
    visual.write_text('''def run(api):
    assert "privileged" not in api.observe()
    api.hold(0.02)
''')
    result = run_all(privileged, visual, tmp_path / 'all', randomized_seeds=[2, 5])
    assert result['visual_policy_unchanged_in_randomized']
    assert list(result['stages']) == ['privileged_exact', 'visual_exact', 'visual_randomized']
    assert result['stages']['privileged_exact']['total_episodes'] == 1
    assert result['stages']['visual_exact']['total_episodes'] == 1
    assert result['stages']['visual_randomized']['total_episodes'] == 2
    assert result['stages']['visual_exact']['policy_sha256'] == result['stages']['visual_randomized']['policy_sha256']
    assert (tmp_path / 'all/comparison.json').is_file()


def test_frozen_manifest_rejects_modified_hash(monkeypatch):
    import active_sysid.robot.stage_freeze as freeze
    real_sha256 = freeze.sha256
    sentinel = freeze.ROOT / 'src/active_sysid/robot/controller.py'
    monkeypatch.setattr(freeze, 'sha256', lambda path: '0' * 64 if path == sentinel else real_sha256(path))
    with pytest.raises(RuntimeError, match='controller.py'):
        freeze.verify_manifest()
