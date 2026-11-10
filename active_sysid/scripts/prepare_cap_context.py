"""Prepare three immutable, auditable CaP context packages; never run Codex or Sim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from active_sysid.robot.fixture_backend import ROOT
from active_sysid.robot.stage_freeze import MANIFEST, verify_manifest

CAP = ROOT / 'cap_development'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path) -> dict:
    frozen = verify_manifest()
    cfg = yaml.safe_load((CAP / 'pipeline.yaml').read_text())
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Context directory already exists: {output}')
    output.mkdir(parents=True)
    from active_sysid.robot.stages import load_stage
    stages = cfg['runtime']['stage_order']
    for i, stage in enumerate(stages):
        spec = cfg['stages'][stage]
        stage_cfg = load_stage(stage)
        assert stage_cfg['sim_variant'] == cfg['runtime']['simulator']
        assert stage_cfg['observations'] == spec['observations']
        folder = output / f'stage{i+1}_{stage}'
        inp = folder / 'input'
        inp.mkdir(parents=True)
        (folder / 'output').mkdir()
        (inp / 'task.md').write_text(
            'Open the lever door in the task-grounded Robot-Space Sim to >=0.30 rad. '
            'Use only the allowed robot actions and observations. World is held out.\n')
        (inp / 'robot_api.md').write_text((CAP / 'prompts/shared.md').read_text())
        (inp / 'stage_specific.md').write_text((CAP / 'prompts' / f'{stage}.md').read_text())
        (inp / 'grounding_summary.md').write_text(
            'M0: robot contact feasible. M1: robot probing supported a rotate-before-pull task condition. '
            'M2: task-relevant pull response calibrated. This is evidence for task planning, '
            'not a reconstruction of the native World latch. '
            'Reference fixed program was transferred once; agent-generated CaP remains untested.\n')
        predecessor = stages[i-1] if i else None
        inputs = ['input/task.md', 'input/robot_api.md', 'input/grounding_summary.md',
                  'input/stage_specific.md', 'input/manifest.json']
        if predecessor:
            (inp / 'previous_stage.md').write_text(
                f'Previous stage: {predecessor}. Before writing this stage, read its accepted '
                'policy, run summary, and SHA256 from the previous stage output. '
                'Do not invent these artifacts; stop if they do not exist.\n')
            inputs.append('input/previous_stage.md')
        manifest = {
            'stage': stage, 'status': 'prepared_not_executed',
            'coding_agent': cfg['coding_agent'],
            'sim_variant': cfg['runtime']['simulator'],
            'role': 'development_sim',
            'observations': spec['observations'],
            'allowed_cameras': cfg['runtime']['allowed_cameras'],
            'task_goal_door_angle_rad': cfg['runtime']['task_goal_door_angle_rad'],
            'predecessor': predecessor,
            'predecessor_required_status': 'accepted' if predecessor else None,
            'policy_target': str((ROOT / spec['output_policy']).resolve()),
            'development_seeds': spec['development_seeds'],
            'acceptance_seeds': spec.get('acceptance_seeds', []),
            'frozen_execution_id': frozen['baseline_id'],
            'frozen_execution_sha256': sha(MANIFEST),
            'input_files': inputs,
            'input_sha256': {name: sha(folder / name) for name in inputs if name != 'input/manifest.json'},
            'output_contract': ['policy.py', 'agent_trace.jsonl', 'development_summary.json',
                                'acceptance_summary.json', 'handoff.md'],
            'outputs_generated': False,
        }
        (inp / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
        status = json.loads((CAP / 'schemas/stage_output.json').read_text())
        status['stage'] = stage
        status['policy_path'] = manifest['policy_target']
        (folder / 'output/status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False)+'\n')
    summary = {'status': 'prepared_not_executed', 'model': cfg['coding_agent']['model'],
               'frozen_execution_id': frozen['baseline_id'],
               'stages': [f'stage{i+1}_{stage}' for i,stage in enumerate(stages)],
               'next_action': 'Start Stage 1 in a new Codex session; do not skip stage acceptance.'}
    (output / 'pipeline.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out),indent=2))
