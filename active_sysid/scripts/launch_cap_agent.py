"""Explicitly launch one Codex Astra stage; never evaluate the policy here.

By default print the command only. Pass --execute after reviewing stage inputs.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

STAGES = ('privileged_exact', 'visual_exact', 'visual_randomized')


def command(context: Path, stage: str):
    idx = STAGES.index(stage)+1
    folder = context / f'stage{idx}_{stage}'
    manifest = json.loads((folder / 'input/manifest.json').read_text())
    if manifest['status'] != 'prepared_not_executed':
        raise ValueError('Unknown stage input status')
    if idx > 1:
        previous = context / f'stage{idx-1}_{STAGES[idx-2]}' / 'output/status.json'
        if json.loads(previous.read_text())['status'] != 'accepted':
            raise ValueError(f'Previous stage is not accepted: {previous}')
    prompt = (f"Read these exact files before coding: {folder/'input/robot_api.md'}, "
              f"{folder/'input/stage_specific.md'}, {folder/'input/manifest.json'}, "
              f"{folder/'input/task.md'}, {folder/'input/grounding_summary.md'}. "
              f"Stage: {stage}. Write ONLY the policy source at {manifest['policy_target']}. "
              'Do not read private World or evaluator data. Do not modify the Sim, controller, '
              'tests, or scoring thresholds. Do not run final World evaluation. '
              'Report what you wrote and what has not been evaluated.')
    return ['codex','exec','-m','gpt-6-astra','-c','model_reasoning_effort="high"',
            '--sandbox','workspace-write','-C',str(context.parents[1]),prompt]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context',type=Path,required=True)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    cmd=command(a.context.resolve(),a.stage)
    print(shlex.join(cmd))
    if a.execute:
        subprocess.run(cmd,check=True)

if __name__=='__main__':main()
