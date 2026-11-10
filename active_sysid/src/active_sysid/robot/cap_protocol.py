"""Reuse Stage 3's revision/holdout protocol with the current Robot-Space runner.

Three commands: develop a numbered revision, freeze one tested revision, then
accept on independent seeds. This module does not generate or alter CaP code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .fixture_backend import ROOT
from .stage_freeze import sha256
from .stages import run_stage
from .sim2sim import freeze_policy, write_json

DEFAULT_CONFIG = ROOT / 'configs/robot/cap_protocol.yaml'


def config_from(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg['development_seeds']) & set(cfg['independent_acceptance_seeds']):
        raise ValueError('Development and acceptance seeds must not overlap')
    return cfg


def develop(policy, revision, experiment, *, config=DEFAULT_CONFIG):
    cfg = config_from(config)
    if not 0 <= revision < cfg['revisions_max']:
        raise ValueError('Revision exceeds the predeclared budget')
    experiment = Path(experiment)
    if (experiment / 'frozen').exists():
        raise ValueError('Experiment already frozen; start a new experiment')
    output = experiment / 'revisions' / f'rev_{revision:02d}'
    result = run_stage(cfg['stage'], policy, output,
                       seeds=cfg['development_seeds'], sim_variant=cfg['sim_variant'])
    return {'revision': revision, 'stage': cfg['stage'],
            'policy_sha256': result['policy_sha256'],
            'development_success_rate': result['success_rate'],
            'summary_path': str((output / 'summary.json').resolve())}


def freeze(experiment, revision, *, config=DEFAULT_CONFIG):
    cfg = config_from(config)
    if not 0 <= revision < cfg['revisions_max']:
        raise ValueError('Revision exceeds the predeclared budget')
    experiment = Path(experiment)
    output = experiment / 'revisions' / f'rev_{revision:02d}'
    manifest = freeze_policy(output / 'policy.py', output / 'summary.json', experiment / 'frozen')
    manifest['revision'] = revision
    manifest['protocol_sha256'] = sha256(config)
    manifest['acceptance_seeds'] = cfg['independent_acceptance_seeds']
    write_json(experiment / 'frozen' / 'manifest.json', manifest)
    return manifest


def accept(experiment, *, config=DEFAULT_CONFIG):
    cfg = config_from(config)
    experiment = Path(experiment)
    frozen = json.loads((experiment / 'frozen' / 'manifest.json').read_text())
    if frozen['protocol_sha256'] != sha256(config):
        raise ValueError('Experiment protocol changed after freeze')
    policy = experiment / 'frozen' / 'policy.py'
    if sha256(policy) != frozen['policy_sha256']:
        raise ValueError('Policy changed after freeze')
    report = run_stage(cfg['stage'], policy, experiment / 'acceptance',
                       seeds=cfg['independent_acceptance_seeds'],
                       sim_variant=cfg['sim_variant'])
    result = {'stage': cfg['stage'], 'sim_variant': cfg['sim_variant'],
              'policy_sha256': frozen['policy_sha256'],
              'independent_acceptance': True,
              'development_and_acceptance_seeds_disjoint': True,
              'accepted_seed_count': report['total_episodes'],
              'successful_episodes': report['successful_episodes'],
              'success_rate': report['success_rate'],
              'acceptance_summary': str((experiment / 'acceptance/summary.json').resolve())}
    write_json(experiment / 'acceptance.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('develop', 'freeze', 'accept'))
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--revision', type=int)
    args = parser.parse_args()
    if args.command == 'develop':
        if args.policy is None or args.revision is None:
            parser.error('develop requires --policy and --revision')
        result = develop(args.policy, args.revision, args.experiment, config=args.protocol)
    elif args.command == 'freeze':
        if args.revision is None:
            parser.error('freeze requires --revision')
        result = freeze(args.experiment, args.revision, config=args.protocol)
    else:
        result = accept(args.experiment, config=args.protocol)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
