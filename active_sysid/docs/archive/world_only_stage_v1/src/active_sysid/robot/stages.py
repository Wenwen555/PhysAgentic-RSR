"""Frozen robot-space CaP evaluation harness. Does not generate or revise CaP."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

from .backend import ROOT
from .stage_freeze import MANIFEST, sha256, verify_manifest
from .stage_randomization import apply_world_randomization
from .world_evaluator import evaluate_world_policy

STAGES = ('privileged_exact', 'visual_exact', 'visual_randomized')
STAGE_DIR = ROOT / 'configs/robot/stages'


def load_stage(name: str) -> dict:
    return yaml.safe_load((STAGE_DIR / f'{name}.yaml').read_text())


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location('robot_space_cap_policy', path.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def run_stage(stage: str, policy_path: str | Path, out_dir: str | Path, *,
              seeds=None, video=False) -> dict:
    frozen = verify_manifest()
    cfg = load_stage(stage)
    policy_path = Path(policy_path).resolve()
    policy = load_policy(policy_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = cfg['seeds'] if seeds is None else list(seeds)
    episodes = []
    for seed in seeds:
        modifier = (lambda backend, n=seed: apply_world_randomization(
            backend, n, cfg['randomization'])) if cfg['randomization'] else None
        episode = evaluate_world_policy(
            policy,
            out_dir=out_dir / f'seed_{seed:04d}',
            mechanism=cfg['mechanism'],
            observations=cfg['observations'],
            acceptance='task',
            max_steps=cfg['max_steps'],
            goal_angle_rad=cfg['goal_angle_rad'],
            visual_cameras=cfg['visual_cameras'],
            world_modifier=modifier,
            stage_name=stage,
            seed=seed,
            video=video,
        )
        episodes.append({
            'seed': seed,
            'success': episode['success'],
            'final_door_angle_rad': episode.get('final_door_angle_rad'),
            'execution_error': episode.get('execution_error'),
            'metrics_path': str((out_dir / f'seed_{seed:04d}' / 'metrics.json').resolve()),
        })
    summary = {
        'stage': stage,
        'baseline_id': frozen['baseline_id'],
        'frozen_manifest_sha256': sha256(MANIFEST),
        'stage_config_sha256': sha256(STAGE_DIR / f'{stage}.yaml'),
        'policy_path': str(policy_path),
        'policy_sha256': sha256(policy_path),
        'observations': cfg['observations'],
        'visual_cameras': cfg['visual_cameras'],
        'randomization_ranges': cfg['randomization'],
        'total_episodes': len(episodes),
        'successful_episodes': sum(episode['success'] for episode in episodes),
        'success_rate': sum(episode['success'] for episode in episodes) / len(episodes),
        'episodes': episodes,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary


def run_all(privileged_policy: Path, visual_policy: Path, out_dir: Path, *,
            randomized_policy: Path | None = None, randomized_seeds=None,
            video=False) -> dict:
    """Execute all stages; reuse visual CaP unchanged for robustness by default."""
    out_dir = Path(out_dir)
    randomized_policy = randomized_policy or visual_policy
    assignments = (
        ('privileged_exact', privileged_policy),
        ('visual_exact', visual_policy),
        ('visual_randomized', randomized_policy),
    )
    results = {}
    for stage, path in assignments:
        seeds = randomized_seeds if stage == 'visual_randomized' else None
        results[stage] = run_stage(stage, path, out_dir / stage, seeds=seeds, video=video)
    comparison = {
        'baseline_id': results['privileged_exact']['baseline_id'],
        'visual_policy_unchanged_in_randomized': (
            results['visual_exact']['policy_sha256'] ==
            results['visual_randomized']['policy_sha256']),
        'stages': {name: {'success_rate': record['success_rate'],
                          'successful_episodes': record['successful_episodes'],
                          'total_episodes': record['total_episodes'],
                          'policy_sha256': record['policy_sha256']}
                   for name, record in results.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run', 'run-all'))
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--privileged-policy', type=Path)
    parser.add_argument('--visual-policy', type=Path)
    parser.add_argument('--randomized-policy', type=Path,
                        help='Optional separate revision; defaults to the exact visual policy')
    args = parser.parse_args()
    if args.command == 'check':
        manifest = verify_manifest()
        print(json.dumps({'status':'pass','baseline_id':manifest['baseline_id'],
                          'protected_files':len(manifest['files_sha256'])}, indent=2))
    elif args.command == 'run':
        if args.stage is None or args.policy is None or args.out is None:
            parser.error('run requires --stage, --policy and --out')
        result = run_stage(args.stage, args.policy, args.out, seeds=args.seeds, video=args.video)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if args.privileged_policy is None or args.visual_policy is None or args.out is None:
            parser.error('run-all requires --privileged-policy, --visual-policy and --out')
        comparison = run_all(args.privileged_policy, args.visual_policy, args.out,
                             randomized_policy=args.randomized_policy,
                             randomized_seeds=args.seeds, video=args.video)
        print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
