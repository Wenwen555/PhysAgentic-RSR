"""Hash-locked R1 execution baseline shared across robot-space CaP stages.

Write once after R1 acceptance, then verify before running new policies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import mujoco
import numpy as np

from .backend import ROOT

MANIFEST = ROOT / 'configs/robot/frozen_r1.json'
SOURCE = ROOT / 'assets/worlds/unidoor_reference'
PROTECTED = (
    'configs/robot/world_bindings/unidoor_reference.yaml',
    'src/active_sysid/robot/controller.py',
    'src/active_sysid/robot/adapter.py',
    'src/active_sysid/robot/runtime.py',
    'src/active_sysid/robot/recording.py',
    'src/active_sysid/robot/world_backend.py',
    'src/active_sysid/robot/world_runtime.py',
    'src/active_sysid/robot/world_evaluator.py',
    'src/active_sysid/robot/world_fixed_policy.py',
    'src/active_sysid/robot/stage_randomization.py',
    'src/active_sysid/robot/stages.py',
    'src/active_sysid/robot/stage_freeze.py',
    'configs/robot/stages/privileged_exact.yaml',
    'configs/robot/stages/visual_exact.yaml',
    'configs/robot/stages/visual_randomized.yaml',
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files() -> list[Path]:
    assets = sorted(p for p in SOURCE.rglob('*') if p.is_file()
                    and p.name != 'README.md' and 'validation' not in p.parts)
    return [ROOT / name for name in PROTECTED] + assets


def versions() -> dict:
    return {
        'python_major_minor': '.'.join(platform.python_version_tuple()[:2]),
        'mujoco': mujoco.__version__,
        'numpy': np.__version__,
    }


def make_manifest() -> dict:
    ref = json.loads((ROOT / 'runs/m3r_real_world/r1/metrics.json').read_text())
    assert ref['success'] and not ref['failed_required_gates']
    return {
        'baseline_id': 'unidoor_r1_native_robot_v1',
        'description': 'Native double-door World, fixed 7-DOF robot and lever-door R1',
        'runtime_versions': versions(),
        'baseline_reference': {
            'physics_steps': ref['physics_steps'],
            'final_door_angle_rad': ref['final_door_angle_rad'],
            'max_bolt_extension_m': ref['max_bolt_extension_m'],
            'bilateral_manipulation_contact_ratio': ref['bilateral_manipulation_contact_ratio'],
            'required_gates': ref['required_gates'],
            'reference_metrics_sha256': sha256(ROOT / 'runs/m3r_real_world/r1/metrics.json'),
        },
        'files_sha256': {
            str(path.relative_to(ROOT)): sha256(path) for path in protected_files()
        },
    }


def write_manifest(path: Path = MANIFEST) -> dict:
    manifest = make_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return manifest


def verify_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    expected = manifest['files_sha256']
    actual = {str(p.relative_to(ROOT)): sha256(p) for p in protected_files()}
    changes = [name for name in sorted(set(expected) | set(actual))
               if expected.get(name) != actual.get(name)]
    if changes:
        raise RuntimeError('Frozen R1 execution files changed: ' + ', '.join(changes[:15]))
    if versions() != manifest['runtime_versions']:
        raise RuntimeError(f"Frozen runtime versions differ: expected {manifest['runtime_versions']}, got {versions()}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'write'))
    args = parser.parse_args()
    manifest = write_manifest() if args.command == 'write' else verify_manifest()
    print(json.dumps({'status': 'pass', 'baseline_id': manifest['baseline_id'],
                      'protected_files': len(manifest['files_sha256']),
                      'runtime': manifest['runtime_versions']}, indent=2))


if __name__ == '__main__':
    main()
