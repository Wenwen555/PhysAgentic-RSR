"""Versioned Sim2Sim boundary release. Legacy World-only v1 is retained."""
import hashlib
import json
import platform
from pathlib import Path
import mujoco
import numpy as np
from .fixture_backend import ROOT

MANIFEST=ROOT/'configs/robot/frozen_sim2sim.json'
SIM_ROOT=ROOT/'assets/reconstructed_sim/visual_stage3_v1'
LEGACY=ROOT/'docs/archive/world_only_stage_v1/frozen_r1.json'
MODULES=('controller','adapter','runtime','recording','fixture_backend','fixture_evaluator','fixture_policy','reference_world_backend','reference_world_evaluator',
         'reference_oracle_policy','reference_world_runtime','reconstructed_sim_backend','sensors','boundary_runtime',
         'policy_worker','policy_rpc','cap_protocol','sim2sim','stage_randomization','stages','stage_freeze','__main__')
PROTECTED=tuple(f'src/active_sysid/robot/{n}.py' for n in MODULES)+(
    'configs/robot/simulators/legacy_visual_stage3.yaml','configs/robot/simulators/robot_grounding_m0_m2.yaml','configs/robot/world_bindings/unidoor_native.yaml',
    'configs/robot/cap_protocol.yaml','configs/robot/stages/privileged_exact.yaml','configs/robot/stages/visual_exact.yaml',
    'configs/robot/stages/visual_randomized.yaml','scripts/build_reconstructed_robot_sim.py',
    'scripts/run_robot_grounding.py','src/active_sysid/grounding/__init__.py','src/active_sysid/grounding/task_interaction.py',
    'pyproject.toml','uv.lock')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {'python_major_minor':'.'.join(platform.python_version_tuple()[:2]),
            'mujoco':mujoco.__version__,'numpy':np.__version__}


def protected_files():
    return [ROOT/p for p in PROTECTED]+sorted(p for p in SIM_ROOT.rglob('*') if p.is_file())+sorted(p for p in (ROOT/'assets/reconstructed_sim/task_grounded_v1').rglob('*') if p.is_file())


def write_manifest(path=MANIFEST):
    legacy=json.loads(LEGACY.read_text())
    manifest={'baseline_id':'unidoor_sim2sim_boundary_v2',
        'role':'sim_development_world_probe_frozen_evaluation',
        'supersedes':'unidoor_r1_native_robot_v1_world_only_stage_harness',
        'runtime_versions':versions(),'baseline_reference':legacy['baseline_reference'],
        'files_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in protected_files()},
        # Values copied from previously frozen manifest. Stage validation does not
        # parse/read World assets. They are checked only at the World boundary.
        'world_files_sha256':{
            k.replace('assets/worlds/unidoor_reference/', 'assets/reference_world/unidoor_native/'): v
            for k, v in legacy['files_sha256'].items()
            if k.startswith('assets/worlds/unidoor_reference/')}}
    Path(path).write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def verify_manifest(path=MANIFEST):
    manifest=json.loads(Path(path).read_text());expected=manifest['files_sha256']
    actual={str(p.relative_to(ROOT)):sha256(p) for p in protected_files()}
    changes=[k for k in sorted(set(expected)|set(actual)) if expected.get(k)!=actual.get(k)]
    if changes:raise RuntimeError('Frozen Sim2Sim files changed: '+', '.join(changes[:15]))
    if versions()!=manifest['runtime_versions']:raise RuntimeError('Frozen runtime version mismatch')
    return manifest


def verify_world_assets():
    manifest=verify_manifest()
    changed=[k for k,v in manifest['world_files_sha256'].items() if sha256(ROOT/k)!=v]
    if changed:raise RuntimeError('Reference World assets changed: '+', '.join(changed[:8]))
    return manifest


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('check','write'))
    args=p.parse_args();m=write_manifest() if args.command=='write' else verify_manifest()
    print(json.dumps({'status':'pass','baseline_id':m['baseline_id'],
                     'protected_sim_and_code_files':len(m['files_sha256']),
                     'separately_pinned_world_files':len(m['world_files_sha256'])},indent=2))

if __name__=='__main__':main()
