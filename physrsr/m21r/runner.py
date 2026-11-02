"""Trusted evaluation orchestration. Each call creates a role-dedicated pool."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import tarfile
import numpy as np
from physrsr.paths import PACKAGE_ROOT, WORKSPACE_ROOT, CAPX_ROOT


def batch(job):
    from physrsr.worker import RolloutWorker
    role, index, physics, programs, seeds = job
    worker = RolloutWorker(physics)
    out = []
    try:
        if not np.array_equal(worker.reset_fingerprint(seeds[0]), worker.reset_fingerprint(seeds[0])):
            raise RuntimeError('Initial state fingerprint mismatch')
        for p in programs:
            for seed in seeds:
                r = asdict(worker.rollout(p, seed, role=role))
                item = dict(role=role, index=index, motion=p.id, program=p.specification(), observations=list(p.trace), record=r)
                out.append(item)
                if r['status']=='error':
                    # Preserve the error record, then abort in the parent.
                    return out
        return out
    finally:
        worker.close()


def execute(jobs, destination, workers=8):
    roles = {j[0] for j in jobs}
    if len(roles) != 1:
        raise ValueError('Never mix Model and World roles in a pool')
    records = []
    with Path(destination).open('x') as f:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(batch, j) for j in jobs]
            for future in as_completed(futures):
                completed = future.result()
                for r in completed:
                    f.write(json.dumps(r)+'\n')
                f.flush()
                if any(r['record']['status']=='error' for r in completed):
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError('Infrastructure error persisted; experiment aborted')
                records.extend(completed)
                print(f'{next(iter(roles))}: {len(records)} rollouts', flush=True)
    return records


def save_config(root, config):
    root.mkdir(parents=True, exist_ok=False)
    paths = list(PACKAGE_ROOT.rglob('*.py'))
    paths += [CAPX_ROOT / 'capx/envs/simulators/robosuite_rod_lift.py', CAPX_ROOT / 'capx/integrations/franka/control_privileged.py']
    config.update(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CAPX_ROOT,text=True).strip(),
                  dirty_status=subprocess.check_output(['git','status','--short'],cwd=CAPX_ROOT,text=True),
                  commit_scope='src/cap-x dependency repository; PhysRSR source recorded by SHA256',
                  source_sha256={str(p.relative_to(WORKSPACE_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (root/'config.json').write_text(json.dumps(config,indent=2))
    with tarfile.open(root/'source.tar.gz','w:gz') as archive:
        for p in paths: archive.add(p, arcname=str(p.relative_to(WORKSPACE_ROOT)))


def write_programs(root, programs):
    root.mkdir(exist_ok=True)
    for i,p in enumerate(programs):
        (root/f'{i:02d}.py').write_text(p.source())
    (root/'programs.json').write_text(json.dumps([p.specification() for p in programs],indent=2))
