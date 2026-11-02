"""Build and read the shared fixed inverse-model calibration library."""
import argparse
from physrsr.paths import output_path
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from physrsr.physics_spec import PhysicsSpec
from .core import PROBES, response
from .runner import save_config, execute


def load_library(root):
    config=json.loads((root/'config.json').read_text())
    records=[json.loads(l) for l in (root/'model_rollouts.jsonl').read_text().splitlines()]
    responses=[]
    for i in range(len(config['models'])):
        responses.append([response(next(r['observations'] for r in records if r['index']==i and r['motion']==p.id)).tolist() for p in PROBES])
    return config['models'], responses


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=output_path,required=True);a=ap.parse_args()
    specs=[PhysicsSpec(com_offset=round(float(x),4),friction=mu,grip_force=2.) for x in np.linspace(-.09,.09,19) for mu in (1.,1.5,2.)]
    save_config(a.out,dict(models=[asdict(s) for s in specs],seeds=[41],probe_specs=[p.specification() for p in PROBES],
        response_scales=[.005,.005,.05],protocol='Development calibration, fixed regular grid, signed tracking, continuous interpolation'))
    execute([('model_id',i,s,PROBES,(41,)) for i,s in enumerate(specs)],a.out/'model_rollouts.jsonl',workers=8)
    print(f'Calibration library saved to {a.out}')

if __name__=='__main__':main()
