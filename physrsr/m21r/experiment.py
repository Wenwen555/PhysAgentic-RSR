"""Frozen held-out paired physics-conditioned synthesis experiment.

World truths live in the trusted evaluator. The adaptation function consumes
only tracking traces, shared calibration library, and known control constraints.
"""
import argparse
from physrsr.paths import OUTPUT_ROOT, output_path
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from physrsr.physics_spec import PhysicsSpec
from .core import PROBES, Program, response, identify, generate_candidates, select_sim
from .identification import load_library
from .runner import save_config, execute, write_programs


def sensor_view(records):
    # Evaluator RolloutRecord.case_id encodes physics. Never give that metadata
    # to the adaptation controller, even though inference never read it.
    return [{k:r[k] for k in ('index','motion','observations')} for r in records]


def adapt(probe_records, index, library, constraints):
    if any(set(r) != {'index','motion','observations'} for r in probe_records):
        raise ValueError('Adaptation input must contain permitted sensors only')
    specs, predictions = library
    observed = [response(next(r['observations'] for r in probe_records if r['index']==index and r['motion']==p.id)).tolist() for p in PROBES]
    estimate = identify(observed, specs, predictions)
    candidates = generate_candidates('rod_lift', estimate, dict(com_support=estimate['com_support']), constraints)
    for p in candidates: compile(p.source(),'<candidate>','exec')
    direct = next(p for p in candidates if abs(p.offset-round(estimate['com_offset'],6))<1e-7 and p.speed==.35)
    return estimate, candidates, direct


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=output_path,required=True)
    ap.add_argument('--library',type=Path,default=OUTPUT_ROOT / 'physics_cap_m21r_id_library')
    ap.add_argument('--deltas',type=float,nargs='+',required=True)
    ap.add_argument('--test-seeds',type=int,default=8)
    a=ap.parse_args()
    rng=np.random.default_rng(20261102)
    # New continuous physical cases. Two prespecified low-friction controls.
    worlds=[PhysicsSpec(com_offset=float(sign*rng.uniform(.025,.08)),friction=float(rng.uniform(1.1,2.)),grip_force=2.)
            for sign in (-1,1,-1,1,-1,1,-1,1)]
    worlds += [PhysicsSpec(com_offset=float(sign*rng.uniform(.025,.07)),friction=float(rng.uniform(.55,.65)),grip_force=2.) for sign in (-1,1)]
    constraints=dict(deltas=a.deltas,speeds=[.35,None],max_offset=.105)
    seeds=tuple(range(5101,5101+a.test_seeds))
    save_config(a.out,dict(protocol='M2.1-R frozen holdout, 8 nominal + 2 low-friction controls, no case filtering',
        evaluator_only_worlds=[asdict(p) for p in worlds],world_probe_seeds=[4201],model_validation_seeds=[61,62],test_seeds=seeds,
        constraints=constraints,library=str(a.library),generation='deterministic template, one verification round; no LLM',
        predeclared_objective='empirical model success >=.95 then minimum physical execution time; max rate fallback',
        direct_compute_budget='same candidate generation and compilation count as simulator group',
        shared_identification_cost='inverse-model library common to both identified methods; no online task rollout for direct'))
    library=load_library(a.library)
    probes=execute([('world_probe',i,p,PROBES,(4201,)) for i,p in enumerate(worlds)],a.out/'world_probes.jsonl',workers=8)
    sensors=sensor_view(probes)
    (a.out/'agent_observations.json').write_text(json.dumps(sensors,indent=2))
    adaptations=[]
    model_jobs=[]
    prior_estimate=dict(com_offset=0.,friction=1.5,com_support=[-.09,.09])
    prior_candidates=generate_candidates('rod_lift',prior_estimate,{},constraints)
    write_programs(a.out/'prior_programs',prior_candidates)
    model_jobs.append(('model_task',-1,PhysicsSpec(com_offset=0.,friction=1.5,grip_force=2.),prior_candidates,(61,62)))
    for i in range(len(worlds)):
        estimate,candidates,direct=adapt(sensors,i,library,constraints)
        write_programs(a.out/f'case_{i:02d}_programs',candidates)
        adaptations.append(dict(index=i,estimate=estimate,candidates=[p.specification() for p in candidates],direct=direct.id,
            before=prior_estimate,static_checks_per_method=len(candidates)))
        model_jobs.append(('model_task',i,PhysicsSpec(com_offset=estimate['com_offset'],friction=estimate['friction'],grip_force=2.),candidates,(61,62)))
    # Persist model revision and program changes before any final World trials.
    (a.out/'adaptations.json').write_text(json.dumps(adaptations,indent=2))
    predictions=execute(model_jobs,a.out/'model_predictions.jsonl',workers=8)
    decisions=[];world_jobs=[]
    for i,truth in enumerate(worlds):
        entry=adaptations[i]
        candidates=[Program(**d) for d in entry['candidates']]
        sim,feedback=select_sim(candidates,[r for r in predictions if r['index']==i])
        decisions.append(dict(index=i,direct=entry['direct'],sim=sim.id,feedback=feedback,
            revision_reason='verification of local grasp and speed candidates with shared identified point model',
            oracle_and_world_outcomes_unavailable_at_selection=True))
        controls=[Program('geometric',0.),Program('oracle_com',truth.com_offset),
                  Program('wrong_com',-entry['estimate']['com_offset'])]
        # Evaluate every generated candidate on heldout World seeds for regret
        # and heldout prediction; labels remain evaluator-only.
        world_jobs.append(('world_task',i,truth,candidates+controls,seeds))
        write_programs(a.out/f'case_{i:02d}_evaluator_controls',controls)
    (a.out/'decisions.json').write_text(json.dumps(decisions,indent=2))
    execute(world_jobs,a.out/'world_tasks.jsonl',workers=8)
    from .report import report
    report(a.out)

if __name__=='__main__':main()
