"""Four-arm Codex experiment; shared initial source and paired World trials."""
import argparse
from physrsr.paths import OUTPUT_ROOT, output_path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import threading
from pathlib import Path
import numpy as np

from physrsr.physics_spec import PhysicsSpec
from physrsr.m21r.core import PROBES,response,identify
from physrsr.m21r.identification import load_library
from physrsr.m21r.runner import execute,save_config
from .agent import CodexAgent
from .runtime import CodeProgram
from .prompts import initial_messages,information_message,static_review,simulation_review


_REUSE_ROOT=None
_ABORT_EVENT=threading.Event()


def turn(agent,messages,destination):
    if _ABORT_EVENT.is_set():raise RuntimeError('Canceled after peer experiment failure')
    cached=None
    if _REUSE_ROOT is not None:
        cached=_REUSE_ROOT/destination.parent.name/destination.name
    if cached is not None and (cached/'answer.json').is_file() and (cached/'metadata.json').is_file():
        previous=json.loads((cached/'messages.json').read_text())
        metadata=json.loads((cached/'metadata.json').read_text())
        if previous==messages and metadata['returncode']==0 and metadata['model']==agent.model and metadata['reasoning_effort']==agent.effort and metadata.get('disabled_features')==['code_mode_host','shell_tool','apps','multi_agent']:
            import shutil
            shutil.copytree(cached,destination)
            (destination/'reused_from.txt').write_text(str(cached))
            answer=json.loads((destination/'answer.json').read_text())
            return answer,messages+[dict(role='assistant',content=json.dumps(answer))]
    answer=agent.generate(messages,destination)
    return answer,messages+[dict(role='assistant',content=json.dumps(answer))]


def run_codes(physics,programs,seeds,role):
    from physrsr.worker import RolloutWorker
    worker=RolloutWorker(physics)
    # Fixed-template runs fit the old 1000-step horizon; general LLM programs
    # may legitimately execute longer. Keep task/hold physics unchanged and
    # enforce the SAME explicit RPC/wall budgets for every code arm instead.
    env,_=worker._ensure_env()
    env.robosuite_env.ignore_done=True
    rows=[]
    try:
        if not np.array_equal(worker.reset_fingerprint(seeds[0]),worker.reset_fingerprint(seeds[0])):
            raise RuntimeError('Non-deterministic reset fingerprint')
        for program in programs:
            for seed in seeds:
                program.error_kind='infrastructure';program.trace.clear();program.runtime={}
                record=asdict(worker.rollout(program,seed,role=role))
                kind=program.error_kind if record['status']=='error' else None
                samples=record['phases']
                rise=None if not samples else samples[-1]['rod_z']-samples[0]['rod_z']
                record['achieved_rise_m']=rise
                record['nominal_target_error_m']=None if rise is None else max(0.,.10-rise)
                rows.append(dict(program=program.specification(),record=record,error_kind=kind,
                                 rpc=program.runtime))
                if kind=='infrastructure':break
            if rows[-1]['error_kind']=='infrastructure':break
    finally:worker.close()
    return rows


def model_feedback(rows):
    feedback=[]
    for item in rows:
        r=item['record']
        phases=r['phases']
        feedback.append(dict(status=r['status'],reason=r['reason'],error_kind=item['error_kind'],
            failure_phase=r['failure_phase'],execution_s=r['sim_duration_s'],
            achieved_rise_m=r.get('achieved_rise_m'),nominal_target_error_m=r.get('nominal_target_error_m'),
            grasp_established=r['grasp_established'],
            samples=[{k:p[k] for k in ('phase','rod_tilt','rod_position','achieved','tcp_error','lifted','grasped')} for p in phases[-4:]]))
    return feedback


def check_infrastructure(rows):
    if any(r['error_kind']=='infrastructure' for r in rows):
        raise RuntimeError('Infrastructure error persisted; aborting experiment')


def model_trial(physics,source,pid,path,seeds):
    if _ABORT_EVENT.is_set():raise RuntimeError('Canceled after peer experiment failure')
    # Spawned Model process never shares World object/env memory.
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
        rows=pool.submit(run_codes,physics,[CodeProgram(pid,source)],seeds,'model').result()
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    check_infrastructure(rows)
    return rows


def generate_branch(index,generation,estimate,oracle_physics,root,model_seeds):
    destination=root/f'case_{index:02d}_generation_{generation:02d}';destination.mkdir()
    agent=CodexAgent()
    initial,common=turn(agent,initial_messages(),destination/'initial')
    vanilla,vmsg=turn(agent,common+[dict(role='user',content=information_message())],destination/'vanilla_initial')
    physics,pmsg=turn(agent,common+[dict(role='user',content=information_message(estimate))],destination/'physics_initial')
    oracle_info=dict(com_offset=oracle_physics.com_offset,friction=oracle_physics.friction,
                     com_support=[oracle_physics.com_offset,oracle_physics.com_offset])
    oracle,omsg=turn(agent,common+[dict(role='user',content=information_message(oracle_info))],destination/'oracle_initial')
    fitted=PhysicsSpec(com_offset=estimate['com_offset'],friction=estimate['friction'],grip_force=2.)
    simulator_rows=model_trial(fitted,physics['code'],'shared_physics_initial',destination/'sim_before.jsonl',model_seeds)
    oracle_rows=model_trial(oracle_physics,oracle['code'],'oracle_initial',destination/'oracle_before.jsonl',model_seeds)
    # Same model/effort and one revision call for all arms. The ONLY difference
    # between the two identified forks is static review vs Model execution data.
    vanilla_final,_=turn(agent,vmsg+[dict(role='user',content=static_review(vanilla['code']))],destination/'vanilla_final')
    direct_final,_=turn(agent,pmsg+[dict(role='user',content=static_review(physics['code']))],destination/'physics_final')
    sim_final,_=turn(agent,pmsg+[dict(role='user',content=simulation_review(model_feedback(simulator_rows)))],destination/'sim_final')
    oracle_final,_=turn(agent,omsg+[dict(role='user',content=simulation_review(model_feedback(oracle_rows)))],destination/'oracle_final')
    model_trial(fitted,sim_final['code'],'sim_final',destination/'sim_after.jsonl',model_seeds)
    model_trial(oracle_physics,oracle_final['code'],'oracle_final',destination/'oracle_after.jsonl',model_seeds)
    sources={name:answer['code'] for name,answer in (
        ('vanilla',vanilla_final),('physics',direct_final),('sim',sim_final),('oracle_sim',oracle_final))}
    # Evaluate shared initial code too: separates actual simulator revision from
    # the extra static-review turn in the compute-matched Physics baseline.
    sources['physics_initial']=physics['code']
    decision=dict(case=index,generation=generation,physics_initial_sha256=hashlib.sha256(physics['code'].encode()).hexdigest(),
        shared_physics_sim_initial=True,
        branches={name:hashlib.sha256(code.encode()).hexdigest() for name,code in sources.items()},
        simulator_revised_code=sim_final['code']!=physics['code'],
        llm_calls_unique=8,llm_calls_per_final_path=3,
        note='Two identified methods share the identical physics_initial assistant turn; all methods receive one final revision/review slot')
    (destination/'decision.json').write_text(json.dumps(decision,indent=2))
    return dict(case=index,generation=generation,sources=sources,path=str(destination))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=output_path,required=True)
    parser.add_argument('--library',type=Path,default=OUTPUT_ROOT / 'physics_cap_m21r_id_library')
    parser.add_argument('--cases',type=int,default=2)
    parser.add_argument('--generations',type=int,default=1)
    parser.add_argument('--world-seeds',type=int,default=4)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--physics-seed',type=int,default=20261105)
    parser.add_argument('--seed-base',type=int,default=8101)
    parser.add_argument('--reuse-generations',type=Path)
    args=parser.parse_args()
    global _REUSE_ROOT
    _REUSE_ROOT=args.reuse_generations
    _ABORT_EVENT.clear()
    if not 1<=args.cases<=10 or args.generations<1 or args.world_seeds<1:parser.error('Invalid experiment size')
    rng=np.random.default_rng(args.physics_seed)
    specs=[PhysicsSpec(com_offset=float(sign*rng.uniform(.025,.08)),friction=float(rng.uniform(1.1,2.)),grip_force=2.) for sign in (-1,1,-1,1,-1,1,-1,1)]
    specs += [PhysicsSpec(com_offset=float(sign*rng.uniform(.025,.07)),friction=float(rng.uniform(.55,.65)),grip_force=2.) for sign in (-1,1)]
    specs=specs[:args.cases]
    model_seeds=(args.seed_base-900,args.seed_base-899);world_seeds=tuple(range(args.seed_base,args.seed_base+args.world_seeds))
    probe_seed=args.seed_base-200
    save_config(args.out,dict(protocol='Codex Astra low, paired shared initial code; equal three inference slots per final path',
        evaluator_only_worlds=[asdict(s) for s in specs],generations=args.generations,
        model='gpt-6-astra',reasoning_effort='low',physics_seed=args.physics_seed,probe_seed=probe_seed,model_seeds=model_seeds,world_seeds=world_seeds,
        modes=['vanilla','physics','sim','oracle_sim'],library=str(args.library),reuse_generations=None if _REUSE_ROOT is None else str(_REUSE_ROOT),
        backend_horizon='ignore_done=True; original task success and hold unchanged; 96 RPC calls and 120s host wall limit',
        cases_note='Pilot uses first N of frozen 8 nominal + 2 negative control sequence; no outcome selection',
        sandbox='Separate bubblewrap namespaces, repository hidden from Codex and generated Python; RPC robot API only'))
    probes=execute([('coding_world_probe',i,s,PROBES,(probe_seed,)) for i,s in enumerate(specs)],args.out/'probes.jsonl',workers=min(args.jobs,len(specs)))
    library_specs,library_y=load_library(args.library)
    estimates=[]
    for i in range(len(specs)):
        observation=[response(next(r['observations'] for r in probes if r['index']==i and r['motion']==p.id)).tolist() for p in PROBES]
        e=identify(observation,library_specs,library_y)
        estimates.append({k:e[k] for k in ('com_offset','friction','com_support')})
    (args.out/'estimates.json').write_text(json.dumps(estimates,indent=2))
    branches=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(generate_branch,i,g,e,s,args.out,model_seeds)
                 for i,(e,s) in enumerate(zip(estimates,specs)) for g in range(args.generations)]
        try:
            for f in as_completed(futures):
                branches.append(f.result());print(f'Frozen programs {len(branches)}/{len(futures)}',flush=True)
        except BaseException:
            _ABORT_EVENT.set()
            for pending in futures:pending.cancel()
            raise
    branches.sort(key=lambda b:(b['case'],b['generation']))
    # All programs and revisions are frozen before the first World task score.
    (args.out/'frozen_programs.json').write_text(json.dumps(branches,indent=2))
    with (args.out/'world_rollouts.jsonl').open('x') as out:
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(run_codes,specs[b['case']],[CodeProgram(n,s) for n,s in b['sources'].items()],world_seeds,'world'):b for b in branches}
            for f in as_completed(futures):
                b=futures[f];rows=f.result()
                for r in rows:out.write(json.dumps(dict(case=b['case'],generation=b['generation'],**r))+'\n')
                out.flush();check_infrastructure(rows)
                print(f"World evaluated case{b['case']} generation{b['generation']}",flush=True)
    from .report import report
    report(args.out)

if __name__=='__main__':main()
