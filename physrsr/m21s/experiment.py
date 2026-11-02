"""M2.1-S paired protocol. No World outcomes enter generation or gating."""
import argparse
from physrsr.paths import OUTPUT_ROOT, output_path
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
from dataclasses import asdict
import difflib
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import threading
import numpy as np
from physrsr.coding.agent import CodexAgent
from physrsr.coding.prompts import initial_messages,information_message
from physrsr.m21r.runner import save_config,execute
from physrsr.m21r.core import PROBES,response,identify
from physrsr.m21r.identification import load_library
from physrsr.physics_spec import PhysicsSpec
from .trace import run_batch
from .static import check,REQUIREMENTS
from .gate import decide,GateConfig
from .feedback import world_to_base,completion_error

HISTORY=OUTPUT_ROOT / 'physics_cap_coding_final'
STOP=threading.Event()
REUSE=None


def save(path,value):
    path.write_text(json.dumps(value,indent=2))


def trial(physics,programs,seeds,role,path):
    if STOP.is_set():raise RuntimeError('Stopped after peer failure')
    cached=None if REUSE is None else REUSE/path.parent.name/path.name
    # Only Model results can be cached before generation freezes. Match exact
    # physics from the prior estimate, all source hashes, roles and paired seeds.
    if role=='model' and cached is not None and cached.exists():
        old_estimates=json.loads((REUSE/'estimates.json').read_text())
        index=int(path.parent.name.split('_')[1]);estimate=old_estimates[index]
        expected_physics=PhysicsSpec(com_offset=estimate['com_offset'],friction=estimate['friction'],grip_force=2.)
        rr=[json.loads(s) for s in cached.read_text().splitlines()]
        expected={(n,int(s),hashlib.sha256(c.encode()).hexdigest()) for n,c in programs.items() for s in seeds}
        actual={(r['program']['id'],r['record']['seed'],r['program']['source_sha256']) for r in rr}
        if physics==expected_physics and len(rr)==len(expected) and actual==expected and all(r['record']['role']=='model' and r['error_kind'] is None and r.get('frame_metadata') for r in rr):
            for r in rr:
                r['record']['nominal_target_error_m']=completion_error(r['trajectory'])
                r['reused_from']=str(cached)
            path.write_text(''.join(json.dumps(r)+'\n' for r in rr))
            return rr
    with ProcessPoolExecutor(1,mp_context=mp.get_context('spawn')) as pool:
        rr=pool.submit(run_batch,physics,programs,seeds,role).result()
    path.write_text(''.join(json.dumps(r)+'\n' for r in rr))
    if any(r['error_kind']=='infrastructure' for r in rr):raise RuntimeError(f'Infrastructure failure: {path}')
    return rr


def original_feedback(rows):
    # Legacy last-four representation, with the coordinate bug FIXED in all arms.
    feedback=[]
    for r in rows:
        samples=[];frame=r['frame_metadata']
        for s in r['record']['phases'][-4:]:
            item={k:s[k] for k in ('phase','rod_tilt','achieved','tcp_error','lifted','grasped')}
            item['rod_position']=world_to_base(s['rod_position'],frame['base_position_world'],frame['base_rotation_world'])
            samples.append(item)
        feedback.append(dict(coordinate_frame='robot_base',status=r['record']['status'],reason=r['record']['reason'],
            failure_phase=r['record']['failure_phase'],execution_s=r['record'].get('sim_duration_s'),
            error_kind=r['error_kind'],grasp_established=r['record']['grasp_established'],samples=samples))
    return feedback


def revision_message(static_result,evidence):
    return REQUIREMENTS+'\nDeterministic static report:\n'+json.dumps(static_result)+\
        '\nExecution evidence (null means unavailable; never invent outcomes):\n'+json.dumps(evidence)+\
        '\nSimulator results concern an estimated model, not World truth. Use the SAME static requirements.'


def call(agent,messages,path):
    if STOP.is_set():raise RuntimeError('Stopped after peer failure')
    cached=None if REUSE is None else REUSE/path.parent.name/path.name
    if cached is not None and (cached/'answer.json').exists() and (cached/'metadata.json').exists():
        meta=json.loads((cached/'metadata.json').read_text())
        if json.loads((cached/'messages.json').read_text())==messages and meta['model']==agent.model and meta['reasoning_effort']==agent.effort and meta['returncode']==0:
            shutil.copytree(cached,path);(path/'reused_from.txt').write_text(str(cached))
            answer=json.loads((path/'answer.json').read_text())
            return answer,messages+[dict(role='assistant',content=json.dumps(answer))]
    answer=agent.generate(messages,path)
    return answer,messages+[dict(role='assistant',content=json.dumps(answer))]


def branch(case,generation,estimate,root,cfg):
    folder=root/f'case_{case:02d}_generation_{generation:02d}';folder.mkdir()
    agent=CodexAgent()
    if cfg['cohort']=='regression':
        old=HISTORY/folder.name/'physics_initial'
        initial=json.loads((old/'answer.json').read_text())
        messages=json.loads((old/'messages.json').read_text())+[dict(role='assistant',content=json.dumps(initial))]
        (folder/'initial').mkdir();save(folder/'initial/answer.json',initial)
        save(folder/'initial/messages.json',messages[:-1]);(folder/'initial/program.py').write_text(initial['code'])
        save(folder/'initial/cache.json',dict(source=str(old),historical_logical_calls=2,new_calls=0))
    else:
        initial,messages=call(agent,initial_messages()+[dict(role='user',content=information_message(estimate))],folder/'initial')
    initial_check=check(initial['code']);save(folder/'initial_static.json',initial_check)
    reviewed,common=call(agent,messages+[dict(role='user',content=revision_message(initial_check,None))],folder/'shared_review')
    checked=check(reviewed['code']);save(folder/'shared_static.json',checked)
    shared_decision=dict(action='accept_review')
    if not checked['passed']:
        if not initial_check['passed']:raise RuntimeError(f'Neither initial nor shared review passes static contract: {folder}')
        # A failed static revision is a candidate rejection, not a system fault.
        # The SAME verified baseline and rejection evidence are shared by all arms.
        shared_decision=dict(action='rollback_static',reason='Reviewed candidate has a deterministic counterexample; initial passes')
        common += [dict(role='user',content='Shared deterministic verification rejected the reviewed candidate: '+json.dumps(checked)+
            '\nThe approved shared baseline is the original initial program below. Revise THIS program, retaining static requirements:\n'+initial['code'])]
        reviewed=initial;checked=initial_check
    save(folder/'shared_decision.json',shared_decision)
    (folder/'reviewed_baseline.py').write_text(reviewed['code'])
    fitted=PhysicsSpec(com_offset=estimate['com_offset'],friction=estimate['friction'],grip_force=2.)
    dev=trial(fitted,{'baseline':reviewed['code']},cfg['development_seeds'],'model',folder/'baseline_dev.jsonl')
    if any(r['record']['status']=='error' for r in dev):raise RuntimeError('Shared baseline program error')
    candidates={}
    for arm,evidence in [('A',None),('B',original_feedback(dev)),('D',[r['feedback'] for r in dev])]:
        candidates[arm],_=call(agent,common+[dict(role='user',content=revision_message(checked,evidence))],folder/f'candidate_{arm}')
    # C reuses B EXACTLY; no additional LLM draw.
    candidates['C']=candidates['B']
    checks={arm:check(candidates[arm]['code']) for arm in ('A','B','D')}
    checks['C']=checks['B']
    save(folder/'candidate_static.json',checks)
    before=trial(fitted,{'baseline':reviewed['code']},cfg['validation_seeds'],'model',folder/'baseline_validation.jsonl')
    after={}
    for arm in ('B','D'):
        after[arm]=trial(fitted,{arm:candidates[arm]['code']},cfg['development_seeds']+cfg['validation_seeds'],'model',folder/f'{arm}_after.jsonl')
    gatecfg=GateConfig(**cfg['gate'])
    val=lambda rr:[r for r in rr if r['record']['seed'] in cfg['validation_seeds']]
    decisions={arm:decide(before,val(after[candidate]),checks[arm],gatecfg) for arm,candidate in [('C','B'),('D','D')]}
    sources={'baseline':reviewed['code']}
    for arm in ('A','B'):
        ok=checks[arm]['passed']
        decisions[arm]=dict(accepted=ok,action='static_accept' if ok else 'reject_static',reason='Same deterministic static safety requirement for all arms')
        sources[arm]=candidates[arm]['code'] if ok else reviewed['code']
    for arm in ('C','D'):sources[arm]=candidates[arm]['code'] if decisions[arm]['accepted'] else reviewed['code']
    for arm in ('A','B','D'):
        (folder/f'{arm}.diff').write_text(''.join(difflib.unified_diff(reviewed['code'].splitlines(True),candidates[arm]['code'].splitlines(True),fromfile='shared_review',tofile=f'candidate_{arm}')))
    save(folder/'decisions.json',decisions)
    assert json.loads((folder/'candidate_A/messages.json').read_text())[:-1]==json.loads((folder/'candidate_B/messages.json').read_text())[:-1]==json.loads((folder/'candidate_D/messages.json').read_text())[:-1]
    result=dict(case=case,generation=generation,path=str(folder),sources=sources,
                candidate_hashes={a:hashlib.sha256(c['code'].encode()).hexdigest() for a,c in candidates.items()},decisions=decisions)
    save(folder/'frozen.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=output_path,required=True)
    p.add_argument('--cohort',choices=['regression','independent'],required=True)
    p.add_argument('--cases',type=int,default=4);p.add_argument('--generations',type=int,default=2)
    p.add_argument('--world-seeds',type=int,default=4);p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--physics-seed',type=int,default=20261107)
    p.add_argument('--seed-base',type=int,default=12001)
    p.add_argument('--gate-alpha',type=float,default=.10)
    p.add_argument('--reuse',type=Path)
    args=p.parse_args();STOP.clear()
    global REUSE
    REUSE=args.reuse
    if not 1<=args.cases<=4 or args.generations<1:p.error('Invalid prespecified size')
    if args.cohort=='regression':
        specs=[PhysicsSpec(**s) for s in json.loads((HISTORY/'config.json').read_text())['evaluator_only_worlds'][:args.cases]]
        estimates=json.loads((HISTORY/'estimates.json').read_text())[:args.cases]
    else:
        rng=np.random.default_rng(args.physics_seed)
        specs=[PhysicsSpec(com_offset=float(sign*rng.uniform(.025,.08)),friction=float(rng.uniform(1.1,2.)),grip_force=2.) for sign in (-1,1,-1,1)][:args.cases]
        estimates=None
    cfg=dict(protocol='M2.1-S v1; shared static review; A static, B original tail, C same B + gate, D structured + same gate',
        cohort=args.cohort,evaluator_only_worlds=[asdict(s) for s in specs],generations=args.generations,
        physics_seed=args.physics_seed,model='gpt-6-astra',reasoning_effort='low',
        development_seeds=[args.seed_base-200,args.seed_base-199],validation_seeds=list(range(args.seed_base-100,args.seed_base-96)),
        world_seeds=list(range(args.seed_base,args.seed_base+args.world_seeds)),probe_seed=args.seed_base-300,
        gate=asdict(GateConfig(alpha=args.gate_alpha)),
        reuse=None if args.reuse is None else str(args.reuse),
        note='Shared static revisions roll back to initial only if initial passes; otherwise abort. Tail frame corrected universally; C/D differ only representation. Thresholds fixed before World.')
    assert not(set(cfg['development_seeds'])&set(cfg['validation_seeds']) or set(cfg['world_seeds'])&(set(cfg['development_seeds'])|set(cfg['validation_seeds'])))
    save_config(args.out,cfg)
    if estimates is None:
        probe_rows=execute([('world_probe',i,s,list(PROBES),[cfg['probe_seed']]) for i,s in enumerate(specs)],args.out/'probes.jsonl',args.jobs)
        lib,y=load_library(OUTPUT_ROOT / 'physics_cap_m21r_id_library')
        estimates=[]
        for i in range(args.cases):
            obs=[response(next(r for r in probe_rows if r['index']==i and r['motion']==probe.id)['observations']) for probe in PROBES]
            e=identify(obs,lib,y);estimates.append({k:e[k] for k in ('com_offset','friction','com_support')})
    save(args.out/'estimates.json',estimates)
    branches=[]
    try:
        with ThreadPoolExecutor(args.jobs) as pool:
            fs=[pool.submit(branch,i,g,e,args.out,cfg) for i,e in enumerate(estimates) for g in range(args.generations)]
            try:
                for f in as_completed(fs):branches.append(f.result());print(f'Frozen {len(branches)}/{len(fs)}',flush=True)
            except BaseException:
                STOP.set()
                for f in fs:f.cancel()
                raise
        branches.sort(key=lambda b:(b['case'],b['generation']))
        save(args.out/'frozen_programs.json',branches)
        with (args.out/'world_rollouts.jsonl').open('x') as out:
            with ProcessPoolExecutor(args.jobs,mp_context=mp.get_context('spawn')) as pool:
                fs={pool.submit(run_batch,specs[b['case']],b['sources'],cfg['world_seeds'],'world'):b for b in branches}
                for f in as_completed(fs):
                    b=fs[f];rr=f.result()
                    for r in rr:out.write(json.dumps(dict(case=b['case'],generation=b['generation'],**r))+'\n')
                    out.flush()
                    if any(r['record']['status']=='error' for r in rr):raise RuntimeError('World error: diagnose before statistics')
                    print(f"World case{b['case']} gen{b['generation']} done",flush=True)
        from .report import report
        report(args.out)
    except BaseException as exc:
        STOP.set();save(args.out/'ABORTED.json',dict(error=str(exc)));raise


if __name__=='__main__':main()
