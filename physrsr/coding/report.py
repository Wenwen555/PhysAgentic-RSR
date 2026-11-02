"""Paired, case-clustered LLM evaluation and code-change artifacts."""
import ast
from collections import Counter
import difflib
import json
from pathlib import Path
import numpy as np
from physrsr.evaluate import cluster_ci


def report(root):
    cfg=json.loads((root/'config.json').read_text())
    rows=[json.loads(l) for l in (root/'world_rollouts.jsonl').read_text().splitlines()]
    programs=json.loads((root/'frozen_programs.json').read_text())
    methods=cfg['modes']+['physics_initial']
    expected={(i,g,m,seed) for i in range(len(cfg['evaluator_only_worlds'])) for g in range(cfg['generations']) for m in methods for seed in cfg['world_seeds']}
    keys=[(r['case'],r['generation'],r['program']['id'],r['record']['seed']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Missing/duplicate paired World evaluations')
    cases=[]
    for i,truth in enumerate(cfg['evaluator_only_worlds']):
        cases.append(dict(case=i,physics=truth,methods={m:dict(
            success=float(np.mean([r['record']['status']=='success' for r in rows if r['case']==i and r['program']['id']==m])),
            status_counts=dict(Counter(r['record']['status'] for r in rows if r['case']==i and r['program']['id']==m)),
            error_kinds=dict(Counter(r['error_kind'] for r in rows if r['case']==i and r['program']['id']==m and r['error_kind'])),
            execution_s=[r['record']['sim_duration_s'] for r in rows if r['case']==i and r['program']['id']==m]) for m in methods}))
    summary={m:dict(success=float(np.mean([c['methods'][m]['success'] for c in cases])),
                    ci95=cluster_ci([c['methods'][m]['success'] for c in cases])) for m in methods}
    for method in methods:
        selected=[r for r in rows if r['program']['id']==method]
        execution=[r['record']['sim_duration_s'] for r in selected if r['record']['sim_duration_s'] is not None]
        rises=[]
        for r in selected:
            phases=r['record']['phases']
            if phases:rises.append(phases[-1]['rod_z']-phases[0]['rod_z'])
        summary[method].update(program_error_rate=sum(r['error_kind']=='program' for r in selected)/len(selected),
            mean_execution_s=None if not execution else float(np.mean(execution)),
            mean_achieved_rise_m=None if not rises else float(np.mean(rises)),
            mean_nominal_target_error_m=None if not rises else float(np.mean(np.maximum(0.,.10-np.array(rises)))))
    # Seeds/generation aligned within each physical case; uncertainty clusters
    # over cases, not individual code generations as independent physics cases.
    paired={}
    for name,a,b in [('physics_minus_vanilla','physics','vanilla'),('sim_minus_physics','sim','physics'),('sim_minus_shared_initial','sim','physics_initial')]:
        x=[c['methods'][a]['success']-c['methods'][b]['success'] for c in cases]
        a_rows={(r['case'],r['generation'],r['record']['seed']):r['record']['status']=='success' for r in rows if r['program']['id']==a}
        b_rows={(r['case'],r['generation'],r['record']['seed']):r['record']['status']=='success' for r in rows if r['program']['id']==b}
        counts=dict(both_success=0,success_to_failure=0,failure_to_success=0,both_fail=0)
        for key in a_rows:
            aa,bb=a_rows[key],b_rows[key]
            label='both_success' if aa and bb else 'success_to_failure' if bb else 'failure_to_success' if aa else 'both_fail'
            counts[label]+=1
        dt=[float(np.mean(c['methods'][a]['execution_s']))-float(np.mean(c['methods'][b]['execution_s'])) for c in cases if None not in c['methods'][a]['execution_s']+c['methods'][b]['execution_s']]
        paired[name]=dict(mean=float(np.mean(x)),ci95=cluster_ci(x),trial_pairs=counts,
            execution_difference_s=None if not dt else dict(mean=float(np.mean(dt)),ci95=cluster_ci(dt)))
    changes=[]
    for branch in programs:
        directory=Path(branch['path']);before=(directory/'physics_initial/program.py').read_text();after=branch['sources']['sim']
        direct_inputs=json.loads((directory/'physics_final/messages.json').read_text())
        sim_inputs=json.loads((directory/'sim_final/messages.json').read_text())
        if direct_inputs[:-1]!=sim_inputs[:-1] or branch['sources']['physics_initial']!=before:
            raise ValueError('Physics/Sim paired initial code or conversation mismatch')
        def structure(code):
            try:
                tree=ast.parse(code)
                return dict(loops=sum(isinstance(n,(ast.For,ast.While)) for n in ast.walk(tree)),
                            conditionals=sum(isinstance(n,ast.If) for n in ast.walk(tree)),
                            functions=sum(isinstance(n,ast.FunctionDef) for n in ast.walk(tree)))
            except SyntaxError:return dict(syntax_error=True)
        before_runs=[json.loads(l) for l in (directory/'sim_before.jsonl').read_text().splitlines()]
        before_status=Counter(r['record']['status'] for r in before_runs)
        changes.append(dict(case=branch['case'],generation=branch['generation'],code_changed=before!=after,
            ast_changed=None if 'syntax_error' in structure(before) or 'syntax_error' in structure(after) else ast.dump(ast.parse(before))!=ast.dump(ast.parse(after)),
            model_before_status=dict(before_status),
            before_structure=structure(before),after_structure=structure(after),
            interpretation='Structural counts are descriptive; inspect source/diff for actual semantic changes'))
        (directory/'simulation_revision.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='physics_initial.py',tofile='sim_final.py')))
    predictions=[]
    for branch in programs:
        directory=Path(branch['path'])
        for method,filename in [('sim','sim_after.jsonl'),('oracle_sim','oracle_after.jsonl')]:
            model=[json.loads(l) for l in (directory/filename).read_text().splitlines()]
            observed=[r for r in rows if r['case']==branch['case'] and r['generation']==branch['generation'] and r['program']['id']==method]
            predicted=float(np.mean([r['record']['status']=='success' for r in model]))
            actual=float(np.mean([r['record']['status']=='success' for r in observed]))
            predictions.append(dict(case=branch['case'],generation=branch['generation'],method=method,
                model_success=predicted,world_success=actual,empirical_false_pass=predicted>=.95 and actual<.95))
        v=(directory/'vanilla_initial/program.py').read_text()
        p=(directory/'physics_initial/program.py').read_text()
        for change in changes:
            if change['case']==branch['case'] and change['generation']==branch['generation']:
                change['physical_information_changed_code']=v!=p
        (directory/'physical_information.diff').write_text(''.join(difflib.unified_diff(v.splitlines(True),p.splitlines(True),fromfile='no_physics_initial.py',tofile='physics_initial.py')))
    predictions_summary={}
    for method in ('sim','oracle_sim'):
        subset=[p for p in predictions if p['method']==method]
        passed=[p for p in subset if p['model_success']>=.95]
        predictions_summary[method]=dict(success_prediction_mae=float(np.mean([abs(p['model_success']-p['world_success']) for p in subset])),
            predicted_pass_programs=len(passed),false_pass_world_trial_fraction=None if not passed else float(np.mean([1-p['world_success'] for p in passed])))
    usages=[];new_usages=[]
    metadata_files=list(root.glob('case_*_generation_*/*/metadata.json'))
    reused=0
    for p in metadata_files:
        metadata=json.loads(p.read_text());used=[u for u in metadata['usage'] if u]
        usages.extend(used)
        if (p.parent/'reused_from.txt').exists():reused+=1
        else:new_usages.extend(used)
    def token_totals(items):
        return {key:sum(u.get(key,0) for u in items) for key in ('input_tokens','output_tokens','cached_input_tokens','reasoning_output_tokens')}
    tokens=token_totals(usages)
    for method in methods:
        selected=[r for r in rows if r['program']['id']==method]
        summary[method]['mean_rpc_calls']=float(np.mean([sum(r.get('rpc',{}).get('rpc_counts',{}).values()) for r in selected]))
        summary[method]['mean_close_commands']=float(np.mean([r.get('rpc',{}).get('rpc_counts',{}).get('close_gripper',0) for r in selected]))
    model_rollouts=sum(len(p.read_text().splitlines()) for p in root.glob('case_*_generation_*/*_before.jsonl'))+sum(len(p.read_text().splitlines()) for p in root.glob('case_*_generation_*/*_after.jsonl'))
    estimates=json.loads((root/'estimates.json').read_text())
    parameter_errors=[dict(case=i,com_error_m=e['com_offset']-p['com_offset'],friction_error=e['friction']-p['friction']) for i,(e,p) in enumerate(zip(estimates,cfg['evaluator_only_worlds']))]
    probes=[json.loads(l) for l in (root/'probes.jsonl').read_text().splitlines()]
    probe_duration=sum(r['record']['sim_duration_s'] for r in probes)/len(cases)
    for method in cfg['modes']:
        summary[method]['logical_probe_count_per_object']=2 if method in ('physics','sim') else 0
        summary[method]['probe_plus_execution_s']=summary[method]['mean_execution_s']+(probe_duration if method in ('physics','sim') else 0)
    result=dict(summary=summary,paired=paired,cases=cases,code_changes=changes,predictions=predictions,prediction_summary=predictions_summary,parameter_errors=parameter_errors,
        costs=dict(logical_codex_invocations=len(metadata_files),new_codex_invocations=len(metadata_files)-reused,reused_codex_outputs=reused,tokens_logical=tokens,tokens_new=token_totals(new_usages),
            world_probes=len(probes),mean_probe_duration_per_object_s=probe_duration,world_task_trials=len(rows),model_task_rollouts=model_rollouts,
            shared_inverse_library_rollouts=114,monetary_cost=None),
        limitations=['Only four nominal cases in controlled run; case CI conditions on observed generations/seeds and may be degenerate',
            'Programs may use allowed online observations/reactive correction in every group',
            'Program errors separate from task failures; primary intention-to-treat SR counts errors as unsuccessful',
            'Same inference slots, not identical realized token usage; branch prefix is exact for Physics/Sim',
            'Oracle changes physical information and model only; no extra controller authority',
            'Template-era cases/results are development history, not this fresh heldout evaluation'])
    (root/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(summary=summary,paired=paired,costs=result['costs']),indent=2))
    return result

if __name__=='__main__':
    import argparse
    from physrsr.paths import output_path
    p=argparse.ArgumentParser();p.add_argument('root',type=output_path);report(p.parse_args().root)
