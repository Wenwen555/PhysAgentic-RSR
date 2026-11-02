"""Paired case-cluster summaries, retained candidates and actual compute costs."""
import ast
from collections import Counter
import json
from pathlib import Path
import numpy as np
from physrsr.evaluate import cluster_ci


def report(root):
    root=Path(root);cfg=json.loads((root/'config.json').read_text())
    branches=json.loads((root/'frozen_programs.json').read_text())
    rr=[json.loads(s) for s in (root/'world_rollouts.jsonl').read_text().splitlines()]
    idx={(r['case'],r['generation'],r['program']['id'],r['record']['seed']):r for r in rr}
    expected={(c,g,m,s) for c in range(len(cfg['evaluator_only_worlds'])) for g in range(cfg['generations']) for m in ('baseline','A','B','C','D') for s in cfg['world_seeds']}
    assert len(idx)==len(rr) and set(idx)==expected
    assert not any(r['record']['status']=='error' for r in rr),'Do not fold errors into failures'
    summary={};case_values={}
    for m in ('baseline','A','B','C','D'):
        vals=[np.mean([r['record']['status']=='success' for (c,g,a,s),r in idx.items() if c==case and a==m]) for case in range(len(cfg['evaluator_only_worlds']))]
        case_values[m]=vals
        summary[m]=dict(success=float(np.mean(vals)),ci95=cluster_ci(vals),successes=sum(r['record']['status']=='success' for r in rr if r['program']['id']==m),
            n=sum(r['program']['id']==m for r in rr),mean_execution_s=float(np.mean([r['record']['sim_duration_s'] for r in rr if r['program']['id']==m])),per_case=vals)
    paired={}
    for a,b in [('A','baseline'),('B','baseline'),('C','baseline'),('D','baseline'),('C','B'),('D','C'),('D','A'),('B','A')]:
        counts=Counter();keys=[]
        for c,g,m,s in idx:
            if m!=b:continue
            before=idx[c,g,b,s]['record']['status']=='success';after=idx[c,g,a,s]['record']['status']=='success'
            label={(True,True):'both_success',(False,True):'failure_to_success',(True,False):'success_to_failure',(False,False):'both_fail'}[before,after]
            counts[label]+=1;keys.append(dict(case=c,generation=g,seed=s,transition=label))
        delta=np.asarray(case_values[a])-case_values[b]
        paired[f'{a}_minus_{b}']=dict(mean=float(np.mean(delta)),ci95=cluster_ci(delta),transitions=dict(counts),trials=keys)
    decisions={m:dict(Counter(b['decisions'][m]['action'] for b in branches)) for m in ('A','B','C','D')}
    model_rows=[];transfer=[];changes=[];false_pass={}
    for b in branches:
        folder=Path(b['path']);base=(folder/'reviewed_baseline.py').read_text()
        for file in folder.glob('*.jsonl'):model_rows += [json.loads(s) for s in file.read_text().splitlines()]
        for m in ('B','D'):
            dev=[json.loads(s) for s in (folder/'baseline_dev.jsonl').read_text().splitlines()]
            after=[json.loads(s) for s in (folder/f'{m}_after.jsonl').read_text().splitlines()]
            repaired=[s for s in cfg['development_seeds'] if next(x for x in dev if x['record']['seed']==s)['record']['status']=='fail' and next(x for x in after if x['record']['seed']==s)['record']['status']=='success']
            worldpair=paired[f'{m}_minus_baseline']['trials']
            wins=sum(x['transition']=='failure_to_success' and x['case']==b['case'] and x['generation']==b['generation'] for x in worldpair)
            transfer.append(dict(case=b['case'],generation=b['generation'],method=m,development_repairs=len(repaired),
                deployed_world_repairs=wins,raw_candidate_world_repairs=wins if b['decisions'][m]['accepted'] else None,
                accepted=b['decisions'][m]['accepted']))
            candidate=(folder/f'candidate_{m}/program.py').read_text()
            changes.append(dict(case=b['case'],generation=b['generation'],method=m,
                ast_changed=ast.dump(ast.parse(base))!=ast.dump(ast.parse(candidate)),
                rationale=json.loads((folder/f'candidate_{m}/answer.json').read_text())['rationale']))
        for m in ('B','C','D'):
            selected_file=(('B' if m=='C' else m)+'_after.jsonl') if b['decisions'][m]['accepted'] else 'baseline_validation.jsonl'
            rows=[json.loads(s) for s in (folder/selected_file).read_text().splitlines()]
            val=[r for r in rows if r['record']['seed'] in cfg['validation_seeds']]
            if all(r['record']['status']=='success' for r in val):
                fail=sum(idx[b['case'],b['generation'],m,s]['record']['status']!='success' for s in cfg['world_seeds'])
                entry=false_pass.setdefault(m,dict(world_failures=0,world_trials=0,programs=0))
                entry['world_failures']+=fail;entry['world_trials']+=len(cfg['world_seeds']);entry['programs']+=1
    for v in false_pass.values():v['rate']=v['world_failures']/v['world_trials']
    transfer_summary={}
    for m in ('B','D'):
        eligible=[t for t in transfer if t['method']==m and t['development_repairs']>0]
        transfer_summary[m]=dict(programs_with_development_repair=len(eligible),
            programs_with_deployed_world_repair=sum(t['deployed_world_repairs']>0 for t in eligible),
            deployed_repair_transfer_rate=None if not eligible else sum(t['deployed_world_repairs']>0 for t in eligible)/len(eligible))
    calls=[]
    for file in root.glob('case_*/**/metadata.json'):
        metadata=json.loads(file.read_text());calls.append(dict(path=str(file),reused=(file.parent/'reused_from.txt').exists(),**metadata))
    tokens=Counter()
    for c in calls:
        for u in c.get('usage',[]):
            for k,v in (u or {}).items():
                if isinstance(v,(int,float)):tokens[k]+=v
    result=dict(summary=summary,paired=paired,decisions=decisions,repair_transfer=transfer,repair_transfer_summary=transfer_summary,code_changes=changes,
        false_pass=false_pass,cost=dict(new_llm_calls=sum(not c['reused'] for c in calls),logical_llm_nodes=len(calls),
            reused_calls=sum(c['reused'] for c in calls),logical_token_usage=dict(tokens),
            new_token_usage={k:sum((u or {}).get(k,0) for c in calls if not c['reused'] for u in c.get('usage',[])) for k in tokens},
            model_rollouts=len(model_rows),world_task_trials=len(rr),
            new_model_rollouts=sum('reused_from' not in r for r in model_rows),
            reused_model_rollouts=sum('reused_from' in r for r in model_rows),
            model_duration_s=sum(r['record'].get('sim_duration_s') or 0 for r in model_rows),
            program_errors=sum(r['error_kind']=='program' for r in model_rows)),
        limitations=['Small number of physical case clusters; generations and World trials are not independent cases.',
            'AST change is not a behavioral change; inspect stored diffs/rationales.',
            'Diagnostics are event measurements, not validated unique causal attributions.',
            'Raw D candidates rolled back are not World evaluated; candidate-level transfer for them is unobserved.'])
    (root/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(summary=summary,decisions=decisions,cost=result['cost']),indent=2),flush=True)
    return result


if __name__=='__main__':
    import argparse
    from physrsr.paths import output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=output_path)
    report(parser.parse_args().root)
