"""Case-cluster paired analysis, no decisions are changed by this module."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from physrsr.evaluate import cluster_ci, lower_bound


def read(path):return [json.loads(l) for l in path.read_text().splitlines()]


def report(root):
    cfg=json.loads((root/'config.json').read_text())
    adaptations=json.loads((root/'adaptations.json').read_text())
    decisions=json.loads((root/'decisions.json').read_text())
    tasks=read(root/'world_tasks.jsonl');predictions=read(root/'model_predictions.jsonl');probes=read(root/'world_probes.jsonl')
    rows=[]
    for i,(adapt,decision,truth) in enumerate(zip(adaptations,decisions,cfg['evaluator_only_worlds'])):
        rs=[r for r in tasks if r['index']==i]
        def records(pid):return sorted([r for r in rs if r['motion']==pid],key=lambda r:r['record']['seed'])
        def outcomes(pid):return [int(r['record']['status']=='success') for r in records(pid)]
        def rate(pid):return float(np.mean(outcomes(pid)))
        choices=dict(geometric='geometric',oracle='oracle_com',identified=adapt['direct'],identified_sim=decision['sim'],wrong_com='wrong_com')
        methods={}
        for method,pid in choices.items():
            recs=records(pid);p=recs[0]['program']
            methods[method]=dict(program=pid,offset=p['offset'],speed=p['speed'],success=rate(pid),outcomes=outcomes(pid),
                lower95=lower_bound(sum(outcomes(pid)),len(recs)),grasp_error_m=p['offset']-truth['com_offset'],
                execution_s=float(np.mean([r['record']['sim_duration_s'] for r in recs])),
                failure_modes=dict(Counter(r['record']['reason'] for r in recs if r['record']['status']=='fail')))
        ids=[p['id'] for p in adapt['candidates']]
        predicted=np.array([decision['feedback']['rates'][p] for p in ids]);actual=np.array([rate(p) for p in ids])
        # Independent seed split oracle: choose on 5101,5103,... evaluate on 5102,5104,...
        select_idx=list(range(0,len(cfg['test_seeds']),2));eval_idx=list(range(1,len(cfg['test_seeds']),2))
        oracle_pid=max(ids,key=lambda pid:np.mean(np.array(outcomes(pid))[select_idx]))
        oracle_rate=float(np.mean(np.array(outcomes(oracle_pid))[eval_idx]))
        rho=None if np.std(actual)==0 or np.std(predicted)==0 else float(spearmanr(predicted,actual).statistic)
        rows.append(dict(case=i,negative_control=i>=8,truth=truth,estimate=adapt['estimate'],methods=methods,
            com_error_m=adapt['estimate']['com_offset']-truth['com_offset'],
            parameter_support_covers_truth=adapt['estimate']['com_support'][0]<=truth['com_offset']<=adapt['estimate']['com_support'][1],
            prediction_mae=float(np.mean(abs(predicted-actual))),prediction_accuracy=float(np.mean((predicted>=.95)==(actual>=.95))),
            false_pass_n=int(np.sum((predicted>=.95)&(actual<.95))),predicted_pass_n=int(sum(predicted>=.95)),
            selected_false_pass=bool(decision['feedback']['rates'][decision['sim']]>=.95 and rate(decision['sim'])<.95),
            ranking_spearman=rho,candidate_rates={p:float(v) for p,v in zip(ids,actual)},
            independent_oracle_program=oracle_pid,independent_oracle_rate=oracle_rate,
            selection_regret={m:oracle_rate-float(np.mean(np.array(outcomes(pid))[eval_idx])) for m,pid in choices.items() if m in ('identified','identified_sim')},
            before_after=dict(before_com=0.,after_com=adapt['estimate']['com_offset'],before_grasp=0.,after_grasp=methods['identified']['offset'],
                before_world_success=methods['geometric']['success'],after_world_success=methods['identified']['success'])))
    summary={}
    for method in rows[0]['methods']:
        vals=[r['methods'][method]['success'] for r in rows]
        summary[method]=dict(success=float(np.mean(vals)),ci95=cluster_ci(vals),execution_s=float(np.mean([r['methods'][method]['execution_s'] for r in rows])))
    contrasts={}
    for name,a,b in [('delta_id','identified','geometric'),('delta_sim','identified_sim','identified')]:
        diff=[r['methods'][a]['success']-r['methods'][b]['success'] for r in rows]
        contrasts[name]=dict(mean=float(np.mean(diff)),ci95=cluster_ci(diff))
        dt=[r['methods'][a]['execution_s']-r['methods'][b]['execution_s'] for r in rows]
        contrasts[name]['execution_difference_s']=dict(mean=float(np.mean(dt)),ci95=cluster_ci(dt))
    errors=np.array([abs(r['com_error_m']) for r in rows]);success=np.array([r['methods']['identified']['success'] for r in rows])
    association=None if np.std(errors)==0 or np.std(success)==0 else float(spearmanr(errors,success).statistic)
    # Costs are not multiplied by methods that share exact same probe dataset.
    costs=dict(world_probe_count=len(probes),world_probe_duration_s=sum(r['record']['sim_duration_s'] for r in probes),
        world_evaluation_task_trials=len(tasks),model_task_rollouts=len(predictions),
        model_task_rollouts_per_case={i:sum(r['index']==i for r in predictions) for i in range(len(rows))},
        shared_id_library_rollouts=114,shared_id_cost_note='Both identified methods use common offline inverse-simulation calibration library; Direct has zero online TASK simulation',
        llm_tokens=0,explicit_world_reset_calls_including_fingerprints=len(probes)+len(tasks)+4*len(rows),
        reset_note='Counts explicit worker.reset calls; excludes backend construction initialization',
        world_evaluation_note='Full candidate scoring is evaluator cost, not Agent exploration')
    result=dict(summary=summary,paired_contrasts=contrasts,cases=rows,costs=costs,
        com_mae_m=float(np.mean(errors)),com_error_success_spearman=association,
        prediction_mae=float(np.mean([r['prediction_mae'] for r in rows])),
        false_pass_fraction=sum(r['false_pass_n'] for r in rows)/max(1,sum(r['predicted_pass_n'] for r in rows)),
        selected_false_pass_cases=sum(r['selected_false_pass'] for r in rows),
        limitations=['Parameterized template synthesis, not autonomous LLM code synthesis',
        '8 seeds per cell: empirical reliability only; case bootstrap conditional on sampled seeds',
        'Point-model verification; identification support is not calibrated uncertainty',
        'Static workspace/end-clearance filter is not a full kinematic/collision certificate',
        '10 physical cases including 2 prespecified negative controls; limited power',
        'Physics truth is evaluator-only, fixed-program process isolation is not hostile-code sandbox'])
    (root/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(summary=summary,paired_contrasts=contrasts,com_mae_m=result['com_mae_m']),indent=2))
    return result

if __name__=='__main__':
    import argparse
    from physrsr.paths import output_path
    ap=argparse.ArgumentParser();ap.add_argument('root',type=output_path)
    report(ap.parse_args().root)
