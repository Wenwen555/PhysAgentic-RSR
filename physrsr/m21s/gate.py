"""Prespecified conservative paired validation; small samples imply abstention."""
from dataclasses import dataclass,asdict
from math import comb


@dataclass(frozen=True)
class GateConfig:
    alpha: float=.10
    minimum_repairs: int=2
    efficiency_fraction: float=.10
    constraint_gain_m: float=.005
    error_tolerance_m: float=.003


def sign_p(wins,losses):
    n=wins+losses
    return 1. if not n else sum(comb(n,i) for i in range(wins,n+1))/2**n


def decide(before,after,static_result,config=GateConfig()):
    by_seed=lambda rr:{r['record']['seed']:r for r in rr}
    a,b=by_seed(before),by_seed(after)
    if len(a)!=len(before) or len(b)!=len(after) or a.keys()!=b.keys() or not a:raise ValueError('Unpaired validation')
    result=dict(action='keep_baseline',accepted=False,rules=asdict(config),n=len(a))
    if not static_result['passed']:return dict(result,action='reject_static',reason='Deterministic contract counterexample')
    if any(r['record']['status']=='error' for r in before):raise RuntimeError('Baseline validation error must be diagnosed')
    if any(r['record']['status']=='error' for r in after):return dict(result,action='rollback',reason='Candidate program error')
    pairs=[(a[s],b[s]) for s in sorted(a)]
    repairs=sum(x['record']['status']=='fail' and y['record']['status']=='success' for x,y in pairs)
    breaks=sum(x['record']['status']=='success' and y['record']['status']=='fail' for x,y in pairs)
    result.update(repairs=repairs,breaks=breaks,repair_sign_p=sign_p(repairs,breaks))
    if breaks:return dict(result,action='rollback',reason='Paired success regression observed')
    severe={'release_while_lifted','lifted_then_dropped','program_error','execution_timeout'}
    for x,y in pairs:
        old=set(x['feedback']['execution_summary']['failure_events'])
        new=set(y['feedback']['execution_summary']['failure_events'])
        if (new-old)&severe:return dict(result,action='rollback',reason='New severe trajectory event')
    if repairs>=config.minimum_repairs and result['repair_sign_p']<=config.alpha:
        return dict(result,action='accept',accepted=True,reason='Paired repair evidence passes sign criterion')
    all_success=all(x['record']['status']==y['record']['status']=='success' for x,y in pairs)
    if all_success:
        gains=[x['record']['nominal_target_error_m']-y['record']['nominal_target_error_m'] for x,y in pairs]
        times=[(x['record']['sim_duration_s']-y['record']['sim_duration_s'])/max(x['record']['sim_duration_s'],1e-6) for x,y in pairs]
        if all(g>=-config.error_tolerance_m for g in gains) and all(t>=config.efficiency_fraction for t in times) and sign_p(len(times),0)<=config.alpha:
            return dict(result,action='accept',accepted=True,reason='All-success paired >=10% efficiency benefit with no target-error regression')
        if all(g>=config.constraint_gain_m for g in gains) and all(t>=-.05 for t in times) and sign_p(len(gains),0)<=config.alpha:
            return dict(result,action='accept',accepted=True,reason='All-success paired target-error improvement with <=5% time penalty')
    return dict(result,reason='Insufficient benefit evidence; equality alone is not acceptance')
