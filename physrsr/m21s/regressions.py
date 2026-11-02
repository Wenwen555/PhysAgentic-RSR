"""Prespecified known-bug replay; diagnostic evidence, not generalization."""
import argparse
from physrsr.paths import output_path
from dataclasses import asdict
import json
from pathlib import Path
from .experiment import trial,save,HISTORY
from .static import check
from .gate import decide
from physrsr.physics_spec import PhysicsSpec
from physrsr.m21r.runner import save_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=output_path,required=True);a=p.parse_args()
    old=json.loads((HISTORY/'config.json').read_text());est=json.loads((HISTORY/'estimates.json').read_text())
    save_config(a.out,dict(protocol='Known regressions only; no inference; old 8201/8202 diagnostic, new validation 11901..11904',world_seeds=[9101,9102,9103,9104]))
    def code(c,g,n):return (HISTORY/f'case_{c:02d}_generation_{g:02d}'/n/'program.py').read_text()
    bad=code(3,0,'physics_initial');fixed=code(3,0,'physics_final')
    fit=PhysicsSpec(com_offset=est[3]['com_offset'],friction=est[3]['friction'],grip_force=2.)
    replay=trial(fit,{'bad':bad,'fixed':fixed},[8201,8202],'model',a.out/'tilt_model.jsonl')
    duplicate=trial(fit,{'bad':bad},[8201],'model',a.out/'determinism.jsonl')[0]
    assert replay[0]['trajectory']==duplicate['trajectory'],'Instrumentation replay not deterministic'
    assert replay[0]['record']['status']=='fail'
    assert replay[0]['feedback']['milestones']['ever_lifted']
    assert 'never_grasped' not in replay[0]['feedback']['execution_summary']['failure_events']
    world=trial(PhysicsSpec(**old['evaluator_only_worlds'][3]),{'bad':bad,'fixed':fixed},[9101,9102,9103,9104],'world',a.out/'tilt_world.jsonl')
    before=code(1,1,'oracle_initial');after=code(1,1,'oracle_final')
    truth=PhysicsSpec(**old['evaluator_only_worlds'][1])
    oracle=trial(truth,{'before':before,'after':after},[8201,8202,11901,11902,11903,11904],'model',a.out/'oracle_model.jsonl')
    select=lambda name,seeds:[r for r in oracle if r['program']['id']==name and r['record']['seed'] in seeds]
    diagnostic=decide(select('before',[8201,8202]),select('after',[8201,8202]),dict(passed=True))
    assert diagnostic['action']=='rollback'
    normal=decide(select('before',[11901,11902,11903,11904]),select('after',[11901,11902,11903,11904]),check(after))
    assert not normal['accepted']
    bad_run=select('after',[8202])[0]
    assert bad_run['feedback']['milestones']['ever_lifted']
    assert any(e['while_lifted'] for e in bad_run['feedback']['milestones']['open_events'])
    def counts(rr):
        return {name:sum(r['record']['status']=='success' for r in rr if r['program']['id']==name) for name in {r['program']['id'] for r in rr}}
    result=dict(tilt_static_bad=check(bad),tilt_static_fixed=check(fixed),oracle_static_before=check(before),oracle_static_after=check(after),
        deterministic_trajectory_replay=True,tilt_model=counts(replay),tilt_world=counts(world),
        oracle_model=counts(oracle),oracle_diagnostic_gate=diagnostic,oracle_independent_validation_gate=normal,
        model_rollouts=len(replay)+1+len(oracle),world_rollouts=len(world),
        diagnostic_metrics=dict(known_process_labels_correct=2,total_known_cases=2,
            first_anomaly_detection='Checked against full control-step trace; no independent human time-label dataset; generic accuracy unmeasured'),
        limitations='Uses already diagnosed programs and old World seeds; not evidence of unseen-case generalization')
    save(a.out/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
