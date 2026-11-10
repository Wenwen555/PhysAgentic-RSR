"""All three CaP development stages run in independent reconstructed Sim only."""
import argparse
import json
from pathlib import Path
import yaml
from .fixture_backend import ROOT
from .stage_freeze import MANIFEST,sha256,verify_manifest
from .sim2sim import run_sim_episode,write_json

STAGES=('privileged_exact','visual_exact','visual_randomized')
STAGE_DIR=ROOT/'configs/robot/stages'


def load_stage(name):
    if name not in STAGES:raise ValueError('Unknown Sim development stage')
    cfg=yaml.safe_load((STAGE_DIR/f'{name}.yaml').read_text())
    if cfg.get('role')!='development_sim' or cfg.get('simulator')!='unidoor':
        raise ValueError('Stage configs must select reconstructed Sim, never World')
    return cfg


def run_stage(stage,policy_path,out_dir,*,seeds=None,video=False,sim_variant=None):
    frozen=verify_manifest();cfg=load_stage(stage)
    variant=sim_variant or cfg['sim_variant'];out_dir=Path(out_dir)
    seeds=cfg['seeds'] if seeds is None else list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:raise ValueError('Seeds must be distinct nonnegative integers')
    if variant not in ('initial','m1','m2','robot_m0','robot_m1','robot_m2'):raise ValueError('Unknown Sim variant')
    out_dir.mkdir(parents=True,exist_ok=False)
    snapshot=out_dir/'policy.py';snapshot.write_bytes(Path(policy_path).read_bytes())
    episodes=[]
    for seed in seeds:
        epdir=out_dir/f'seed_{seed:04d}'
        metrics=run_sim_episode(snapshot,epdir,variant=variant,observations=cfg['observations'],
          max_steps=cfg['max_steps'],stage=stage,seed=seed,ranges=cfg['randomization'],video=video,
          goal_angle_rad=cfg['goal_angle_rad'],cameras=cfg['visual_cameras'])
        episodes.append({'seed':seed,'success':metrics['success'],
          'execution_error':metrics.get('execution_error'),
          'metrics_path':str((epdir/'metrics.json').resolve()),'metrics_sha256':sha256(epdir/'metrics.json')})
    summary={'stage':stage,'role':'development_sim','baseline_id':frozen['baseline_id'],
      'frozen_manifest_sha256':sha256(MANIFEST),'stage_config_sha256':sha256(STAGE_DIR/f'{stage}.yaml'),
      'policy_sha256':sha256(snapshot),'policy_path':str(snapshot.resolve()),
      'sim_variant':variant,'observations':cfg['observations'],'visual_cameras':cfg['visual_cameras'],
      'randomization_ranges':cfg['randomization'],'total_episodes':len(episodes),
      'successful_episodes':sum(e['success'] for e in episodes),
      'success_rate':sum(e['success'] for e in episodes)/len(episodes),'episodes':episodes,
      'world_evaluations_performed':0}
    verify_manifest();write_json(out_dir/'summary.json',summary);return summary


def run_all(privileged_policy,visual_policy,out_dir,*,randomized_policy=None,
            randomized_seeds=None,video=False,sim_variant=None):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=False)
    paths=(privileged_policy,visual_policy,randomized_policy or visual_policy)
    results={}
    for stage,path in zip(STAGES,paths):
        results[stage]=run_stage(stage,path,out_dir/stage,
          seeds=randomized_seeds if stage=='visual_randomized' else None,video=video,sim_variant=sim_variant)
    comparison={'role':'development_sim','world_evaluations_performed':0,
      'baseline_id':results['privileged_exact']['baseline_id'],
      'visual_policy_unchanged_in_randomized':results['visual_exact']['policy_sha256']==results['visual_randomized']['policy_sha256'],
      'stages':{n:{k:r[k] for k in ('sim_variant','success_rate','successful_episodes','total_episodes','policy_sha256')} for n,r in results.items()}}
    write_json(out_dir/'comparison.json',comparison);return comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','run','run-all'))
    p.add_argument('--stage',choices=STAGES);p.add_argument('--policy',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--video',action='store_true')
    p.add_argument('--sim-variant',choices=('initial','m1','m2','robot_m0','robot_m1','robot_m2'))
    p.add_argument('--privileged-policy',type=Path);p.add_argument('--visual-policy',type=Path)
    p.add_argument('--randomized-policy',type=Path)
    a=p.parse_args()
    if a.command=='check':
        m=verify_manifest();result={'status':'pass','baseline_id':m['baseline_id'],'role':'development_sim'}
    elif a.command=='run':
        if a.stage is None or a.policy is None or a.out is None:p.error('run requires --stage, --policy, --out')
        result=run_stage(a.stage,a.policy,a.out,seeds=a.seeds,video=a.video,sim_variant=a.sim_variant)
    else:
        if a.privileged_policy is None or a.visual_policy is None or a.out is None:p.error('run-all requires --privileged-policy, --visual-policy, --out')
        result=run_all(a.privileged_policy,a.visual_policy,a.out,randomized_policy=a.randomized_policy,
                      randomized_seeds=a.seeds,video=a.video,sim_variant=a.sim_variant)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
