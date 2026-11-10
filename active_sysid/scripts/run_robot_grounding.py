"""Reproduce Robot-Space M0/M1/M2 from robot-only evidence and minimal models.

Use --probes to replay existing public evidence rather than query World again.
The fixed opening programs below are validation references, not generated CaP.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from active_sysid.robot.fixture_backend import ROOT
from active_sysid.grounding.task_interaction import write_probe,write_arc_probe,build_task_sim,summarize_probe,summarize_arc
from active_sysid.robot.sim2sim import probe_world,run_sim_episode,write_json
from active_sysid.robot.stage_freeze import write_manifest

DOC=ROOT/'docs/ROBOT_SPACE_M0_M2_IMPLEMENTATION.md'
ASSET=ROOT/'assets/reconstructed_sim/task_grounded_v1'
TRAIN={'direct':0.,'rotate100':.10,'rotate015':.15,'rotate035':.35}
HELD={'heldout_direct':0.,'heldout_rotate045':.45}
ARC={'arc_r65':.65,'arc_r75':.75,'arc_r85':.85}


def mark(stage,summary):
    text=DOC.read_text()
    lines=[]
    for line in text.splitlines():
        if line.startswith(f'| {stage}-R：'):
            columns=line.split('|');columns[2]=' **完成** ';columns[4]=f' `{summary}` '
            line='|'.join(columns)
        lines.append(line)
    DOC.write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('m0','m1','m2','all'))
    p.add_argument('--out',type=Path,default=Path('runs/robot_grounding_final'))
    p.add_argument('--probes',type=Path,help='Existing PUBLIC probe evidence directory')
    p.add_argument('--demo',type=Path,default=ASSET/'robot_contact_demonstration.json',help='Robot-only contact demonstration')
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    probes=args.probes or out/'probes';programs=out/'programs';programs.mkdir(exist_ok=True)
    demo_path=out/'m0/robot_demonstration.json'
    def raw_probe(name,**kwargs):
        policy=write_probe(demo,programs/f'{name}.py',**kwargs)
        directory=probes/name
        if not directory.exists():
            write_manifest();probe_world(policy,directory)
        assert json.loads((directory/'probe.json').read_text())['policy_sha256']==hashlib.sha256(policy.read_bytes()).hexdigest()
        return directory
    def raw_arc(name,radius,angle=.20):
        policy=write_arc_probe(demo,programs/f'{name}.py',radius=radius,angle=angle)
        directory=probes/name
        if not directory.exists():
            write_manifest();probe_world(policy,directory)
        assert json.loads((directory/'probe.json').read_text())['policy_sha256']==hashlib.sha256(policy.read_bytes()).hexdigest()
        return directory
    def sim(name,variant,directory):
        if not directory.exists():
            run_sim_episode(programs/f'{name}.py',directory,variant=variant,observations='visual')
        return directory
    def model(variant,**kwargs):
        return build_task_sim(demo,contact,ASSET/f'{variant}.xml',**kwargs)

    if args.stage in ('m0','all'):
        demo=json.loads(args.demo.read_text())
        demo_path.parent.mkdir(exist_ok=True);write_json(demo_path,demo)
        contact=summarize_probe(raw_probe('contact',pull=0))
        model('robot_m0')
        s=summarize_probe(sim('contact','robot_m0',out/'m0/contact_sim'))
        result={'stage':'M0-R','completed':contact['completed'] and s['completed'] and
                contact['grasp_bilateral_ratio']>.95 and s['grasp_bilateral_ratio']>.95,
                'world_contact':contact,'sim_contact':s,
                'gripper_width_error_m':abs(contact['gripper_width_m']-s['gripper_width_m']),
                'grasp_tcp_error_m':float(np.linalg.norm(np.array(contact['grasp_tcp_m'])-s['grasp_tcp_m'])),
                'geometry_model':str(ASSET/'robot_m0.xml')}
        write_json(out/'m0/completion.json',result)
        assert result['completed'];mark('M0',out/'m0/completion.json')
        print('M0 completed',result['gripper_width_error_m'],flush=True)
    else:
        demo=json.loads(demo_path.read_text());contact=summarize_probe(probes/'contact')

    if args.stage in ('m1','all'):
        world={}
        for name,angle in {**TRAIN,**HELD}.items():
            directory=raw_probe(name,rotate=angle,pull=.010 if name in HELD else .015,
                                duration=1.0 if name in HELD else 1.6)
            world[name]=summarize_probe(directory)
        closed=[a for n,a in TRAIN.items() if world[n]['tracking_ratio']<=.95]
        mobile=[a for n,a in TRAIN.items() if world[n]['tracking_ratio']>.95]
        interval=[max(closed),min(mobile)];threshold=sum(interval)/2
        model('robot_m1',latch=True,threshold=threshold)
        records=[]
        for name in world:
            record={'probe':name,'split':'heldout' if name in HELD else 'fit','world':world[name]}
            for v in ('robot_m0','robot_m1'):
                record[v]=summarize_probe(sim(name,v,out/'m1'/v/name))
            records.append(record)
        result={'stage':'M1-R','transition_interval_rad':interval,'threshold_sim_rad':threshold,
                'condition':'relative handle rotation enables low-resistance pull',
                'records':records,'complete':True}
        for split in ('fit','heldout'):
            rs=[x for x in records if x['split']==split]
            result[split+'_accuracy']={v:sum((x[v]['tracking_ratio']>.95)==(x['world']['tracking_ratio']>.95) for x in rs)/len(rs)
                                      for v in ('robot_m0','robot_m1')}
        result['completed']=result['fit_accuracy']['robot_m1']==1 and result['heldout_accuracy']['robot_m1']==1
        write_json(out/'m1/task_rule.json',result);assert result['completed']
        mark('M1',out/'m1/task_rule.json');print('M1 completed',result['fit_accuracy'],result['heldout_accuracy'],flush=True)

    if args.stage in ('m2','all'):
        threshold=json.loads((out/'m1/task_rule.json').read_text())['threshold_sim_rad']
        world={n:summarize_arc(raw_arc(n,r)) for n,r in ARC.items()}
        candidates=[]
        for radius in (.65,.70,.75,.80,.85):
            model('robot_m2',radius=radius,latch=True,threshold=threshold)
            predictions={n:summarize_arc(sim(n,'robot_m2',out/'m2/fit'/f'r{radius:.2f}'/n)) for n in ARC}
            cost=np.mean([(np.linalg.norm(np.array(predictions[n]['arc_tcp_m'])-world[n]['arc_tcp_m'])/.005)**2+
                          ((predictions[n]['arc_force_change_n']-world[n]['arc_force_change_n'])/5)**2 for n in ARC])
            candidates.append({'radius_m':radius,'cost':float(cost),'predictions':predictions})
            print('M2 radius fit',radius,float(cost),flush=True)
        best=min(candidates,key=lambda x:x['cost']);model('robot_m2',radius=best['radius_m'],latch=True,threshold=threshold)
        held=[]
        for name,radius,angle in [('heldout_arc_long',.75,.30),('heldout_arc_short',.65,.10)]:
            w=summarize_arc(raw_arc(name,radius,angle))
            rec={'probe':name,'world':w}
            for v in ('robot_m1','robot_m2'):
                rec[v]=summarize_arc(sim(name,v,out/'m2/heldout'/v/name))
            held.append(rec)
        mae={v:float(np.mean([abs(rec[v]['arc_force_change_n']-rec['world']['arc_force_change_n']) for rec in held])) for v in ('robot_m1','robot_m2')}
        # Choose a task program from the Sim's response ranking, then validate in Sim.
        preferred=min(ARC,key=lambda n:best['predictions'][n]['arc_force_change_n'])
        selected_pull_radius=ARC[preferred]
        policy=write_arc_probe(demo,programs/'reference_grounded_turn_pull.py',radius=selected_pull_radius,rotate=.50,angle=.45,pull_duration=4.)
        if not (out/'m2/task_in_sim').exists():run_sim_episode(policy,out/'m2/task_in_sim',variant='robot_m2',observations='visual')
        task=json.loads((out/'m2/task_in_sim/metrics.json').read_text())
        result={'stage':'M2-R','fitted_effective_radius_m':best['radius_m'],
                'selected_program_pull_radius_m':selected_pull_radius,'fit_world':world,'candidates':candidates,
                'heldout':held,'heldout_force_change_mae_n':mae,'sim_task_success':task['success'],
                'sim_task_door_angle_rad':task.get('final_door_angle_rad'),
                'completed':mae['robot_m2']<mae['robot_m1'] and task['success'],
                'world_transfer':'pending separate frozen evaluation',
                'scope':'task-conditioned interaction model; not hidden lock identification'}
        write_json(out/'m2/completion.json',result);assert result['completed']
        mark('M2',out/'m2/completion.json');print('M2 Sim update completed',mae,flush=True)
    write_manifest()

if __name__=='__main__':main()
