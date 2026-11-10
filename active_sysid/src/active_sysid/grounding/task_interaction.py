"""Minimal robot task grounding: sensor-only probes, task rule, contact model.

The contact demonstration supplies robot waypoints, not hidden object state.
No real lock/bolt identification is attempted.
"""
from pathlib import Path
import json
import numpy as np
from active_sysid.robot.controller import quat_to_matrix


def write_probe(demo, path, *, rotate=0.0, pull=0.015, width=0.010,
                pivot_offset=(0.010,0.070,0.040), duration=1.6):
    """Self-contained Robot API program. Offsets are in demonstrated tool frame."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    constants={'PRE':demo['pregrasp']['q'],'TARGET':demo['reach']['ee_pos_m'],
               'QUAT':demo['reach']['ee_quat_wxyz'],'ANGLE':float(rotate),
               'PULL':float(pull),'WIDTH':float(width),'OFFSET':list(pivot_offset),
               'DURATION':float(duration)}
    source='import numpy as np\n'+''.join(f'{key} = {value!r}\n' for key,value in constants.items())
    source+='''
def matrix(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def run(api):
    api.phase('settle'); api.hold(.4)
    api.observe(rgb=True,camera='overview')
    api.phase('pregrasp'); api.move_joints(PRE,duration_s=4.5)
    api.phase('reach'); api.move_ee(TARGET,QUAT,duration_s=2.0)
    api.phase('grasp'); api.set_gripper(WIDTH,duration_s=1.2)
    observation=api.hold(.4)
    frame=matrix(QUAT)
    if min(observation['finger_contact_normal_n']) < .5:
        raise RuntimeError('Contact not established')
    api.observe(rgb=True,camera='robot0_eye_in_hand')
    if ANGLE:
        api.phase('rotate')
        pivot=np.asarray(observation['ee_pos_m'])+frame@np.asarray(OFFSET)
        api.move_arc(pivot,-frame[:,2],ANGLE,duration_s=2.8)
        api.hold(.2)
    observation=api.observe()
    if PULL:
        api.phase('pull')
        target=np.asarray(observation['ee_pos_m'])-PULL*frame[:,2]
        api.move_ee(target,observation['ee_quat_wxyz'],duration_s=DURATION)
        api.hold(.4)
    api.observe(rgb=True,camera='overview')
'''
    path.write_text(source)
    return path


def summarize_probe(directory):
    directory=Path(directory)
    public=(directory/'probe.json').exists()
    meta=json.loads((directory/('probe.json' if public else 'metrics.json')).read_text())
    actions=json.loads((directory/'actions.json').read_text())
    with np.load(directory/('observations.npz' if public else 'trajectory.npz')) as data:
        time=data['time_s'];grip=data['finger_contact_normal_n']
        def tail(phase):
            selected=[a for a in actions if a['phase']==phase]
            if not selected:return None
            end=selected[-1]['end_time_s']
            return (time>max(selected[0]['start_time_s'],end-.15))&(time<=end+1e-8)
        grasp=tail('grasp');pull=tail('pull');rotate=tail('rotate')
        out={'directory':str(directory),'completed':meta['execution_error'] is None,
             'error':meta['execution_error'],'steps':meta['physics_steps']}
        if grasp is None:return out
        q=data['ee_quat_wxyz'][np.flatnonzero(grasp)[-1]];R=quat_to_matrix(q)
        out.update(grasp_tcp_m=data['ee_pos_m'][grasp].mean(axis=0).tolist(),
                   gripper_width_m=float(data['gripper_width_m'][grasp].mean()),
                   grasp_bilateral_ratio=float(np.mean(np.all(grip[grasp]>.5,axis=1))))
        if pull is not None:
            command=next(a for a in actions if a['phase']=='pull' and a['action']=='move_ee')
            start_index=np.searchsorted(time,command['start_time_s'])-1
            start=data['ee_pos_m'][start_index]
            end=data['ee_pos_m'][pull].mean(axis=0)
            target=np.array(command['args'][0])
            delta=target-start;direction=delta/np.linalg.norm(delta)
            out.update(commanded_pull_m=float(np.linalg.norm(delta)),
                       achieved_pull_m=float((end-start)@direction),
                       tracking_ratio=float((end-start)@direction/np.linalg.norm(delta)),
                       endpoint_error_m=float(np.linalg.norm(target-end)),
                       pull_bilateral_ratio=float(np.mean(np.all(grip[pull]>.5,axis=1))),
                       pull_tcp_m=end.tolist())
            # Wrist force is a real robot sensor; subtract the pre-pull bias and
            # rotate each reading into the fixed base frame before comparison.
            before=(time>command['start_time_s']-.15)&(time<=command['start_time_s']+1e-8)
            def world_force(mask):
                return np.array([quat_to_matrix(q)@f for q,f in zip(data['ee_quat_wxyz'][mask],data['wrist_wrench_ee'][mask,:3])])
            out['delta_wrist_force_n']=float(np.linalg.norm(world_force(pull).mean(axis=0)-world_force(before).mean(axis=0)))
        if rotate is not None:
            out['rotate_tcp_m']=data['ee_pos_m'][rotate].mean(axis=0).tolist()
            out['rotate_bilateral_ratio']=float(np.mean(np.all(grip[rotate]>.5,axis=1)))
    return out


def write_arc_probe(demo,path,*,radius=.75,angle=.20,rotate=.50,pull_duration=2.5):
    write_probe(demo,path,rotate=rotate,pull=0)
    path=Path(path);source=path.read_text()
    marker="    api.observe(rgb=True,camera='overview')\n"
    last=source.rfind(marker)
    source=source[:last]+f'''    api.phase('pull_arc')
    s=api.observe()
    center=np.asarray(s['ee_pos_m'])+{radius!r}*(-frame[:,1])+.06*frame[:,2]
    api.move_arc(center,-frame[:,0],{angle!r},duration_s={pull_duration!r})
    api.hold(.3)
'''+source[last:]
    path.write_text(source);return path


def summarize_arc(directory):
    directory=Path(directory);report=summarize_probe(directory)
    actions=json.loads((directory/'actions.json').read_text())
    cmd=next((a for a in actions if a['phase']=='pull_arc' and a['action']=='move_arc'),None)
    if cmd is None:return report
    with np.load(directory/('observations.npz' if (directory/'probe.json').exists() else 'trajectory.npz')) as d:
        t=d['time_s'];t0=cmd['start_time_s'];tail=(t>t[-1]-.15)
        before=(t>t0-.15)&(t<=t0+1e-8)
        R=np.array([quat_to_matrix(q) for q in d['ee_quat_wxyz']])
        forces=np.einsum('nij,nj->ni',R,d['wrist_wrench_ee'][:,:3])
        moments=np.einsum('nij,nj->ni',R,d['wrist_wrench_ee'][:,3:])
        report.update(arc_force_change_n=float(np.linalg.norm(forces[tail].mean(axis=0)-forces[before].mean(axis=0))),
                      arc_torque_change_nm=float(np.linalg.norm(moments[tail].mean(axis=0)-moments[before].mean(axis=0))),
                      arc_bilateral_ratio=float(np.mean(np.all(d['finger_contact_normal_n'][t>t0]>.5,axis=1))),
                      arc_tcp_m=d['ee_pos_m'][tail].mean(axis=0).tolist())
    return report


def build_task_sim(demo, contact, output, *, radius=.80, stiffness=.2, damping=.05,
                   friction=1.0, door_damping=.1, door_friction=.05, latch=False,
                   threshold=.125, mass=6.0):
    """Task-local contact proxy fitted to robot evidence; no World scene reads.

    Visual reconstruction supplies the robot model and coarse door dimensions.
    The robot contact demonstration supplies placement and measured grasp width.
    A functional condition represents the task dependency, not the native bolt.
    """
    import copy
    import xml.etree.ElementTree as ET
    from active_sysid.robot.fixture_backend import ROOT
    from active_sysid.robot.controller import matrix_to_quat
    source=ROOT/'assets/reconstructed_sim/visual_stage3_v1/scene.xml'
    original=ET.parse(source).getroot()
    root=ET.Element('mujoco',model='robot_probe_task_surrogate')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true',
        meshdir='../visual_stage3_v1',texturedir='../visual_stage3_v1')
    ET.SubElement(root,'option',timestep='.002',integrator='implicitfast',cone='elliptic',iterations='80')
    root.append(copy.deepcopy(original.find('asset')))
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
    wb=ET.SubElement(root,'worldbody')
    common=original.find(".//body[@name='common_world_frame']")
    mount=ET.SubElement(wb,'body',name='common_world_frame',**{k:v for k,v in common.attrib.items() if k!='name'})
    mount.append(copy.deepcopy(common.find("body[@name='panda_mount']")))
    for tag in ('actuator','sensor'):root.append(copy.deepcopy(original.find(tag)))
    def fmt(values):return ' '.join(f'{v:.10g}' for v in np.asarray(values).ravel())
    R=quat_to_matrix(demo['reach']['ee_quat_wxyz']);forward,along,up=R[:,2],-R[:,1],R[:,0]
    B=np.column_stack([forward,along,up])
    grip=np.asarray(contact['grasp_tcp_m']);pad=grip-.008*forward
    pivot=grip+R@np.array([.010,.070,.040])
    hinge=pad+.060*forward+radius*along
    # Coarse dimensions are from reconstruction, not World geometry.
    panel=original.find(".//geom[@name='LeverDoor_Leaf_collision']")
    size=np.fromstring(panel.get('size'),sep=' ');width=2*size[0];height=2*size[2];thick=2*size[1]
    handle_local=np.fromstring(original.find(".//body[@name='LeverDoor_LeverHub']").get('pos'),sep=' ')
    panel_height_offset=-handle_local[2]
    frame=ET.SubElement(wb,'body',name='task_frame')
    door=ET.SubElement(frame,'body',name='task_door',pos=fmt(hinge),quat=fmt(matrix_to_quat(B)))
    ET.SubElement(door,'joint',name='task_door_hinge',axis='0 0 -1',range='0 1.5',
                  damping=str(door_damping),frictionloss=str(door_friction))
    ET.SubElement(door,'geom',name='task_panel',type='box',pos=fmt([0,-width/2,panel_height_offset]),
                  size=fmt([thick/2,width/2,height/2]),mass=str(mass),rgba='.23 .12 .06 1',
                  contype='1',conaffinity='1')
    handle=ET.SubElement(door,'body',name='task_handle',pos=fmt(B.T@(pivot-hinge)))
    ET.SubElement(handle,'inertial',pos='0 0 0',mass='.08',diaginertia='.0001 .0001 .0001')
    ET.SubElement(handle,'joint',name='task_handle_hinge',axis='-1 0 0',range='-.3 1.5',
                  stiffness=str(stiffness),damping=str(damping),armature='.005')
    c=B.T@(pad-pivot);bar_radius=(contact['gripper_width_m']-.001)/2
    ET.SubElement(handle,'geom',name='task_handle_contact',type='capsule',
                  fromto=fmt(np.r_[c+[0,-.055,0],c+[0,.055,0]]),size=str(bar_radius),
                  friction=f'{friction} .005 .0001',condim='4',solref='.01 1',
                  contype='1',conaffinity='1',rgba='.8 .6 .2 1')
    ET.SubElement(wb,'geom',name='ground',type='plane',size='3 3 .1',rgba='.3 .3 .3 1')
    ET.SubElement(wb,'light',pos=fmt(grip+[-1,0,2]),dir='0 0 -1',diffuse='.8 .8 .8')
    eye=grip+np.array([-1.6,1.1,1.3]);z=eye-grip;z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    ET.SubElement(wb,'camera',name='overview',pos=fmt(eye),xyaxes=fmt(np.r_[x,y]),fovy='50')
    eq=ET.SubElement(root,'equality')
    ET.SubElement(eq,'weld',name='sim_latch_lever',body1='task_door',body2='task_frame',
                  active=str(latch).lower(),solref='.01 1',solimp='.98 .999 .001')
    # Source contains four reconstructed object joints before the robot joints;
    # q home is read only from the reconstructed robot's joint indices.
    import mujoco as mj
    model=mj.MjModel.from_xml_path(str(source));key=model.key('stage2_initial').id
    q=[model.key_qpos[key,model.joint(f'robot0_joint{i}').qposadr[0]] for i in range(1,8)]+[0,0,0,0]
    keyframe=ET.SubElement(root,'keyframe');ET.SubElement(keyframe,'key',name='stage2_initial',qpos=fmt(q))
    ET.indent(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(output,encoding='unicode')
    params={'latch_enabled':bool(latch),'unlock_threshold':float(threshold),
      'model_type':'task_condition_contact_surrogate','door_radius_m':float(radius),
      'handle_stiffness':float(stiffness),'handle_damping':float(damping),
      'handle_contact_friction':float(friction),'door_damping':float(door_damping),
      'door_friction':float(door_friction),'door_mass_prior_kg':float(mass),
      'contact_radius_m':float(bar_radius),'grasp_tcp_m':grip.tolist(),
      'handle_pivot_prior_m':pivot.tolist(),'door_pivot_estimate_m':hinge.tolist(),
      'geometry_source':'coarse reconstruction + robot demonstration + public contact probe',
      'hidden_world_structure_copied':False}
    output.with_suffix('.json').write_text(json.dumps(params,indent=2)+'\n')
    return params
