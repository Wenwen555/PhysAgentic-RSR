"""Trajectory observations, not causal diagnoses. All geometry is robot-base."""
import numpy as np


def completion_error(samples):
    return None if not samples else abs(.10-(samples[-1]['rod_position'][2]-samples[0]['rod_position'][2]))


def world_to_base(position, base_position, base_rotation):
    return (np.asarray(base_rotation).reshape(3, 3).T @
            (np.asarray(position)-np.asarray(base_position))).tolist()


def summarize(samples, record, window=3):
    if not samples:
        return dict(coordinate_frame='robot_base', execution_summary=dict(
            terminal_status=record['status'], failure_events=['program_error'] if record['status']=='error' else []),
            milestones={}, first_failure_event=None, trajectory_evidence=[],
            possible_causes=[], diagnostic_uncertainty=['No trajectory available'])
    assert all(s['coordinate_frame']=='robot_base' for s in samples)
    z0=samples[0]['rod_position'][2]
    rises=[s['rod_position'][2]-z0 for s in samples]
    grasp=[s['grasped'] for s in samples]
    first=lambda pred: next((i for i,s in enumerate(samples) if pred(i,s)),None)
    established=first(lambda i,s: grasp[i])
    # The task uses tabletop-relative clearance, not 4cm relative to the rod's
    # initial centre (which already rests above the tabletop).
    lifted=first(lambda i,s: s['lifted'])
    target=first(lambda i,s: grasp[i] and rises[i]>=.095)
    tilt=first(lambda i,s: established is not None and i>=established and s['rod_tilt']>.25)
    lost=first(lambda i,s: established is not None and i>established and not grasp[i])
    drop=first(lambda i,s: lifted is not None and i>lifted and not s['lifted'] and not grasp[i])
    releases=[i for i,s in enumerate(samples) if s.get('event')=='open_gripper:before']
    unsafe=[i for i in releases if grasp[i] and samples[i]['lifted']]
    violations=[(i,k) for i,k in ((tilt,'excessive_tilt'),(lost,'grasp_lost'),
                                 (drop,'lifted_then_dropped')) if i is not None]
    violations += [(i,'release_while_lifted') for i in unsafe]
    tags=[]
    if established is None: tags.append('never_grasped')
    if lifted is None: tags.append('never_lifted')
    tags += list(dict.fromkeys(k for i,k in sorted(violations)))
    if target is None: tags.append('target_not_reached')
    if record['status']=='error':
        tags.append('execution_timeout' if any(x in record.get('reason','').lower() for x in ('time','budget')) else 'program_error')
    def event(index):
        return None if index is None else dict(sample=index,time_s=samples[index]['time_s'])
    first_bad=min(violations) if violations else None
    indices={0,len(samples)-1,int(np.argmax(rises))}
    for i in [established,lifted,target,tilt,lost,drop]+releases:
        if i is not None: indices.update(range(max(0,i-window),min(len(samples),i+window+1)))
    # Relative displacement referenced to each close, in the SAME base frame.
    reference=None;slips=[]
    for s in samples:
        relative=np.asarray(s['rod_position'])-s['tcp_position']
        if s.get('event')=='close_gripper:after': reference=relative
        slips.append(None if reference is None else float(np.linalg.norm(relative-reference)))
    evidence=[dict(sample=i,**samples[i],rise_m=rises[i],relative_displacement_since_close_m=slips[i]) for i in sorted(indices)]
    return dict(coordinate_frame='robot_base', raw_metadata=dict(
        position_source='base-transformed MuJoCo body pose; base TCP',
        time_source='MuJoCo simulation clock; every control step plus API boundaries',
        grasp_source='simulator contact/grasp predicate; diagnostic, not agent runtime API',
        lift_source='unchanged task tabletop-clearance predicate; not a separate relative-rise threshold',
        command_source='last requested policy API pose in base; not the internal joint servo target',
        sample_count=len(samples)),
        execution_summary=dict(terminal_status=record['status'],terminal_reason=record['reason'],
                               failure_events=tags,execution_s=record.get('sim_duration_s')),
        milestones=dict(grasp_established=established is not None,ever_lifted=lifted is not None,
            max_lift_height_m=max(rises),max_lift=event(int(np.argmax(rises))),
            first_grasp=event(established),first_lift=event(lifted),first_target=event(target),
            first_excessive_tilt=event(tilt),first_grasp_loss=event(lost),first_drop=event(drop),
            max_tilt=max(s['rod_tilt'] for s in samples),
            max_relative_displacement_since_close_m=max([s for s in slips if s is not None],default=None),
            open_events=[dict(**event(i),while_lifted=i in unsafe) for i in releases],
            close_events=[event(i) for i,s in enumerate(samples) if s.get('event')=='close_gripper:after']),
        first_failure_event=None if first_bad is None else dict(type=first_bad[1],**event(first_bad[0])),
        trajectory_evidence=evidence,possible_causes=[],diagnostic_uncertainty=[
            'Events are sampled at control-step/API boundaries, not continuous contact time.',
            'Relative displacement includes rotation; it is not pure contact slip.',
            'No unique cause inferred. Review mathematics, control and contact hypotheses separately.',
            'never_* and target_not_reached are whole-trajectory outcomes, not first-failure times.'])
