"""Seeded perturbations of DEVELOPMENT SIM only. Never randomize reference World."""
import numpy as np
import mujoco


def apply_sim_randomization(backend,seed,ranges):
    if getattr(backend,'role',None)!='development_sim':
        raise ValueError('Randomization is restricted to development_sim')
    allowed={'door_damping','door_frictionloss','handle_damping','handle_stiffness',
             'handle_contact_friction','illumination'}
    if set(ranges)-allowed:raise ValueError('Unknown Sim randomization parameter')
    rng=np.random.default_rng(seed);m=backend.model
    draw={name:float(rng.uniform(*bounds)) for name,bounds in sorted(ranges.items())}
    for key,jid,field in [('door_damping',backend.door_jid,'dof_damping'),
                           ('door_frictionloss',backend.door_jid,'dof_frictionloss'),
                           ('handle_damping',backend.handle_jid,'dof_damping')]:
        if key in draw:getattr(m,field)[m.jnt_dofadr[jid]]*=draw[key]
    if 'handle_stiffness' in draw:m.jnt_stiffness[backend.handle_jid]*=draw['handle_stiffness']
    if 'handle_contact_friction' in draw:m.geom_friction[backend.handle_geom,0]*=draw['handle_contact_friction']
    if 'illumination' in draw:m.light_diffuse[:]=np.clip(m.light_diffuse*draw['illumination'],0,1)
    mujoco.mj_forward(m,backend.data)
    return draw
