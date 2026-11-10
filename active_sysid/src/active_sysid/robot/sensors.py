"""Robot sensors with no object identity, scene parameters or mechanism truth."""
import numpy as np
import mujoco as mj
from .controller import matrix_to_quat


def robot_observation(b):
    m,d=b.model,b.data
    p,r=b.tcp_pose()
    finger_normal=np.zeros(2)
    for i in range(d.ncon):
        c=d.contact[i];g1,g2=int(c.geom1),int(c.geom2)
        f=np.zeros(6);mj.mj_contactForce(m,d,i,f)
        for side,geoms in enumerate(b.finger_geom_sets):
            other=g2 if g1 in geoms else g1 if g2 in geoms else None
            if other is not None and int(m.geom_bodyid[other]) not in b.robot_bodies:
                finger_normal[side]+=max(float(f[0]),0.0)
    return {'time_s':float(d.time),'q':d.qpos[b.arm_qpos].copy(),
            'dq':d.qvel[b.arm_dofs].copy(),'joint_torque_nm':d.actuator_force[b.arm_actuators].copy(),
            'ee_pos_m':p,'ee_quat_wxyz':matrix_to_quat(r),
            'gripper_width_m':float(d.qpos[b.finger_qpos[0]]-d.qpos[b.finger_qpos[1]]),
            'wrist_wrench_ee':np.r_[d.sensor(b.cfg['wrist_force_sensor']).data,
                                    d.sensor(b.cfg['wrist_torque_sensor']).data].copy(),
            'finger_contact_normal_n':finger_normal}


def contact_audit(b):
    """Private contact identities used only for controller cutoffs/evaluation."""
    m,d=b.model,b.data
    target=unexpected=self_force=0.0
    for i in range(d.ncon):
        c=d.contact[i];g1,g2=int(c.geom1),int(c.geom2)
        r1=int(m.geom_bodyid[g1]) in b.robot_bodies
        r2=int(m.geom_bodyid[g2]) in b.robot_bodies
        if not (r1 or r2):continue
        f=np.zeros(6);mj.mj_contactForce(m,d,i,f);v=float(np.linalg.norm(f[:3]))
        if r1 and r2:self_force=max(self_force,v)
        elif b.handle_geom in (g1,g2):
            other=g2 if g1==b.handle_geom else g1
            if any(other in gs for gs in b.finger_geom_sets):target+=v
            else:unexpected=max(unexpected,v)
        else:unexpected=max(unexpected,v)
    return {'total_handle_contact_force_n':target,'unexpected_contact_force_n':unexpected,
            'self_contact_force_n':self_force}
