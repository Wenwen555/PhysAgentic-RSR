"""Standalone geometric and force checks; no robot policy is executed."""
from pathlib import Path
import json
import mujoco
import numpy as np
m=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parent/'scene.xml'))
d=mujoco.MjData(m)
def reset():
 mujoco.mj_resetDataKeyframe(m,d,m.key('home').id);mujoco.mj_forward(m,d)
reset()
normals=[d.xmat[m.body(n+'_frame').id].reshape(3,3)[:,2] for n in ('lever','round')]
angle=float(np.rad2deg(np.arccos(np.clip(abs(np.dot(*normals)),0,1))))
assert abs(angle-90)<1e-6
origtype=m.geom_contype.copy();origaff=m.geom_conaffinity.copy()
for i in range(m.ngeom):
 n=m.geom(i).name or ''
 if (n.startswith(('lever_','round_')) and n.endswith(('_collision','_left','_right','_header','_strike'))) or (n.startswith(('robot0_','gripper0_')) and m.geom_group[i]==3) or n=='robot_pedestal':m.geom_contype[i]=m.geom_conaffinity[i]=1
for n in ('lever','round'):
 d.joint(n+'_handle_hinge').qpos[0]=.5;d.joint(n+'_bolt_joint').qpos[0]=.0225
category=lambda n:'lever' if n.startswith('lever_') else 'round' if n.startswith('round_') else 'robot' if n.startswith(('robot0_','gripper0_')) or n=='robot_pedestal' else 'other'
angles=np.linspace(0,1.5,51);bad=[]
for a in angles:
 d.joint('lever_door_hinge').qpos[0]=a
 for b in angles:
  d.joint('round_door_hinge').qpos[0]=b;mujoco.mj_forward(m,d)
  for c in d.contact:
   n1=m.geom(c.geom1).name or '';n2=m.geom(c.geom2).name or '';pair=tuple(sorted((category(n1),category(n2))))
   if c.dist < -1e-5 and pair in [('lever','round'),('lever','robot'),('robot','round')]:bad.append([n1,n2,float(c.dist),float(a),float(b)])
assert not bad,bad[:5]
m.geom_contype[:]=origtype;m.geom_conaffinity[:]=origaff
arm=np.array([m.joint(f'robot0_joint{i}').dofadr[0] for i in range(1,8)])
results=[]
for mode in ('lever_locked','round_locked','lever','round','both'):
 reset();names=['lever','round'] if mode=='both' else [mode.replace('_locked','')];locked=mode.endswith('_locked');cross=set();strike=set()
 for _ in range(2500):
  d.qfrc_applied[arm]=d.qfrc_bias[arm]
  for n in names:
   d.qfrc_applied[m.joint(n+'_handle_hinge').dofadr[0]]=0 if locked else .6
   d.qfrc_applied[m.joint(n+'_door_hinge').dofadr[0]]=4
  mujoco.mj_step(m,d)
  for c in d.contact:
   n1=m.geom(c.geom1).name or '';n2=m.geom(c.geom2).name or '';pair=tuple(sorted((category(n1),category(n2))))
   if pair in [('lever','round'),('lever','robot'),('robot','round')]:cross.add((n1,n2))
   if (n1.endswith('_strike') and n2.endswith('_bolt_collision')) or (n2.endswith('_strike') and n1.endswith('_bolt_collision')):strike.add((n1,n2))
 vals={n:float(d.joint(n+'_door_hinge').qpos[0]) for n in names}
 assert not cross,(mode,cross)
 if locked:assert vals[names[0]]<.02 and strike,(mode,vals,strike)
 else:assert all(v>1.4 for v in vals.values()),(mode,vals)
 results.append({'mode':mode,'door_angles_rad':vals,'cross_contacts':len(cross),'strike_contacts':len(strike)})
print(json.dumps({'plane_angle_deg':angle,'sampled_configurations':len(angles)**2,'inter_object_collisions':len(bad),'diagnostic_motion':results},ensure_ascii=False,indent=2))
