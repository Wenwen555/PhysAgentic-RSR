import numpy as np
PRE = [-1.616539341642199, 0.6086180292957257, -0.7438553384107736, -2.136267320729601, -2.0828396180853583, 2.1901220205231633, -1.0041841335714556]
TARGET = [0.5612639663413977, 0.41988714354912265, 0.9969942052419122]
QUAT = [0.00022388830896791212, 0.7070507729049974, -0.00015117108560019901, 0.7071627334325332]
ANGLE = 0.5
PULL = 0.0
WIDTH = 0.01
OFFSET = [0.01, 0.07, 0.04]
DURATION = 1.6

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
    api.phase('pull_arc')
    s=api.observe()
    center=np.asarray(s['ee_pos_m'])+0.75*(-frame[:,1])+.06*frame[:,2]
    api.move_arc(center,-frame[:,0],0.45,duration_s=4.0)
    api.hold(.3)
    api.observe(rgb=True,camera='overview')
