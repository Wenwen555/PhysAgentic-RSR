"""Server-side robot API. World observations are sensor-only, always."""
from pathlib import Path
import numpy as np
import mujoco
from .adapter import RobotExecutionAdapter
from .sensors import robot_observation

ACTIONS={'move_joints','move_ee','move_arc','set_gripper','hold'}


class BoundaryRuntime:
    def __init__(self,backend,*,role,observations,max_steps=30000,image_dir=None,
                 cameras=('overview','robot0_eye_in_hand')):
        if role not in ('development_sim','world_probe','world_evaluation'):
            raise ValueError('Explicit execution role is required')
        if observations not in ('privileged','visual','robot'):
            raise ValueError('Unknown observation contract')
        if role!='development_sim' and observations=='privileged':
            raise ValueError('World truth is not a policy/probe observation')
        if role=='development_sim' and getattr(backend,'role',None)!='development_sim':
            raise ValueError('Development requires independent Sim backend')
        self.backend=backend;self.role=role;self.observations=observations
        self.adapter=RobotExecutionAdapter(backend,max_steps=max_steps,
            contact_force_limit_n=80.0,unexpected_contact_limit_n=5.0)
        self.actions=[];self.rgb_observations=[];self.renderer=None
        self.image_dir=Path(image_dir) if image_dir is not None else None
        self.cameras=tuple(cameras)

    def observe(self,*,rgb=False,camera='overview'):
        observation=robot_observation(self.backend)
        if self.observations=='privileged':observation['privileged']=self.backend.privileged()
        if rgb:
            if camera not in self.cameras:raise ValueError('Camera not allowed by observation contract')
            if self.renderer is None:self.renderer=mujoco.Renderer(self.backend.model,height=480,width=640)
            self.renderer.update_scene(self.backend.data,camera=camera)
            pixels=self.renderer.render().copy();observation['rgb']=pixels
            entry={'time_s':observation['time_s'],'camera':camera}
            if self.image_dir is not None:
                import imageio.v2 as iio
                self.image_dir.mkdir(parents=True,exist_ok=True)
                filename=f'{len(self.rgb_observations):05d}_{camera}.png'
                iio.imwrite(self.image_dir/filename,pixels);entry['file']=f'rgb/{filename}'
            self.rgb_observations.append(entry)
        return observation

    def dispatch(self,method,args,kwargs):
        if method=='observe':return self.observe(*args,**kwargs)
        if method=='phase':
            if len(args)!=1 or not isinstance(args[0],str) or kwargs:raise ValueError('phase expects a name')
            self.backend.phase=args[0];return None
        if method not in ACTIONS:raise ValueError('Unknown Robot API method')
        entry={'action':method,'args':args,'kwargs':kwargs,'phase':self.backend.phase,
               'start_time_s':float(self.backend.data.time)}
        self.actions.append(entry)
        try:
            getattr(self.adapter,method)(*args,**kwargs)
            entry['status']='completed'
        except Exception:
            entry['status']='failed';raise
        finally:entry['end_time_s']=float(self.backend.data.time)
        return self.observe()

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
