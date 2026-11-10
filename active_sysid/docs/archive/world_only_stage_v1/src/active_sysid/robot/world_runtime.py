"""Robot-space runtime for the original World; exposes the same motion API."""
from .runtime import RobotRuntime

class WorldRobotRuntime(RobotRuntime):
    def __init__(self, backend, *, observations='robot',
                 visual_cameras=('overview', 'robot0_eye_in_hand'), **controller_options):
        super().__init__(backend, observations=observations, **controller_options)
        self.visual_cameras = tuple(visual_cameras)
        self.rgb_observations = []

    def observe(self, *, rgb=False, camera='overview'):
        b=self._backend
        s=b.sample()
        keys=('time_s','q','dq','joint_torque_nm','wrist_wrench_ee','ee_pos_m',
              'ee_quat_wxyz','gripper_width_m','finger_contact_normal_n',
              'handle_contact_force_world_n','total_handle_contact_force_n',
              'unexpected_contact_force_n','self_contact_force_n')
        result={k:s[k] for k in keys}
        if self.observations=='privileged': result['privileged']=b.privileged()
        if rgb:
            if self.observations == 'visual' and camera not in self.visual_cameras:
                raise ValueError(f'visual stage camera not allowed: {camera}')
            if self._renderer is None:
                import mujoco
                self._renderer=mujoco.Renderer(b.model,height=480,width=640)
            self._renderer.update_scene(b.data,camera=camera)
            result['rgb']=self._renderer.render().copy()
            self.rgb_observations.append({'time_s':float(s['time_s']), 'camera':camera})
        return result
