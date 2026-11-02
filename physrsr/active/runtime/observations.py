"""Geometry projection with no experiment or historical protocol dependency."""
from .robot import public_observation


def observation_provider(env):
    # Trusted backend adapter only. Generated code never receives env or MJCF.
    from physrsr.geometry import rod_pose, down_grasp_quat
    position, axis = rod_pose(env)
    _, tcp = env.get_eef_pose()
    return public_observation(dict(rod_position=position.tolist(), rod_axis=axis.tolist(),
        grasp_quat=down_grasp_quat(env).tolist(), tcp_position=tcp.tolist(),
        coordinate_frame="robot_base", time_s=float(env.robosuite_env.sim.data.time)))
