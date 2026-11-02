"""Neutral rod pose geometry shared by current and historical adapters."""
import numpy as np
from scipy.spatial.transform import Rotation as R

def rod_pose(env) -> tuple[np.ndarray, np.ndarray]:
    """Return (centre, rod_x_axis) of the rod, expressed in the robot base frame."""
    pose = env.get_observation()["cube_poses"]["primary"]
    centre = np.asarray(pose[:3], dtype=np.float64)
    quat = np.asarray(pose[3:], dtype=np.float64)  # wxyz
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # xyzw
    rod_x = rot.apply([1.0, 0.0, 0.0])
    return centre, rod_x


def down_grasp_quat(env) -> np.ndarray:
    """``DOWN_WXYZ`` yawed so the fingers close across the rod's *short* axis.

    The Panda fingers translate along the hand y axis, which under
    ``DOWN_WXYZ`` (= 180 deg about y) is world +y. If the rod is not aligned
    with world x, the fingers close *along* the rod instead of across its 3 cm
    width: they meet on the rod's top surface and pinch its edge, which is the
    dominant failure mode of the original candidates. Rotating about the
    vertical by the rod's yaw maps the closing axis onto the rod's short axis.

    Returns:
        (4,) quaternion in wxyz.
    """
    _, rod_x = rod_pose(env)
    yaw = float(np.arctan2(rod_x[1], rod_x[0]))
    rot = R.from_euler("z", yaw) * R.from_quat([0.0, 1.0, 0.0, 0.0])  # xyzw
    q_xyzw = rot.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
