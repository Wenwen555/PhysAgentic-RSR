"""Thin mapping from robot-space commands to the motion controller."""
from __future__ import annotations
from active_sysid.cap.robot_controller import RobotController, quat_to_matrix


class RobotExecutionAdapter:
    def __init__(self, backend, **controller_options):
        self.controller = RobotController(backend, **controller_options)

    def move_ee(self, position_m, quaternion_wxyz, duration_s):
        return self.controller.move_ee(position_m, quat_to_matrix(quaternion_wxyz), duration_s)

    def move_arc(self, center_m, axis_world, angle_rad, duration_s):
        return self.controller.move_arc(center_m, axis_world, angle_rad, duration_s)

    def set_gripper(self, width_m, duration_s):
        return self.controller.set_gripper(width_m, duration_s)

    def hold(self, duration_s):
        return self.controller.hold(duration_s)
