"""Explicit service factories. Hardware tracking stays behind the public API."""
from __future__ import annotations

from ..api.client import APIError
from .robot import RobotService, TargetService


class SimulatedRobotService:
    """Private target world; its physics is never part of a loop request."""
    def __init__(self, private_config):
        from physrsr.physics_spec import PhysicsSpec
        self.physics = PhysicsSpec(**private_config["physics"])
        self.seeds = tuple(private_config["seeds"])
        if not self.seeds:
            raise ValueError("Private target reset seeds required")
        self.index, self.worker, self.ready = 0, None, False

    def ensure(self):
        if self.worker is None:
            from physrsr.worker import RolloutWorker
            self.worker = RolloutWorker(self.physics)
        env, api = self.worker._ensure_env()
        env.robosuite_env.ignore_done = True
        return env, api

    def prepare(self):
        if self.index >= len(self.seeds):
            raise APIError("Private target reset seeds exhausted")
        env, api = self.ensure()
        seed = int(self.seeds[self.index])
        self.index += 1
        api.cfg = None
        env.robosuite_env.seed_episode(seed)
        env.reset(seed=seed)
        self.ready = True

    def call(self, method, params):
        from .simulator import observation_provider
        env, api = self.ensure()
        if not self.ready and method != "episode.prepare":
            self.prepare()
        return RobotService(api, lambda: observation_provider(env), self.prepare).call(method, params)

    def close(self):
        if self.worker is not None:
            self.worker.close()


def measured_franka_robot(control, env, tracker, setup):
    """Real-arm adapter using an existing controller plus API pose tracking.

    tracker.call('perception.rod', {}) must return measured base-frame rod pose,
    axis, grasp orientation and capture time. TCP comes from env.get_eef_pose()
    (encoder FK). setup.call('episode.prepare', {}) must establish a supported
    start and return ready=true. No simulator-derived hardware observation.
    """
    def observe():
        result = tracker.call("perception.rod", {})
        # Refresh the low-level encoder stream before FK when supported.
        env.get_observation()
        _, tcp = env.get_eef_pose()
        return dict(rod_position=result["rod_position"], rod_axis=result["rod_axis"],
                    grasp_quat=result["grasp_quat"], tcp_position=tcp.tolist(),
                    coordinate_frame=result["coordinate_frame"], time_s=result["time_s"])

    def prepare():
        response = setup.call("episode.prepare", {})
        if response.get("ready") is not True:
            raise APIError("Real task setup is not ready")
        if response.get("grip_force_per_finger_n") != 2.0:
            raise APIError("Setup API must establish and acknowledge the shared 2 N per-finger limit")
        control.cfg = None
        return dict(cost_s=response["cost_s"])

    return RobotService(control, observe, prepare)
