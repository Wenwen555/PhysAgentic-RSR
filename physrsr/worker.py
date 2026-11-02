"""Shared rollout worker: owns a rod-lift env + control API, runs a program."""

from __future__ import annotations

import time

import numpy as np

from capx.envs.simulators.robosuite_rod_lift import FrankaRobosuiteRodLiftLowLevel
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from physrsr.phase_tracker import PhaseTracker
from physrsr.physics_spec import PhysicsSpec, RolloutRecord
from physrsr.paths import CAPX_ROOT


class RolloutWorker:
    """Executes candidate programs in the rod simulator and reports success.

    A program's success is evaluated *after* it finishes by holding the gripper
    closed and stepping the simulation for ``hold_steps`` control steps,
    requiring the rod to remain lifted, grasped and level throughout — i.e. no
    slip / drop during the hold (the Step 3 stable-hold condition).

    Failures are reported with the first violated condition so the experiment
    can break results down by failure mode (``not_lifted`` / ``not_grasped`` /
    ``not_level``); infrastructure exceptions are reported separately as
    ``"error"`` rather than silently counted as physical failures.

    The worker owns exactly one environment, recreated lazily. It is not a safe
    sandbox for arbitrary LLM code (see the plan's note on process isolation);
    for M1 the programs are the *fixed* candidates in
    :mod:`physrsr.candidate_programs`.
    """

    def __init__(
        self,
        physics: PhysicsSpec,
        hold_steps: int = 10,
        level_tolerance: float = 0.25,
    ):
        self.physics = physics
        self.hold_steps = int(hold_steps)
        self.level_tolerance = None if level_tolerance is None else float(level_tolerance)
        self._env: FrankaRobosuiteRodLiftLowLevel | None = None
        self._api: FrankaControlPrivilegedApi | None = None

    def _ensure_env(self) -> tuple[FrankaRobosuiteRodLiftLowLevel, FrankaControlPrivilegedApi]:
        if self._env is None:
            kwargs = self.physics.as_env_kwargs()
            kwargs.update(
                dict(
                    hold_steps=1,
                    level_tolerance=self.level_tolerance,
                    privileged=True,
                    enable_render=False,
                    controller_cfg=str(CAPX_ROOT / "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json"),
                )
            )
            self._env = FrankaRobosuiteRodLiftLowLevel(**kwargs)
            self._api = FrankaControlPrivilegedApi(self._env)
        return self._env, self._api

    def rollout(self, program, seed: int, role: str = "sim") -> RolloutRecord:
        """Run ``program`` once under ``seed`` and return a :class:`RolloutRecord`."""
        start = time.perf_counter()
        env, api = self._ensure_env()
        # The API keeps the last IK solution as a warm start (`prev_cfg`). If it
        # leaks across rollouts, the same (program, seed) solves to different
        # arm configurations and stops being reproducible -- verified to flip
        # P1/seed=7 between success and `not_lifted`. Each rollout must start
        # from a cold solve.
        api.cfg = None
        tracker = PhaseTracker(env, api)
        try:
            # Seed every RNG that drives episode initialisation (robot init
            # noise *and* object placement). See RodLift.seed_episode.
            env.robosuite_env.seed_episode(int(seed))
            env.reset(seed=int(seed))
            sim_start = float(env.robosuite_env.sim.data.time)
            tracker.record_initial()
            program.run(env, tracker, self.physics)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            return RolloutRecord(
                case_id=self.physics.case_id,
                program_id=program.id,
                seed=int(seed),
                role=role,
                status="error",
                reason=f"{type(exc).__name__}: {exc}"[:300],
                failure_phase=tracker.stage,
                grasp_established=tracker.grasp_established,
                phases=tracker.as_dicts(),
                duration_s=time.perf_counter() - start,
            )

        ok, reason, step = self._evaluate_hold(env, tracker)
        return RolloutRecord(
            case_id=self.physics.case_id,
            program_id=program.id,
            seed=int(seed),
            role=role,
            status="success" if ok else "fail",
            reason=reason,
            failure_step=step,
            failure_phase=None if ok else tracker.attribute_failure(),
            grasp_established=tracker.grasp_established,
            phases=tracker.as_dicts(),
            duration_s=time.perf_counter() - start,
            sim_duration_s=float(env.robosuite_env.sim.data.time) - sim_start,
        )

    def rollout_many(self, program, seeds, role: str = "sim") -> list[RolloutRecord]:
        return [self.rollout(program, seed, role=role) for seed in seeds]

    def _evaluate_hold(
        self, env, tracker: PhaseTracker | None = None
    ) -> tuple[bool, str, int | None]:
        """Hold the grasp and require the success state at every step."""
        rod = env.robosuite_env
        env._set_gripper(0.0)  # keep closed
        for step in range(self.hold_steps):
            env._step_once()
            state_ok = True
            reason = ""
            if not rod._is_lifted():
                state_ok, reason = False, "not_lifted"
            elif not rod._is_grasped():
                state_ok, reason = False, "not_grasped"
            elif not rod._is_level():
                state_ok, reason = False, "not_level"
            if tracker is not None:
                tracker.record_hold(step, state_ok)
            if not state_ok:
                return False, reason, step
        return True, "", None

    def reset_fingerprint(self, seed: int) -> np.ndarray:
        """Rod pose + robot joints right after a fresh reset of ``seed``.

        Used by the preflight check: success/failure is too coarse to detect a
        nondeterministic reset (the same seed can place the rod slightly
        differently yet still succeed), whereas the state itself is sensitive.
        """
        env, _ = self._ensure_env()
        env.robosuite_env.seed_episode(int(seed))
        env.reset(seed=int(seed))
        rod = np.asarray(env.get_observation()["cube_poses"]["primary"], dtype=np.float64)
        joints = np.asarray(
            env.robosuite_env._get_observations()["robot0_joint_pos"], dtype=np.float64
        )
        return np.concatenate([rod, joints])

    def read_rod_friction(self, seed: int = 0) -> list[float]:
        """Read back the compiled sliding friction of the rod's contact geoms.

        Used by the preflight check to confirm the physics knobs actually reach
        the compiled MuJoCo model before a long run is launched.
        """
        env, _ = self._ensure_env()
        env.robosuite_env.seed_episode(int(seed))
        env.reset(seed=int(seed))
        sim = env.robosuite_env.sim
        model = sim.model
        bid = env.robosuite_env.cube_body_id
        start = int(model.body_geomadr[bid])
        count = int(model.body_geomnum[bid])
        return [
            float(model.geom_friction[gid, 0])
            for gid in range(start, start + count)
            if model.geom_contype[gid] != 0 or model.geom_conaffinity[gid] != 0
        ]

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._api = None
