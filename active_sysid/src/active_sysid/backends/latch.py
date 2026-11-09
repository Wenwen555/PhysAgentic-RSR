"""M0 功能性闩锁：把手达到阈值才能释放门的 weld 约束。

锁定状态属于 Oracle 内部真值；ProbeTrajectory 只暴露
门、把手的可观测运动与施加的力矩，不把 unlock/relock
状态事件泄漏给辨识器。
"""

from __future__ import annotations

from active_sysid.backends.mujoco_backend import MuJoCoBackEnd


class FunctionalLatchWorld(MuJoCoBackEnd):
    name = "mujoco_functional_latch"

    def __init__(
        self,
        *args,
        unlock_threshold: float,
        relock_door_eps: float = 0.05,
        latch_weld: str = "latch_weld",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.unlock_threshold = float(unlock_threshold)
        self.relock_door_eps = float(relock_door_eps)

        for channel in ("handle_angle", "door_angle"):
            if channel not in self._joints:
                raise ValueError(
                    f"功能性闩锁需要 channel_map 含 {channel!r}"
                )

        mj = self._mj
        self._eq_id = int(
            mj.mj_name2id(
                self.model,
                mj.mjtObj.mjOBJ_EQUALITY,
                latch_weld,
            )
        )

        if self._eq_id < 0:
            raise ValueError(
                f"equality {latch_weld!r} 不存在于 MJCF"
            )

        self._latched = True

    def _reset(self) -> None:
        super()._reset()

        self._latched = True
        self.data.eq_active[self._eq_id] = 1

        self._mj.mj_forward(self.model, self.data)

    def _step(self) -> None:
        handle_q = float(
            self.data.qpos[
                self._joints["handle_angle"]["qadr"]
            ]
        )
        door_q = float(
            self.data.qpos[
                self._joints["door_angle"]["qadr"]
            ]
        )

        if self._latched:
            # 只有把手转到阈值以上，才能解锁。
            if handle_q >= self.unlock_threshold:
                self._latched = False

        elif (
            handle_q < self.unlock_threshold
            and abs(door_q) < self.relock_door_eps
        ):
            # 门接近关闭、把手已释放，才能重新上锁。
            self._latched = True

        self.data.eq_active[self._eq_id] = int(
            self._latched
        )

        super()._step()
