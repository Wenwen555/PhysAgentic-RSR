"""Mock 双门 World: 主链路冒烟与演示用, 行为模型与真值文件无关。

闩锁阈值 0.35 rad (mock 值): 只有把手保持角度 >= 0.35 且执行拉门时门才
打开; 纯拉门门不动; 把手释放后回弹 (回位弹簧)。
只暴露 handle_angle / door_angle / 扭矩与末端力范数, 模拟视觉 + 本体估计。
"""

from __future__ import annotations

import numpy as np

from active_sysid.backends.base import BaseWorld
from active_sysid.types import ProbeSpec


class MockDoorWorld(BaseWorld):
    name = "mock_door_world"
    dt = 0.05
    MOCK_UNLOCK_THRESHOLD = 0.35

    def _plan(self, probe: ProbeSpec) -> tuple[float, float]:
        return float(probe.params.get("duration", 1.0)), self.dt

    def _reset(self) -> None:
        self._i = 0
        self._probe: ProbeSpec | None = None
        self._prev_door = 0.0

    def _actuation(self, probe: ProbeSpec, t: float) -> dict[str, float]:
        self._probe = probe  # mock 无执行器, 只缓存 probe 供合成观测
        return {}

    def _step(self) -> None:
        self._i += 1

    def _read_channels(self) -> dict[str, float]:
        probe = self._probe
        assert probe is not None
        i, n = self._i, self._n
        p = probe.params
        handle = 0.01
        door = 0.0

        if probe.name == "rotate_handle":
            span = float(p.get("angle_span", 0.7))
            handle = span * min(i / (n / 2.0), 1.0)
        elif probe.name == "hold_and_pull":
            angle = float(p.get("handle_angle", 0.4))
            handle = min(angle * i / (n / 4.0), angle)
            if angle >= self.MOCK_UNLOCK_THRESHOLD:
                door = max(0.0, 0.45 * (i - n / 2.0) / (n / 2.0))
        elif probe.name == "release_and_observe":
            hold = float(p.get("hold_angle", 0.6))
            if i < n / 4.0:
                handle = hold * i / (n / 4.0)
            else:
                handle = hold * np.exp(-2.0 * (i - n / 4.0) * self.dt)

        return {
            "handle_angle": float(handle),
            "door_angle": float(door),
            "joint_torque_norm": 0.8 * handle + 0.1,
            "ee_force_norm": 8.0 if door > 0 else 2.0,
        }

    def _detect_events(self, snap: dict[str, float]) -> list[str]:
        door = snap["door_angle"]
        if door > 1e-6 and self._prev_door <= 1e-6:
            self._prev_door = door
            return ["unlock"]
        self._prev_door = door
        return []
