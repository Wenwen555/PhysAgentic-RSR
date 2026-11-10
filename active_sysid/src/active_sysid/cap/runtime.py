
"""CaP object-space Runtime：连续执行、观测及轨迹记录。"""

from __future__ import annotations

from typing import Any


class StepBudgetExceeded(RuntimeError):
    pass


class DoorRuntime:
    def __init__(
        self,
        backend,
        *,
        max_steps: int,
        max_pull_force_n: float,
    ):
        self._backend = backend
        self.max_steps = int(max_steps)
        self.max_pull_force_n = float(max_pull_force_n)
        self.steps = 0
        self.calls = 0
        self.trace: list[dict[str, Any]] = []

        self._backend._reset()

    def observe(self) -> dict[str, float]:
        return {
            "time_s": self.steps * self._backend.dt,
            **self._backend._read_channels(),
        }

    def step(
        self,
        *,
        handle: tuple[str, float] = ("free", 0.0),
        door: tuple[str, float] = ("free", 0.0),
        steps: int = 1,
    ) -> dict[str, float]:

        if steps < 1 or self.steps + steps > self.max_steps:
            raise StepBudgetExceeded(
                "CaP step budget exceeded"
            )

        if (
            door[0] == "force"
            and abs(float(door[1])) > self.max_pull_force_n
        ):
            raise ValueError(
                "door pull force exceeds task limit"
            )

        commands = {
            "handle": handle,
            "door": door,
        }

        self.calls += 1

        for _ in range(steps):
            torques = self._backend.command_torques(
                commands
            )

            self._backend._apply(torques)
            self._backend._step()

            self.steps += 1

            self.trace.append({
                **self.observe(),
                "handle_command": list(handle),
                "door_command": list(door),
            })

        return self.observe()

