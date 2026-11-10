
"""M3 Runtime：动作连续性、预算和观测接口测试。"""

import pytest

from active_sysid.cap.runtime import (
    DoorRuntime,
    StepBudgetExceeded,
)


class FakeDoorBackend:
    dt = 0.01

    def _reset(self):
        self.handle = 0.0
        self.door = 0.0
        self.vel = 0.0

    def _read_channels(self):
        return {
            "handle_angle": self.handle,
            "door_angle": self.door,
            "door_angle_vel": self.vel,
            "handle_angle_vel": 0.0,
        }

    def command_torques(self, commands):
        return commands

    def _apply(self, torques):
        self.actions = torques

    def _step(self):
        handle_mode, target = self.actions["handle"]
        door_mode, force = self.actions["door"]

        if handle_mode == "position":
            self.handle = float(target)

        if (
            door_mode == "force"
            and self.handle >= 0.56
        ):
            self.vel = 0.01 * float(force)
            self.door += self.vel
        else:
            self.vel = 0.0


def test_commands_preserve_episode_state():
    api = DoorRuntime(
        FakeDoorBackend(),
        max_steps=20,
        max_pull_force_n=15,
    )

    api.step(
        handle=("position", 0.7),
        steps=3,
    )

    obs = api.step(
        handle=("position", 0.7),
        door=("force", 10),
        steps=4,
    )

    assert obs["door_angle"] > 0
    assert obs["handle_angle"] == pytest.approx(0.7)
    assert len(api.trace) == 7
    assert all(
        "_latched" not in row
        for row in api.trace
    )


def test_direct_pull_does_not_unlock():
    api = DoorRuntime(
        FakeDoorBackend(),
        max_steps=20,
        max_pull_force_n=15,
    )

    api.step(
        door=("force", 10),
        steps=10,
    )

    assert api.observe()["door_angle"] == 0.0


def test_force_and_step_budgets():
    api = DoorRuntime(
        FakeDoorBackend(),
        max_steps=5,
        max_pull_force_n=15,
    )

    with pytest.raises(ValueError, match="force"):
        api.step(door=("force", 20))

    with pytest.raises(StepBudgetExceeded):
        api.step(steps=6)


def test_evaluator_returns_observable_feedback(monkeypatch):
    from active_sysid.cap import evaluator as module
    from active_sysid.types import TaskSpec

    monkeypatch.setattr(
        module,
        "create_backend",
        lambda spec: FakeDoorBackend(),
    )

    task = TaskSpec(
        task_id="door_open",
        description="Open the door",
        success_criteria={
            "door_angle_min": 0.2,
            "max_steps": 20,
            "max_pull_force_n": 15,
        },
    )

    def policy(api):
        api.step(
            handle=("position", 0.7),
            steps=2,
        )
        api.step(
            handle=("position", 0.7),
            door=("force", 10),
            steps=5,
        )

    result = module.CaPEvaluator().evaluate(
        policy, {}, task
    )

    assert result["success_rate"] == 1.0
    assert (
        result["rollouts"][0]
        ["final_observation"]["door_angle"]
        >= 0.2
    )
    assert (
        "_latched"
        not in result["rollouts"][0]["final_observation"]
    )

