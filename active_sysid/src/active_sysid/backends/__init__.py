"""World 后端: 统一实现 WorldInterface (probes/executor.py 的契约)。

- BaseWorld:       执行循环模板, 所有后端共享
- MuJoCoBackEnd:   MuJoCo object-space 直驱后端
- MockDoorWorld:   mock 双门 World (冒烟/演示)
- probe_actions:   Probe 模板 -> 动作目标的纯函数翻译
- create_backend:  由配置字典构造后端的统一入口
"""

from __future__ import annotations

from typing import Any

from active_sysid.backends.base import BaseWorld
from active_sysid.backends.mock import MockDoorWorld
from active_sysid.backends.mujoco_backend import MuJoCoBackEnd  # mujoco 延迟到构造时才导入

# channel_map 缺省: 双门场景的常规关节命名
DEFAULT_CHANNEL_MAP = {"handle_angle": "handle_hinge", "door_angle": "door_hinge"}


def create_backend(spec: dict[str, Any]) -> BaseWorld:
    """由配置构造后端 (configs/mujoco_world.example.yaml 即本函数的输入示例):

        {"backend": "mujoco", "mjcf": "worlds/reconstructed/x.mjcf",
         "channel_map": {"handle_angle": "handle_hinge", ...}}
    """
    kind = spec.get("backend")
    if kind == "mock":
        return MockDoorWorld()
    if kind == "mujoco":
        from active_sysid.backends.mujoco_backend import MuJoCoBackEnd

        cls = MuJoCoBackEnd
        kwargs = {k: spec[k] for k in ("lever_arm", "gains", "dt") if k in spec}
        if "latch" in spec:  # 功能性闩锁 (oracle 场景): 离散锁定状态
            from active_sysid.backends.latch import FunctionalLatchWorld
            cls = FunctionalLatchWorld
            kwargs.update(spec["latch"])
        return cls(
            mjcf=spec["mjcf"],
            channel_map=spec.get("channel_map", DEFAULT_CHANNEL_MAP),
            **kwargs,
        )
    raise ValueError(f"未知后端 {kind!r}; 可选 'mock' | 'mujoco'")


__all__ = ["BaseWorld", "MockDoorWorld", "MuJoCoBackEnd",
           "DEFAULT_CHANNEL_MAP", "create_backend"]
