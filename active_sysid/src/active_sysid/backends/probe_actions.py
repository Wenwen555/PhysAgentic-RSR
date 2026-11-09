"""Probe 模板 -> 关节动作目标的翻译 (纯函数, 与具体后端无关)。

输出是"时刻 t 该给哪些目标什么指令"的函数:
    profile(t) -> {target: (mode, value)}
    target:  "handle" / "door" 等机构目标名
    mode:    "position" (角度目标, 由后端做 PD) / "force" (把手处拉力, N)
             / "free" (释放, 不施力)

这里只描述动作意图; 后端负责把 (mode, value) 转成执行器输入并记录本体感受。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from active_sysid.types import ProbeSpec

ModeValue = tuple[str, Any]  # ("position"|"force"|"free", value|None)
Profile = Callable[[float], dict[str, ModeValue]]


def build_profile(probe: ProbeSpec) -> Profile:
    p = probe.params
    duration = float(p.get("duration", 2.0))
    name = probe.name

    if name == "gentle_pull":
        force = float(p.get("max_force", 10.0))
        return lambda t: {"door": ("force", force)}

    if name == "rotate_handle":
        span = float(p.get("angle_span", 0.7))
        ramp = max(duration / 2.0, 1e-6)
        return lambda t: {"handle": ("position", min(span * t / ramp, span))}

    if name == "hold_and_pull":
        angle = float(p.get("handle_angle", 0.4))
        force = float(p.get("max_force", 10.0))
        
        ramp = duration * 0.3
        pull_start = duration * 0.5
        
        def profile(t: float) -> dict[str, ModeValue]:
            return {
                "handle": (
                    "position",
                    angle * min(t / ramp, 1.0),
                ),
                "door": (
                    "force",
                    force if t >= pull_start else 0.0,
                ),
            }
            
        return profile

    if name == "release_and_observe":
        hold = float(p.get("hold_angle", 0.6))
        ramp = max(duration / 4.0, 1e-6)
        return lambda t: (
            {"handle": ("position", min(hold * t / ramp, hold))}
            if t < ramp else {"handle": ("free", None)}
        )

    if name == "quasi_static_sweep":
        speed = float(p.get("speed", 0.05))
        return lambda t: {"door": ("position", speed * t)}

    if name == "velocity_sweep":
        speeds = [float(s) for s in p.get("speeds", [0.1, 0.3, 0.6])]
        seg = duration / max(len(speeds), 1)

        def velocity(t: float) -> dict[str, ModeValue]:
            j = min(int(t / seg), len(speeds) - 1)
            done = sum(speeds[k] * seg for k in range(j))
            return {"door": ("position", done + speeds[j] * (t - j * seg))}

        return velocity

    if name == "acceleration_pulse":
        accel = float(p.get("accel", 2.0))
        return lambda t: {"door": ("position", 0.5 * accel * t * t)}

    if name == "bidirectional_probe":
        span = float(p.get("span", 0.3))
        return lambda t: {"door": ("position", -span if t < duration / 2 else span)}

    if name == "limit_search":
        step = float(p.get("step", 0.1))
        max_span = float(p.get("max_span", 1.4))
        n_steps = max(int(max_span / step), 1) if step > 0 else 1
        dwell = duration / n_steps
        return lambda t: {"door": ("position", min(step * (int(t / dwell) + 1), max_span))}

    raise ValueError(
        f"未支持的 probe 模板 {name!r}; 在 probe_actions.build_profile 中扩展"
    )
