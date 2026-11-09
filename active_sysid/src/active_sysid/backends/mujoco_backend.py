"""MuJoCo 后端: 以 object-space 直驱方式执行 Probe。

设计 (统一接口的一部分):
- channel_map: 规范通道名 -> MuJoCo joint 名, 例如
      {"handle_angle": "handle_hinge", "door_angle": "door_hinge"}
  只映射外部可观测的关节; World 的隐藏门栓/等式约束不进 channel_map,
  由是保持"辨识器只能拿到机器人/视觉可观测状态"的契约。
- 执行: probe_actions.build_profile 给出 (mode, value), 本后端把
  "position" 做 PD 关节控制, "force" 换算为铰链扭矩 (力 x 杠杆臂),
  通过 data.qfrc_applied 直驱对象关节。
- 观测: 每帧记录 {channel}, {channel}_vel, {channel}_tau_applied;
  机器人本体通道 (q/tau/ee_wrench) 待接入带机械臂的场景后补充。

限制: 当前不含机械臂 —— 真正的本体感受式执行需要机器人后端
(对同一个 BaseWorld 模板实现 _apply/_read_channels)。

单位约定: probe 参数与通道一律 SI 弧度/米/牛(·米); MJCF 需以
<compiler angle="radian"/> 编写 (MuJoCo 默认 degree, range 会被
 silently 换算, 例如 1.508 deg -> 0.0263 rad, 门几乎开不了)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from active_sysid.backends.base import BaseWorld
from active_sysid.backends.probe_actions import build_profile

# 执行目标名 -> 规范通道名 (channel_map 的键)
TARGET_CHANNEL = {"handle": "handle_angle", "door": "door_angle"}

# object-space PD 增益与扭矩上限 (驱动层默认; 场景级覆盖走后端构造参数)
DEFAULT_GAINS: dict[str, dict[str, float]] = {
    "handle": {"kp": 40.0, "kd": 4.0, "cap": 30.0},
    "door": {"kp": 120.0, "kd": 12.0, "cap": 60.0},
}


class MuJoCoBackEnd(BaseWorld):
    name = "mujoco"

    def __init__(self, mjcf: str | Path,
                 channel_map: dict[str, str],
                 lever_arm: float = 0.4,
                 gains: dict[str, dict[str, float]] | None = None,
                 dt: float | None = None):
        import mujoco as mj  # 延迟导入: 未装 sim extra 时 mock 后端仍可用

        self._mj = mj
        self.model = mj.MjModel.from_xml_path(str(mjcf))
        self.data = mj.MjData(self.model)
        self.dt = float(dt or self.model.opt.timestep)
        self.lever_arm = float(lever_arm)

        self.gains = {k: dict(v) for k, v in DEFAULT_GAINS.items()}
        for key, override in (gains or {}).items():
            self.gains.setdefault(key, {}).update(override)

        self._joints: dict[str, dict[str, Any]] = {}
        for canonical, jname in channel_map.items():
            jid = int(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname))
            if jid < 0:
                raise ValueError(f"joint {jname!r} 不存在于 {mjcf}")
            self._joints[canonical] = {
                "jid": jid,
                "qadr": int(self.model.jnt_qposadr[jid]),
                "vadr": int(self.model.jnt_dofadr[jid]),
            }
        self._profile = None
        self._profile_key = ""
        self._last_tau: dict[str, float] = {}

    # ------------------------------------------------------------------ 钩子
    def _reset(self) -> None:
        self._mj.mj_resetData(self.model, self.data)
        self._last_tau = {}

    def _actuation(self, probe, t: float) -> dict[str, float]:
        if self._profile_key != probe.key():
            self._profile = build_profile(probe)
            self._profile_key = probe.key()
        out: dict[str, float] = {}
        for target, (mode, value) in self._profile(t).items():
            channel = TARGET_CHANNEL.get(target, target)
            joint = self._joints.get(channel)
            if joint is None:
                raise KeyError(f"channel_map 缺少 {channel!r}, 无法驱动 {target!r}")
            if mode == "free":
                out[channel] = 0.0
            elif mode == "force":
                # 把手处拉力 (N) -> 铰链扭矩 (N·m)
                out[channel] = float(value) * self.lever_arm
            else:  # position: PD 控制
                g = self.gains.get(target, DEFAULT_GAINS["door"])
                q = float(self.data.qpos[joint["qadr"]])
                qd = float(self.data.qvel[joint["vadr"]])
                tau = g["kp"] * (float(value) - q) - g["kd"] * qd
                out[channel] = float(np.clip(tau, -g["cap"], g["cap"]))
        return out

    def _apply(self, torques: dict[str, float]) -> None:
        self.data.qfrc_applied[:] = 0.0
        for channel, tau in torques.items():
            joint = self._joints[channel]
            self.data.qfrc_applied[joint["vadr"]] = tau
            self._last_tau[channel] = tau

    def _step(self) -> None:
        self._mj.mj_step(self.model, self.data)

    def _read_channels(self) -> dict[str, float]:
        snap: dict[str, float] = {}
        for channel, joint in self._joints.items():
            snap[channel] = float(self.data.qpos[joint["qadr"]])
            snap[f"{channel}_vel"] = float(self.data.qvel[joint["vadr"]])
            snap[f"{channel}_tau_applied"] = self._last_tau.get(channel, 0.0)
        return snap

    def _detect_events(self, snap: dict[str, float]) -> list[str]:
        events: list[str] = []
        for channel, joint in self._joints.items():
            jid = joint["jid"]
            if not self.model.jnt_limited[jid]:
                continue
            lo, hi = self.model.jnt_range[jid]
            q = snap[channel]
            if abs(self._last_tau.get(channel, 0.0)) > 0.5 and (
                q - lo < 2e-3 or hi - q < 2e-3
            ):
                events.append(f"limit_hit:{channel}")
        return events
