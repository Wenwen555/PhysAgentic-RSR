"""World 后端基类: 固化 run_probe 的执行模板。

执行循环对所有后端一致:
    reset -> 逐帧 [actuation -> apply -> step -> read_channels -> detect_events]
后端只实现钩子 (_reset/_actuation/_apply/_step/_read_channels/_detect_events),
从而 mock / MuJoCo / 真机产出同构的 ProbeTrajectory。真机后端把
_apply/_step 换成实时控制与采样即可, 执行循环与数据记录不用重写。

满足 probes/executor.py 中 WorldInterface 协议 (结构化兼容)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from active_sysid.types import ProbeSpec, ProbeTrajectory

DEFAULT_DURATION = 2.0


class BaseWorld(ABC):
    name: str = "base"
    dt: float = 0.05

    def run_probe(self, probe: ProbeSpec) -> ProbeTrajectory:
        duration, dt = self._plan(probe)
        n = max(1, int(round(duration / dt)))
        self._n = n
        self._reset()
        ts: list[float] = []
        chans: dict[str, list[float]] = defaultdict(list)
        events: list[str] = []
        for i in range(n):
            t = i * dt
            self._t = t
            self._apply(self._actuation(probe, t))
            self._step()
            snap = self._read_channels()
            for key, value in snap.items():
                chans[key].append(value)
            ts.append(t)
            events.extend(self._detect_events(snap))
        return ProbeTrajectory(
            t=np.asarray(ts),
            channels={k: np.asarray(v) for k, v in chans.items()},
            events=list(dict.fromkeys(events)),  # 去重且保序
        )

    # ------------------------------------------------------------------ 钩子
    def _plan(self, probe: ProbeSpec) -> tuple[float, float]:
        """(duration, dt): 默认取 probe 的 duration 参数与后端固定步长。"""
        return float(probe.params.get("duration", DEFAULT_DURATION)), self.dt

    @abstractmethod
    def _reset(self) -> None:
        """恢复到 Probe 起始状态。"""

    @abstractmethod
    def _step(self) -> None:
        """推进一个物理/控制周期。"""

    @abstractmethod
    def _read_channels(self) -> dict[str, float]:
        """当前帧的观测通道 (规范通道名 -> 标量)。"""

    def _actuation(self, probe: ProbeSpec, t: float) -> dict[str, float]:
        """根据 probe 模板与当前时刻计算控制量 (目标名 -> 广义力)。"""
        return {}

    def _apply(self, torques: dict[str, float]) -> None:
        """把控制量写入执行器。"""

    def _detect_events(self, snap: dict[str, float]) -> list[str]:
        """当前帧的离散事件 ("unlock" / "limit_hit:door" ...)。"""
        return []
