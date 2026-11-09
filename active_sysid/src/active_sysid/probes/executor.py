"""Probe 执行器: 在 World 中执行 Probe 并记录 Evidence。

WorldInterface 契约 (研究方案 6.1 节), 后端无关:
- 只暴露机器人本体可获得的状态: 关节位置/速度/控制命令、关节扭矩、
  夹爪状态与末端位姿、由动力学模型估计的外部接触力、以及由视觉估计的
  把手/门叶角度与接触状态。
- 不暴露 World 的隐藏关节、等式约束或物理参数真值。
- 由关节扭矩估计外力时, 需扣除机器人自身重力/惯性/摩擦并经接触雅可比
  转换; 禁止把 qfrc_actuator 直接当作把手接触力。

实现见 active_sysid.backends (BaseWorld 模板 + MuJoCo/Mock 后端),
经 backends.create_backend(spec) 构造; 本模块只依赖协议, 不依赖后端。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from active_sysid.config import require_section
from active_sysid.types import Evidence, ProbeSpec, ProbeTrajectory


@runtime_checkable
class WorldInterface(Protocol):
    """后端无关的交互接口: MuJoCo World / 真机 / mock 均实现它。"""

    name: str

    def run_probe(self, probe: ProbeSpec) -> ProbeTrajectory:
        ...


class ProbeExecutor:
    """feature_eps 是场景标定值, 必须显式传入 (configs/pipeline.yaml 的 features 段):

    - door_immobile_eps: 门叶最大角位移低于此视为不可动 (rad)
    - handle_rotates_eps: 把手角行程高于此视为可转 (rad)
    - door_opens_eps: 门叶开度高于此视为可开 (rad)
    - return_ratio: 把手释放后回位比例

    代码里不留默认副本, 阈值只有 yaml 一处真值。
    """

    REQUIRED_EPS = ("door_immobile_eps", "handle_rotates_eps",
                    "door_opens_eps", "return_ratio")

    def __init__(self, feature_eps: dict[str, float] | None = None,
                 record_dir: str | Path | None = None):
        self.eps = require_section(feature_eps, self.REQUIRED_EPS, "features")
        self.record_dir = Path(record_dir) if record_dir else None

    # ------------------------------------------------------------------
    def execute(self, probe: ProbeSpec, world: WorldInterface,
                index: int = 0) -> Evidence:
        traj = world.run_probe(probe)
        summary = self.extract_summary(probe, traj)
        evidence = Evidence(probe=probe, trajectory=traj, summary=summary)
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            out = self.record_dir / f"probe_{index:03d}_{probe.name}.json"
            out.write_text(
                json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return evidence

    # ------------------------------------------------------------------
    def extract_summary(self, probe: ProbeSpec, traj: ProbeTrajectory) -> dict[str, Any]:
        """从轨迹提取判别性特征 (供假设更新与机构辨识)。

        channel 约定: "handle_angle" / "door_angle" 为视觉或本体估计的
        把手与门叶角 (rad)。真实后端应保证这两个 channel 存在。
        """
        handle = traj.channels.get("handle_angle")
        door = traj.channels.get("door_angle")
        summary: dict[str, Any] = {"probe_name": probe.name}

        handle_span = float(handle.max() - handle.min()) if handle is not None else 0.0
        door_max = float(door.max()) if door is not None else 0.0
        summary["handle_angle_span"] = handle_span
        summary["door_max_angle"] = door_max
        summary["handle_rotates"] = handle_span > self.eps["handle_rotates_eps"]
        summary["door_immobile"] = door_max < self.eps["door_immobile_eps"]
        summary["door_opens"] = door_max > self.eps["door_opens_eps"]

        # release 类实验: 判断把手是否回位 (回位弹簧证据)
        if handle is not None and "release" in probe.name and len(handle) > 4:
            peak_idx = int(handle.argmax())
            tail_mean = float(handle[peak_idx:].mean())
            summary["handle_returns"] = (
                tail_mean < (1.0 - self.eps["return_ratio"]) * float(handle.max())
            )

        # hold_and_pull: 把手角度与门可动性的条件关系 (解锁因果证据)
        if probe.name == "hold_and_pull" and summary["door_opens"]:
            summary["unlock_angle"] = float(
                probe.params.get("handle_angle", handle.max() if handle is not None else 0.0)
            )

        summary["events"] = list(traj.events)
        return summary
