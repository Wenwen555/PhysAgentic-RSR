"""Physical Probe Library: 参数化动作模板 (configs/probes.yaml)。

模板是"实验意图 + 参数范围", 而非一组固定轨迹。实例化时由 Planner/
Generator 根据机械臂能力、接触状态与上一次 Probe 的测量结果确定具体参数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from active_sysid.types import ProbeSpec


class ProbeTemplate:
    def __init__(self, raw: dict[str, Any]):
        self.name: str = raw["name"]
        self.description: str = raw.get("description", "")
        self.identifies: list[str] = raw.get("identifies", [])
        self.default_params: dict[str, Any] = dict(raw.get("default_params", {}))
        self.param_ranges: dict[str, list[float]] = dict(raw.get("param_ranges", {}))
        self.observations: list[str] = raw.get("observations", [])
        self.info_gain: dict[str, float] = {
            "structural": float(raw.get("info_gain", {}).get("structural", 0.0)),
            "parametric": float(raw.get("info_gain", {}).get("parametric", 0.0)),
        }
        self.cost: float = float(raw.get("cost", 1.0))

    def instantiate(self, target: str, overrides: dict[str, Any] | None = None) -> ProbeSpec:
        """用默认参数实例化, 允许覆盖; 覆盖值需落在 param_ranges 内。"""
        params = dict(self.default_params)
        for key, value in (overrides or {}).items():
            rng = self.param_ranges.get(key)
            if rng and isinstance(value, (int, float)):
                lo, hi = float(rng[0]), float(rng[1])
                if not (lo <= float(value) <= hi):
                    raise ValueError(
                        f"probe {self.name!r} param {key}={value} 超出范围 [{lo}, {hi}]"
                    )
            params[key] = value
        return ProbeSpec(
            name=self.name,
            target=target,
            params=params,
            observations=list(self.observations),
        )


class ProbeLibrary:
    def __init__(self, templates: list[ProbeTemplate]):
        self.templates = templates
        self._by_name = {t.name: t for t in templates}

    @classmethod
    def load(cls, path: str | Path) -> "ProbeLibrary":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        templates = [ProbeTemplate(item) for item in raw.get("probes", [])]
        if not templates:
            raise ValueError(f"probe library 为空: {path}")
        return cls(templates)

    def get(self, name: str) -> ProbeTemplate:
        return self._by_name[name]

    def names(self) -> list[str]:
        return [t.name for t in self.templates]
