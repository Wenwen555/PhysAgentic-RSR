"""配置校验: 研究超参只以 configs/pipeline.yaml 为唯一真值。

各组件构造时用 require_section 强制传入对应配置段并校验必需键,
代码里不保留默认副本, 避免 yaml 与代码两处真值漂移。
"""

from __future__ import annotations

from typing import Any


def require_section(section: dict[str, Any] | None, keys: tuple[str, ...],
                    source: str) -> dict[str, Any]:
    """校验配置段存在且包含全部必需键, 返回其浅拷贝。"""
    if section is None:
        raise ValueError(
            f"{source} 未提供; 应来自 configs/pipeline.yaml, 代码中不保留默认副本"
        )
    missing = [k for k in keys if k not in section]
    if missing:
        raise ValueError(f"{source} 缺少键 {missing}; 请补全 configs/pipeline.yaml")
    return dict(section)
