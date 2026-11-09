"""MJCF 结构修复 (Structure Repair) 与参数修复 (Parameter Repair)。

研究方案第七节。结构修复两种形式:
- 显式机构: 增加门栓滑动关节、耦合等式和 strike 接触几何;
- 功能性机构: 增加离散锁定状态, 当把手转角超过辨识阈值时解除门的
  运动约束。第一阶段先用功能性实现闭环 (当前实现), 之后再升级。

参数修复的 MJCF 映射:
    body/inertial          <- I_d (转动惯量, 注意不是直接的 12 kg 质量)
    joint/damping          <- b_d, b_h
    joint/frictionloss     <- mu_d (库仑摩擦近似)
    joint/stiffness        <- k_h (与 springref 配合: springref = theta_h0)
    joint/range            <- 运动限位
    equality / 自定义状态约束 <- theta_unlock 触发的锁定解除

当前实现: 生成修复后的 MJCF 副本 + 修复清单 sidecar (repair.json),
XML 编辑逻辑接入 MuJoCo 后端后补全 (见 TODO)。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from active_sysid.types import MechanismHypothesis


class ModelEditor:
    def apply(self, mjcf_path: str | Path, model: MechanismHypothesis,
              params: dict[str, float], out_path: str | Path) -> Path:
        """把结构 + 参数修复写入新的 MJCF 文件, 返回其路径。"""
        src = Path(mjcf_path)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        text = src.read_text(encoding="utf-8")
        repair = {
            "source": str(src),
            "structure": model.structure,
            "hypothesis": model.hid,
            "params": params,
            "repair_mode": "functional_latch",  # 第一阶段: 功能性机构
            "notes": model.notes,
        }

        # TODO(结构修复 - 功能性闩锁): 在 door hinge joint 上叠加
        #   离散锁定状态: theta_h < theta_unlock 时 constraint 生效,
        #   超过阈值后释放。MuJoCo 侧需自定义状态 + xfrc_applied / 约束开关。
        # TODO(参数修复): 按 docstring 映射表写 joint/damping、
        #   frictionloss、stiffness/springref、range、body/inertial。

        marker = f"<!-- active_sysid repair: {model.hid} -->"
        text = marker + "\n" + text
        out.write_text(text, encoding="utf-8")
        sidecar = out.with_suffix(out.suffix + ".repair.json")
        sidecar.write_text(json.dumps(repair, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        return out
