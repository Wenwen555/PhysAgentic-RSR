
"""将闩锁结构与辨识出的连续物理参数写入 MuJoCo MJCF。"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from active_sysid.types import MechanismHypothesis


class ModelEditor:
    def apply(
        self,
        mjcf_path: str | Path,
        model: MechanismHypothesis,
        params: dict[str, float],
        out_path: str | Path,
    ) -> Path:
        src = Path(mjcf_path)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        tree = ET.parse(src)
        root = tree.getroot()
        merged = {**model.params, **params}

        # M1 结构修复：闩锁 weld 约束
        if "latch_coupling" in model.structure.get("handle", []):
            equality = root.find("equality")
            if equality is None:
                equality = ET.SubElement(root, "equality")

            weld = equality.find("weld[@name='latch_weld']")
            if weld is None:
                weld = ET.SubElement(
                    equality, "weld", name="latch_weld"
                )

            weld.attrib.update({
                "body1": "door",
                "body2": "frame",
                "active": "true",
                "solref": "0.01 1",
                "solimp": "0.98 0.999 0.001",
            })

        door = root.find(".//joint[@name='door_hinge']")
        handle = root.find(".//joint[@name='handle_hinge']")
        leaf = root.find(".//geom[@name='door_leaf']")

        # M2 连续参数修复
        for key, element, attr in (
            ("b_d", door, "damping"),
            ("mu_d", door, "frictionloss"),
            ("k_h", handle, "stiffness"),
            ("b_h", handle, "damping"),
            ("theta_h0", handle, "springref"),
        ):
            if key in params:
                element.set(attr, f"{params[key]:.12g}")

        if "I_d" in params:
            # 本场景 door_leaf 是 box，旋转轴为 z。
            # I_z = m * [(sx^2 + sy^2)/3 + px^2 + py^2]
            sx, sy, sz = map(
                float, leaf.get("size").split()
            )
            px, py, _ = map(
                float, leaf.get("pos").split()
            )

            volume = 8.0 * sx * sy * sz
            inertia_per_mass = (
                (sx * sx + sy * sy) / 3.0
                + px * px + py * py
            )

            density = (
                float(params["I_d"])
                / (volume * inertia_per_mass)
            )
            leaf.set("density", f"{density:.12g}")

        ET.indent(tree, space="  ")
        tree.write(
            out,
            encoding="utf-8",
            xml_declaration=False,
        )

        metadata = {
            "source": str(src),
            "hypothesis": model.hid,
            "structure": model.structure,
            "params": merged,
            "notes": model.notes,
            "repair_mode": (
                "functional_latch"
                if "latch_coupling"
                in model.structure.get("handle", [])
                else "identity"
            ),
        }

        if metadata["repair_mode"] == "functional_latch":
            metadata["latch"] = {
                "unlock_threshold": float(
                    merged["theta_unlock"]
                ),
                "latch_weld": "latch_weld",
            }

        out.with_suffix(
            out.suffix + ".repair.json"
        ).write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return out

