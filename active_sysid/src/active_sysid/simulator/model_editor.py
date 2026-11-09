
"""M1: 将机构修复真正写入重建 MJCF。"""

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

        parser = ET.XMLParser(
            target=ET.TreeBuilder(insert_comments=True)
        )
        tree = ET.parse(src, parser=parser)
        root = tree.getroot()

        merged = {**model.params, **params}
        has_latch = (
            "latch_coupling"
            in model.structure.get("handle", [])
        )

        metadata = {
            "source": str(src),
            "hypothesis": model.hid,
            "structure": model.structure,
            "params": merged,
            "repair_mode": "identity",
            "notes": model.notes,
        }

        if has_latch:
            threshold = float(
                merged["theta_unlock"]
            )

            equality = root.find("equality")
            if equality is None:
                equality = ET.SubElement(
                    root, "equality"
                )

            weld = equality.find(
                "weld[@name='latch_weld']"
            )
            if weld is None:
                weld = ET.SubElement(
                    equality,
                    "weld",
                    name="latch_weld",
                )

            weld.attrib.update({
                "body1": "door",
                "body2": "frame",
                "active": "true",
                "solref": "0.01 1",
                "solimp": "0.98 0.999 0.001",
            })

            metadata["repair_mode"] = (
                "functional_latch"
            )
            metadata["latch"] = {
                "unlock_threshold": threshold,
                "latch_weld": "latch_weld",
            }

        ET.indent(tree, space="  ")
        tree.write(
            out,
            encoding="utf-8",
            xml_declaration=False,
        )

        sidecar = out.with_suffix(
            out.suffix + ".repair.json"
        )
        sidecar.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return out

