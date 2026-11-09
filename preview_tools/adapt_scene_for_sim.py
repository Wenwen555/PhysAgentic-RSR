"""Adapt the unidoor Stage 3 export into the layout prepare_sim_scene.py expects.

Structural input prep only (no pipeline code is modified):
- hoist the embedded-robot mount to a direct worldbody child named fr3_mount,
  carrying the exact world pose the robot base has inside common_world_frame;
- rename the embedded robot root to fr3_link0 (prepare_sim_scene whitelist);
- drop robot-only sensors whose sites live inside the stripped subtree;
- drop keyframes (the CaP runtime resets via its own robot interface).
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("/home/wen/experiment/PhyAgent-RSR/output/unidoor_real2sim_20261108/stage3_final/scene.xml")
OUT = Path("/home/wen/experiment/PhyAgent-RSR/output/unidoor_real2sim_20261108/stage3_sim_source")
WORLD_POS = (-0.14, 0.04, 0.8)
WORLD_QUAT = "0 0 0 1"  # Rz(-pi)

OUT.mkdir(parents=True, exist_ok=True)
assets_link = OUT / "assets"
if not assets_link.exists():
    assets_link.symlink_to((SRC.parent / "assets").resolve(), target_is_directory=True)

root = ET.parse(SRC).getroot()
wb = root.find("worldbody")
cwf = next(b for b in wb if b.get("name") == "common_world_frame")
mount = next(b for b in cwf if b.get("name") == "panda_mount")
robot_root = mount[0]
assert robot_root.get("name") == "robot0_base"

# detach and re-root the mount
cwf.remove(mount)
mount.tag = "body"
mount.set("name", "fr3_mount")
mount.set("pos", " ".join(map(str, WORLD_POS)))
mount.set("quat", WORLD_QUAT)
mount.attrib.pop("euler", None)
robot_root.set("name", "fr3_link0")
wb.insert(0, mount)

# sensors referencing sites inside the (later-stripped) robot subtree
robot_sites = {s.get("name") for s in mount.iter("site") if s.get("name")}
sensor_el = root.find("sensor")
if sensor_el is not None:
    for s in list(sensor_el):
        ref = s.get("site") or s.get("body") or s.get("frame") or ""
        if ref in robot_sites or ref.startswith(("robot0_", "gripper0_")):
            sensor_el.remove(s)
    if not len(sensor_el):
        root.remove(sensor_el)

# keyframes would reference stripped joints
kf = root.find("keyframe")
if kf is not None:
    root.remove(kf)

tree_out = OUT / "scene.xml"
ET.ElementTree(root).write(tree_out, encoding="utf-8", xml_declaration=True)
print("wrote", tree_out)

import mujoco
m = mujoco.MjModel.from_xml_path(str(tree_out))
bid = m.body("fr3_mount").id
print("adapted scene loads | nbody", m.nbody, "nq", m.nq, "nu", m.nu,
      "| fr3_mount pos", m.body("fr3_mount").pos)
print("cameras:", sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)))
