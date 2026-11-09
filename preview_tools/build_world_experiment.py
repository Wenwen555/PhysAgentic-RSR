"""Build the World experiment scene for the visual_exact CaP policy.

Official CaP runtime (robot_control franka_fr3 + sim.stage3_scene) drives a
scene that needs: empty fr3_mount, stage3_camera_0, d435i wrist camera, and
task bodies with joints named as the policy expects (LeverDoor_Leaf etc.).

This adapter copies the pinned World scene (read-only source) and:
- replaces the Panda subtree with the official FR3+d435i mount contract by
  renaming robot0_base->fr3_mount (empty) so prepare_sim_scene can strip the
  Panda and robot_control compiles its own FR3 in the same pose;
- renames World task bodies to the sim2sim naming the policy was authored
  against (lever_door->LeverDoor_Leaf, lever_handle->LeverDoor_LeverHub,
  round_door->RoundDoor_Leaf, round_handle->RoundDoor_RoundKnob) plus their
  collision geoms so randomization/observation see consistent names;
- adds a stage3_camera_0 at the capture video's frame-0 pose;
- drops World keyframes/eef-target bodies that reference stripped joints.

No pipeline code changes; output is a standalone experiment scene.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

WORLD = Path("/home/wen/Documents/Codex/2026-11-07/phyagent-rsr-1-2-3-world-3/outputs/unidoor_task/scene/scene.xml")
OUT_DIR = Path("/home/wen/experiment/PhyAgent-RSR/output/unidoor_real2sim_20261108/world_experiment")
OUT = OUT_DIR / "scene.xml"

BODY_MAP = {
    "lever_door": "LeverDoor_Leaf",
    "lever_handle": "LeverDoor_LeverHub",
    "round_door": "RoundDoor_Leaf",
    "round_handle": "RoundDoor_RoundKnob",
    "lever_frame": "LeverDoor_Frame",
    "round_frame": "RoundDoor_Frame",
    "lever_mount": "LeverDoor_Mount",
    "round_mount": "RoundDoor_Mount",
    "lever_bolt": "LeverDoor_Bolt",
}

root = ET.parse(WORLD).getroot()
wb = root.find("worldbody")

# 1) robot subtree: robot0_base becomes the empty fr3_mount
base = next(b for b in wb.findall("body") if b.get("name") == "robot0_base")
base_pos, base_euler = base.get("pos"), base.get("euler")
wb.remove(base)
mount = ET.Element("body", {"name": "fr3_mount", "pos": base_pos,
                            "quat": "6.123233995736766e-17 0 0 -1"})  # Rz(-pi) as quat (fr3 validator rejects euler)
mount.append(base)  # keep embedded robot; prepare_sim_scene strips it and keeps mount attrs
base.set("pos", "0 0 0"); base.set("euler", "0 0 0"); base.set("name", "fr3_link0")
wb.insert(0, mount)

# remove eef target bodies (reference stripped robot joints)
for name in ("left_eef_target", "right_eef_target"):
    for b in wb.findall("body"):
        if b.get("name") == name:
            wb.remove(b)

# actuators/sensors/equalities referencing robot joints or sites
robot_names = set()
orig = ET.parse(WORLD).getroot()
for b in orig.iter("body"):
    n = b.get("name") or ""
    if n.startswith(("robot0_", "gripper0_")) or n in ("left_eef_target", "right_eef_target"):
        robot_names.add(n)
for j in orig.iter("joint"):
    if j.get("name"):
        robot_names.add(j.get("name"))
for s in orig.iter("site"):
    if s.get("name"):
        robot_names.add(s.get("name"))
for section in ("actuator", "sensor", "equality", "contact", "tendon"):
    el = root.find(section)
    if el is None:
        continue
    for item in list(el):
        refs = [item.get(a) for a in ("joint", "joint1", "joint2", "body", "body1", "body2",
                                      "site", "geom1", "geom2", "childclass")]
        if any(r in robot_names for r in refs if r):
            el.remove(item)
    if not len(el):
        root.remove(el)

# keyframes reference robot qpos layout that changes
kf = root.find("keyframe")
if kf is not None:
    root.remove(kf)

# 2) rename task bodies/joints to policy naming
def rename_deep(el, mapping):
    for b in el.iter("body"):
        n = b.get("name")
        if n in mapping:
            b.set("name", mapping[n])
for section in ("worldbody",):
    rename_deep(root.find(section), BODY_MAP)
joint_map = {k + "_hinge": v + "_joint" for k, v in
             (("lever_door", "LeverDoor_Leaf"), ("round_door", "RoundDoor_Leaf"),
              ("lever_handle", "LeverDoor_LeverHub"), ("round_handle", "RoundDoor_RoundKnob"))}
for j in root.iter("joint"):
    n = j.get("name")
    if n in joint_map:
        j.set("name", joint_map[n])
# bolt joint
for j in root.iter("joint"):
    if j.get("name") == "lever_bolt_joint":
        j.set("name", "LeverDoor_Bolt_joint")

# 3) stage3_camera_0: capture frame-0 pose from the Stage 3 camera manifest
import json
manifest = json.loads((Path("/home/wen/experiment/PhyAgent-RSR/output/unidoor_real2sim_20261108/stage3/camera_manifest.json")).read_text())
cam0 = next(c for c in manifest["cameras"] if c["name"] == "stage3_camera_0")
ET.SubElement(wb, "camera", {
    "name": "stage3_camera_0",
    "pos": " ".join(map(str, cam0["pos"])),
    "quat": " ".join(map(str, cam0["quat"])),
    "focalpixel": "327.847629 276.621437", "principalpixel": "0 0",
    "sensorsize": "36 24", "resolution": "512 288"})

OUT_DIR.mkdir(parents=True, exist_ok=True)
comp = root.find("compiler")
if comp is not None:
    v = comp.get("meshdir")
    if v and not Path(v).is_absolute():
        comp.set("meshdir", str(WORLD.parent / v))
    if not comp.get("texturedir"):
        comp.set("texturedir", comp.get("meshdir"))
ET.ElementTree(root).write(OUT, encoding="utf-8", xml_declaration=True)
print("wrote", OUT)

import mujoco
m = mujoco.MjModel.from_xml_path(str(OUT))
print("loads | nbody", m.nbody, "| cameras:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)][:4])
for probe in ("LeverDoor_Leaf", "RoundDoor_Leaf", "LeverDoor_LeverHub", "RoundDoor_RoundKnob"):
    print(probe, "->", m.body(probe).id)
print("fr3_mount pos:", m.body("fr3_mount").pos.round(3))
