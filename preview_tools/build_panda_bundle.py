"""Build a robot-only Panda MJCF bundle from the pinned World scene.

Extracts the robot0_base subtree (arm + gripper), its meshes/materials,
the World actuators/sensors, and a robot-only reset keyframe. The base is
rebased to pos 0 0 0 (orientation kept) so the Stage 3 exporter can mount it
and --world-offset can place the scene in World coordinates.
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

WORLD = Path("/home/wen/Documents/Codex/2026-11-07/phyagent-rsr-1-2-3-world-3/outputs/unidoor_task/scene/scene.xml")
OUT = Path("/home/wen/experiment/PhyAgent-RSR/output/unidoor_real2sim_20261108/panda_robot_only")

root = ET.parse(WORLD).getroot()
worldbody = root.find("worldbody")
robot_base = next(b for b in worldbody.findall("body") if b.get("name") == "robot0_base")

# --- collect robot subtree geoms and their asset references ---
robot_geoms = robot_base.findall(".//geom")
used_meshes, used_materials, used_textures = {}, {}, {}
for g in robot_geoms:
    if g.get("mesh"):
        used_meshes[g.get("mesh")] = None
    if g.get("material"):
        used_materials[g.get("material")] = None

asset_el = root.find("asset")
for m in asset_el.findall("mesh"):
    if m.get("name") in used_meshes:
        used_meshes[m.get("name")] = m
for mat in asset_el.findall("material"):
    if mat.get("name") in used_materials:
        used_materials[mat.get("name")] = mat
        if mat.get("texture"):
            used_textures[mat.get("texture")] = None
for t in asset_el.findall("texture"):
    if t.get("name") in used_textures:
        used_textures[t.get("name")] = t

# --- new document ---
new = ET.Element("mujoco", {"model": "panda_robot_only"})
ET.SubElement(new, "compiler", {"angle": "radian", "autolimits": "true",
                                "meshdir": "assets", "texturedir": "assets"})
ET.SubElement(new, "option", {"timestep": "0.002"})

new_asset = ET.SubElement(new, "asset")
for el in list(used_textures.values()) + list(used_materials.values()) + list(used_meshes.values()):
    if el is not None:
        new_asset.append(ET.fromstring(ET.tostring(el)))

# rebase robot0_base to origin, keep orientation
new_world = ET.SubElement(new, "worldbody")
new_base = ET.fromstring(ET.tostring(robot_base))
new_base.set("pos", "0 0 0")
new_world.append(new_base)

# actuators: World's torq arm + gripper actuators, by name
WANT_ACTUATORS = [f"robot0_torq_j{i}" for i in range(1, 8)] + [
    "gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"]
world_actuators = {a.get("name"): a for a in root.find("actuator").findall("position")}
new_act = ET.SubElement(new, "actuator")
for name in WANT_ACTUATORS:
    new_act.append(ET.fromstring(ET.tostring(world_actuators[name])))

# sensors
WANT_SENSORS = ["gripper0_force_ee", "gripper0_torque_ee"]
world_sensors = {s.get("name"): s for s in root.find("sensor")}
new_sens = ET.SubElement(new, "sensor")
for name in WANT_SENSORS:
    new_sens.append(ET.fromstring(ET.tostring(world_sensors[name])))

# robot-only reset keyframe: appended after mujoco computes qpos/ctrl layout
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "assets").mkdir(exist_ok=True)

# fix mesh/texture file paths into assets/ and copy files
src_dir = WORLD.parent
for el in new_asset.iter("mesh"):
    f = el.get("file")
    if f:
        if str(Path(f)).startswith("assets/"):
            continue
        shutil.copy2(src_dir / f, OUT / "assets" / Path(f).name)
        el.set("file", Path(f).name)
for el in new_asset.iter("texture"):
    f = el.get("file")
    if f:
        if str(Path(f)).startswith("assets/"):
            continue
        shutil.copy2(src_dir / f, OUT / "assets" / Path(f).name)
        el.set("file", Path(f).name)

xml_bytes = ET.tostring(new, encoding="unicode")
(OUT / "model.xml").write_text(xml_bytes)

# append keyframe with proper qpos size via mujoco
import mujoco
m = mujoco.MjModel.from_xml_path(str(OUT / "model.xml"))
d = mujoco.MjData(m)
names = {}
for i in range(m.njnt):
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    names[jn] = m.jnt_qposadr[i]
WORLD_Q = {"robot0_joint1": 0.0, "robot0_joint2": 0.19634954084936207,
           "robot0_joint3": 0.0, "robot0_joint4": -2.617993877991494,
           "robot0_joint5": 0.0, "robot0_joint6": 2.941592653589793,
           "robot0_joint7": 0.7853981633974483,
           "gripper0_finger_joint1": 0.0, "gripper0_finger_joint2": 0.0}
qpos = [0.0] * m.nq
for jn, v in WORLD_Q.items():
    qpos[names[jn]] = v
mujoco.mj_forward(m, d)
ctrl = [0.0] * m.nu
WORLD_CTRL = {"robot0_torq_j1": 0.0, "robot0_torq_j2": 0.19634954084936207,
              "robot0_torq_j3": 0.0, "robot0_torq_j4": -2.617993877991494,
              "robot0_torq_j5": 0.0, "robot0_torq_j6": 2.941592653589793,
              "robot0_torq_j7": 0.7853981633974483,
              "gripper0_gripper_finger_joint1": 0.006,
              "gripper0_gripper_finger_joint2": -0.006}
for i in range(m.nu):
    an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    ctrl[i] = WORLD_CTRL.get(an, 0.0)
kf = ET.Element("key", {"name": "home", "qpos": " ".join(map(repr, qpos)),
                        "ctrl": " ".join(map(repr, ctrl))})
kfwrap = ET.Element("keyframe")
kfwrap.append(kf)
tree = ET.parse(OUT / "model.xml")
tree.getroot().append(kfwrap)
tree.write(OUT / "model.xml")

m2 = mujoco.MjModel.from_xml_path(str(OUT / "model.xml"))
print("bundle OK: nq=%d nu=%d nsensor=%d nkey=%d nbody=%d" %
      (m2.nq, m2.nu, m2.nsensor, m2.nkey, m2.nbody))
print("sensors:", [mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m2.nsensor)])
