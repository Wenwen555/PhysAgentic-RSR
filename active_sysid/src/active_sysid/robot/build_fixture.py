"""Rebuild the checked-in M3-R scenes from the pinned Menagerie Panda assets."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
ROBOT = ROOT / "assets/fixtures/panda_single_door"


def build():
    tree = ET.parse(ROBOT / "franka_emika_panda/panda.xml")
    root = tree.getroot()
    root.set("model", "panda_robot_execution")
    root.find("compiler").set("meshdir", "franka_emika_panda/assets")
    root.remove(root.find("keyframe"))
    root.find("worldbody/body").set("pos", "1.0 0.8 0.30")
    root.find("worldbody/body").set("quat", "0.48765486 0 0 -0.87303650")
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="tcp", pos="0 0 0.1034", size="0.006", rgba="0 1 0 0")
    ET.SubElement(hand, "site", name="wrist_ft", pos="0 0 0", size="0.004", rgba="0 0 0 0")
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_finger']")
        for i, geom in enumerate(body.findall("geom")):
            geom.set("name", f"{side}_finger_geom_{i}")
            if geom.get("class") != "visual":
                geom.set("friction", "1.0 0.005 0.0001")
                geom.set("condim", "4")
                geom.set("solref", "0.008 1")
                geom.set("solimp", "0.95 0.99 0.001")
    kp = [800, 800, 700, 700, 350, 250, 200]
    kd = [70, 70, 60, 60, 30, 25, 20]
    actuators = root.find("actuator")
    for i in range(7):
        actuator = actuators[i]
        actuator.set("gainprm", str(kp[i]))
        actuator.set("biasprm", f"0 {-kp[i]} {-kd[i]}")
    actuators[7].set("gainprm", str(0.04 * 4000 / 255))
    actuators[7].set("biasprm", "0 -4000 -40")
    actuators[7].set("forcerange", "-60 60")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "force", name="wrist_force", site="wrist_ft")
    ET.SubElement(sensors, "torque", name="wrist_torque", site="wrist_ft")
    ET.indent(root)
    tree.write(ROBOT / "panda_robot.xml", encoding="unicode")
    for oracle in (True, False):
        name = "oracle" if oracle else "reconstructed"
        door_damping, door_friction, density = ((0.4, 0.1, 187.5) if oracle else (0.1, 0.05, 3.9))
        handle_damping, handle_stiffness = ((0.08, 0.5) if oracle else (0.02, 0.0))
        xml = f'''<mujoco model="{name}_robot_door">
  <include file="panda_robot.xml"/>
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" iterations="100" tolerance="1e-9"/>
  <visual>
    <global offwidth="960" offheight="720"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="ground" type="2d" builtin="checker" rgb1="0.28 0.32 0.36" rgb2="0.22 0.26 0.30" width="256" height="256"/>
    <material name="groundmat" texture="ground" texrepeat="4 4" texuniform="true" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0.5 0.8 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="2.5 3.0 2.3" xyaxes="-0.80 0.60 0 -0.25 -0.33 0.91" fovy="48"/>
    <camera name="contact" pos="1.40 1.30 1.55" xyaxes="-0.82 0.57 0 -0.23 -0.33 0.91" fovy="42"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="groundmat"/>
    <geom name="pedestal" type="box" pos="1.0 0.8 0.15" size="0.16 0.16 0.15" rgba="0.17 0.20 0.24 1"/>
    <body name="frame">
      <geom name="post" type="box" size="0.05 0.05 1.0" pos="-0.06 0 1.0" rgba="0.19 0.23 0.28 1"/>
      <body name="door">
        <joint name="door_hinge" type="hinge" axis="0 0 1" range="0 {1.508 if oracle else 1.7}" damping="{door_damping}" frictionloss="{door_friction}"/>
        <geom name="door_leaf" type="box" size="0.4 0.02 1.0" pos="0.4 0 1.05" density="{density}" rgba="0.26 0.48 0.62 1"/>
        <!-- A normal-axis lever in front of the panel. The old object scene is unchanged. -->
        <geom name="handle_mount" type="cylinder" size="0.027 0.035" pos="0.70 0.055 1.0" quat="0.70710678 0.70710678 0 0" mass="0" rgba="0.45 0.48 0.52 1"/>
        <body name="handle" pos="0.70 0.105 1.0">
          <!-- Counterbalanced lever: COM on its axis preserves the old zero gravity torque. -->
          <inertial pos="0 0 0" mass="0.12" diaginertia="0.00004 0.00018 0.00018"/>
          <joint name="handle_hinge" type="hinge" axis="0 -1 0" range="-1.2 1.2" damping="{handle_damping}" stiffness="{handle_stiffness}" springref="0" armature="0.01"/>
          <geom name="handle_bar" type="capsule" fromto="0 0 0 -0.15 0 0" size="0.014" contype="1" conaffinity="1" condim="4" friction="1.0 0.005 0.0001" solref="0.008 1" solimp="0.95 0.99 0.001" rgba="0.88 0.66 0.23 1"/>
          <site name="handle_grasp" pos="-0.10 0 0" size="0.004" rgba="1 0.4 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld name="latch_weld" body1="door" body2="frame" active="{'true' if oracle else 'false'}" solref="0.01 1" solimp="0.98 0.999 0.001"/>
  </equality>
</mujoco>
'''
        (ROBOT / f"{name}_robot_door.xml").write_text(xml, encoding="utf-8")

if __name__ == "__main__":
    build()
