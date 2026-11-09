"""One-off capture driver: record a handheld-style walkthrough video of the World scene.

Renders the pinned World unidoor scene (closed doors, robot at home keyframe)
along a smooth orbit path with mild handheld perturbation, encodes with ffmpeg
h264. Official pipeline code is untouched; this only drives MuJoCo offscreen.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import math
import subprocess
import numpy as np
import mujoco
from PIL import Image

WORLD_XML = "/home/wen/Documents/Codex/2026-11-07/phyagent-rsr-1-2-3-world-3/outputs/unidoor_task/scene/scene.xml"
OUT_MP4 = "/home/wen/experiment/PhyAgent-RSR/output/world_capture_20261108/unidoor_walkthrough_v5.mp4"
FRAME_DIR = "/home/wen/experiment/PhyAgent-RSR/output/world_capture_20261108/samples_v5"
AZ_START, AZ_END = -75, 35  # avoid door-edge silhouette overlap at |az|>40 seen in v3
# Registration template pose (fr3_d435i_official.urdf fixed pose in robot_surface.py);
# the captured arm must match it or rigid Sim(3) registration is capped.
TEMPLATE_QPOS = {"robot0_joint%d" % i: v for i, v in
                 enumerate((0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4), 1)}
TEMPLATE_QPOS.update({"gripper0_finger_joint1": 0.04, "gripper0_finger_joint2": 0.04})

W, H, FPS, SECONDS = 1280, 720, 30, 12.0
N = int(FPS * SECONDS)  # 360 frames

m = mujoco.MjModel.from_xml_path(WORLD_XML)
m.vis.global_.offwidth = W
m.vis.global_.offheight = H
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
for name, value in TEMPLATE_QPOS.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise SystemExit(f"joint not found: {name}")
    d.qpos[m.jnt_qposadr[jid]] = value
mujoco.mj_forward(m, d)

ren = mujoco.Renderer(m, height=H, width=W)
opt = mujoco.MjvOption()
opt.sitegroup[:] = 0  # hide debug sites (e.g. the 10 m grip-site cylinder)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

CENTER = np.array([0.30, 0.0, 1.02])   # between both doors and the robot
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
     "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-crf", "18", OUT_MP4],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

os.makedirs(FRAME_DIR, exist_ok=True)
keep = set(np.linspace(0, N - 1, 8).astype(int))

for i in range(N):
    t = i / (N - 1)
    # walk-in during the first 1.5 s, then one slow 250-degree orbit
    ease = min(1.0, i / (1.5 * FPS))
    dist = 2.05 - 0.25 * ease - 0.08 * math.sin(2 * math.pi * t * 0.5)
    az = AZ_START + (AZ_END - AZ_START) * t
    el = -14 - 3 * math.sin(2 * math.pi * t * 0.7)
    # handheld perturbation: two slow incommensurate wobbles
    az_n = 0.9 * math.sin(2 * math.pi * 0.83 * t + 0.7) + 0.4 * math.sin(2 * math.pi * 2.1 * t)
    el_n = 1.4 * math.sin(2 * math.pi * 1.13 * t + 2.1) + 0.6 * math.sin(2 * math.pi * 2.7 * t + 1.0)
    bob = 0.018 * math.sin(2 * math.pi * 1.7 * t) + 0.008 * math.sin(2 * math.pi * 3.1 * t + 0.5)
    center = CENTER + np.array([0.02 * math.sin(2 * math.pi * 0.9 * t), bob * 0.4, bob])
    cam.lookat[:] = center
    cam.distance = dist + 0.02 * math.sin(2 * math.pi * 1.5 * t)
    cam.azimuth = az + az_n
    cam.elevation = el + el_n
    ren.update_scene(d, camera=cam, scene_option=opt)
    frame = ren.render()
    proc.stdin.write(frame.tobytes())
    if i in keep:
        Image.fromarray(frame).save(f"{FRAME_DIR}/frame_{i:04}.jpg", quality=92)

proc.stdin.close()
proc.wait()
ren.close()

# contact sheet for quick review
imgs = [Image.open(f"{FRAME_DIR}/frame_{i:04}.jpg").resize((426, 240)) for i in sorted(keep)]
sheet = Image.new("RGB", (426 * 4 + 6, 240 * 2 + 4), (0, 0, 0))
for k, im in enumerate(imgs):
    sheet.paste(im, ((k % 4) * 426 + (k % 4) * 2, (k // 4) * 240 + (k // 4) * 2))
sheet.save("/home/wen/experiment/PhyAgent-RSR/output/world_capture_20261108/contact_sheet_v5.jpg", quality=90)
print("frames:", N, "| mp4:", OUT_MP4)
