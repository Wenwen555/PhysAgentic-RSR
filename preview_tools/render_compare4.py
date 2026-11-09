"""Side-by-side World vs sim2sim renders from the four reference camera poses.

Replicates task_packages/unidoor_sim2sim/prepare.py exactly: inject
adapter_cam_0..3 (pos/target, xyaxes, fovy 55) into each scene XML, reset
keyframe + mj_forward, render 512x288 with geomgroup[3]=0, sitegroup[:]=0.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json
import xml.etree.ElementTree as ET
import pathlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw

WORLD_XML = pathlib.Path("/home/wen/Documents/Codex/2026-11-07/phyagent-rsr-1-2-3-world-3/outputs/unidoor_task/scene/scene.xml")
SIM2SIM_XML = pathlib.Path("/home/wen/experiment/PhyAgent-RSR/output/unidoor_real2sim_20261108/stage3/scene.xml")
OUT_DIR = "/home/wen/experiment/PhyAgent-RSR/output/unidoor_real2sim_20261108/compare"

# Same poses prepare.py used for adapter_cam_0..3
VIEWS = [
    ((-1.20, 0.30, 1.65), (0.35, 0.30, 1.15)),
    ((-0.95, -1.15, 1.65), (0.35, -0.30, 1.15)),
    ((1.35, -1.20, 1.65), (0.25, -0.15, 1.10)),
    ((1.30, 1.25, 1.75), (0.25, 0.20, 1.10)),
]
WIDTH, HEIGHT = 512, 288
STRIP = 28


def load(xml_path, keyframe):
    """Load scene with absolute asset paths and the four adapter cameras injected."""
    root = ET.parse(xml_path).getroot()
    base_dir = xml_path.parent
    comp = root.find("compiler")
    has_dirs = False
    if comp is not None:
        for attr in ("meshdir", "texturedir"):
            v = comp.get(attr)
            if v and not pathlib.Path(v).is_absolute():
                comp.set(attr, str((base_dir / v).resolve()))
                has_dirs = True
    for el in root.findall(".//*[@file]"):
        p = pathlib.Path(el.get("file"))
        is_texture = p.suffix.lower() in (".png", ".jpg", ".jpeg")
        if has_dirs and not is_texture:
            continue  # mesh assets resolved via meshdir
        if not p.is_absolute():
            el.set("file", str((base_dir / p).resolve()))
    w = root.find("worldbody")
    for i, (pos, target) in enumerate(VIEWS):
        f = np.array(target) - np.array(pos)
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        ET.SubElement(w, "camera", name=f"adapter_cam_{i}",
                      pos=" ".join(map(str, pos)),
                      xyaxes=" ".join(map(str, np.r_[right, up])), fovy="55")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    kid = next((i for i in range(m.nkey)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) == keyframe), None)
    if kid is not None:
        mujoco.mj_resetDataKeyframe(m, d, kid)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    return m, d


def render_views(m, d):
    ren = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    imgs = []
    for i in range(4):
        ren.update_scene(d, camera=f"adapter_cam_{i}", scene_option=opt)
        imgs.append(ren.render())
    ren.close()
    return imgs


def label(img, text):
    out = Image.new("RGB", (img.shape[1], img.shape[0] + STRIP), (24, 24, 24))
    out.paste(Image.fromarray(img), (0, STRIP))
    ImageDraw.Draw(out).text((8, 7), text, fill=(255, 255, 255))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wm, wd = load(WORLD_XML, "home")
    sm, sd = load(SIM2SIM_XML, "home")
    print("World nbody:", wm.nbody, "| sim2sim nbody:", sm.nbody)
    w_imgs = render_views(wm, wd)
    s_imgs = render_views(sm, sd)
    # sanity: World renders should match the stored reference frames
    ref_dir = pathlib.Path("/home/wen/experiment/PhyAgent-RSR/output/unidoor_sim2sim_20261107/"
                           "calibration_corrected_evaluation/reference/reference")
    if ref_dir.exists():
        diffs = []
        for i in range(4):
            ref = np.asarray(Image.open(ref_dir / f"frame_{i:02}.jpg").convert("RGB"), dtype=np.int16)
            diffs.append(float(np.abs(ref - w_imgs[i].astype(np.int16)).mean()))
        print("World vs stored reference MAE per view:", [round(x, 2) for x in diffs])
    for i in range(4):
        combo = Image.new("RGB", (WIDTH * 2 + 4, HEIGHT + STRIP), (0, 0, 0))
        combo.paste(label(w_imgs[i], f"World (ground truth)  -  camera_{i}"), (0, 0))
        combo.paste(label(s_imgs[i], "Sim2Sim (agentic RSR, diagnostic)"), (WIDTH + 4, 0))
        out = f"{OUT_DIR}/camera_{i}_side_by_side.png"
        combo.save(out)
        print("saved", out)


if __name__ == "__main__":
    main()
