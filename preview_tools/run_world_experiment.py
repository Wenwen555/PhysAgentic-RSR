"""Run the approved visual_exact policy on the World experiment scene.

Uses the official runtime pieces verbatim (Stage3Scene, Stage3RGBDObservation,
Stage3Tools, run_program) against runs/sim_processed_stage3_all/franka/
world_experiment/scene_for_sim.xml. Adds only:
- a fixed-rate World-side third-person capture (sideview camera);
- door hinge telemetry before/after each dispatch for quantified reporting.
"""
import json
import sys
from pathlib import Path

REPO = Path("/home/wen/experiment/PhyAgent-RSR/agentic_rsr")
sys.path.insert(0, str(REPO / "src"))

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SAM3_BACKEND", "local")
os.environ.setdefault("SAM3_DEVICE", "cuda")

OUT = Path("/home/wen/experiment/PhyAgent-RSR/output/unidoor_real2sim_20261108/world_experiment/run_privileged")
FRAMES = OUT / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)

from sim.stage3_scene import Stage3Scene
from sim.stage3_harness import Stage3Tools
from sim.stage3_observation import Stage3RGBDObservation
from sim.program_runner import run_program
from perception.config import sam3_endpoint as configured_endpoint

SCENE = REPO / "runs/sim_processed_stage3_all/franka/world_experiment/scene_for_sim.xml"
POLICY = (REPO / "runs/sim_processed_stage3_all/franka/unidoor_walkthrough_v5/"
          "codex_three_stage_v1/privileged_exact/iterations/011/policy.py").read_text()

NAMES = ["LeverDoor_Leaf", "RoundDoor_Leaf", "LeverDoor_LeverHub", "RoundDoor_RoundKnob"]
INSTRUCTION = ("Open each of two closed doors separately by operating its handle "
               "and pulling the door.")

def main():
    scene = Stage3Scene(SCENE, "franka")
    host, port = configured_endpoint()
    perception = Stage3RGBDObservation(scene, tuple(NAMES), OUT / "observations", host, port)
    tools = Stage3Tools(scene, "unidoor_walkthrough_v5", INSTRUCTION,
                        perception=perception, privileged=True)

    model, data = scene.model, scene.data
    hinge_names = {"LeverDoor_Leaf": "LeverDoor_Leaf_joint", "RoundDoor_Leaf": "RoundDoor_Leaf_joint",
                   "LeverDoor_LeverHub": "LeverDoor_LeverHub_joint", "RoundDoor_RoundKnob": "RoundDoor_RoundKnob_joint"}

    def door_angles():
        out = {}
        for body, joint in hinge_names.items():
            jid = model.joint(joint)
            out[body] = round(float(data.qpos[jid.qposadr[0]]), 4)
        return out

    frame_id = {"n": 0}
    renderer = {"r": None}

    def world_capture(label):
        import numpy as np
        from PIL import Image, ImageDraw
        if renderer["r"] is None:
            import mujoco
            model.vis.global_.offwidth = 800
            model.vis.global_.offheight = 450
            renderer["r"] = mujoco.Renderer(model, height=450, width=800)
        renderer["r"].update_scene(data, camera="sideview")
        img = renderer["r"].render()
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 800, 28), fill=(0, 0, 0))
        draw.text((8, 8), f"World experiment | visual_exact policy | {label} | angles {door_angles()}", fill=(255, 255, 255))
        image.save(FRAMES / f"frame_{frame_id['n']:05d}.png")
        frame_id["n"] += 1

    def dispatch(name, *args):
        before = door_angles()
        world_capture(f"{name} begin")
        result = tools.dispatch(name, *args)
        world_capture(f"{name} end")
        print(f"[{name}] args={str(args)[:80]} doors_before={before} doors_after={door_angles()}", flush=True)
        return result

    world_capture("episode start")
    result = run_program(POLICY, dispatch, timeout=600, interface="stage3")
    world_capture("episode end")
    print("PROGRAM RESULT:", json.dumps(result)[:400])

    scene.set_step_callback(None) if hasattr(scene, "set_step_callback") else None
    report = {"program": result, "final_angles": door_angles(), "calls": len(tools.calls)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
    perception.close()
    scene.close()
    print("REPORT:", json.dumps(report)[:400])


if __name__ == "__main__":
    main()
