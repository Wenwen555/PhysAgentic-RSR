# 当前边界说明

本页记录原始 World 的 Oracle R1 执行验收。**CaP 开发已经改为独立 Sim**，新入口与权限以 [SIM_WORLD_BOUNDARY.md](SIM_WORLD_BOUNDARY.md) 为准。以下 R1 成功不是 Sim2Sim 迁移成绩。

# M3-R: Robot-space execution in the original double-door World

M0–M2 **object-space** physics experiments and the Panda **single-door fixture** remain independent. R1 additionally runs in the original textured *double-door World* with its native 7-DOF robot, native two-finger gripper, original handle/bolt joint coupling and original cameras. No door/handle object actuator, weld grasp or mocap robot controller is added.

## Source and binding

- Source model and resources: `assets/reference_world/unidoor_native/scene.xml`, `assets/reference_world/unidoor_native/resources/` (111 source files matched the source location byte-for-byte on 2026-11-10).
- Named robot/door contract: `configs/robot/world_bindings/unidoor_native.yaml`.
- Robot motion: `src/active_sysid/robot/controller.py` (common IK, joint-space transit, Cartesian motion, finger servo), `world_backend.py` (source model), `world_runtime.py` (same action/observation API).
- Explicit policy: `src/active_sysid/robot/reference_oracle_policy.py`. Its fixed pregrasp joint waypoint is specialized to this *exact* World configuration.
- Physics-trace evaluator: `src/active_sysid/robot/reference_world_evaluator.py`. Evaluates measured contacts, force/torque, slip, handle/bolt/door angles and forbidden object actuation, separately from policy-returned success.
- Panda fixture: `assets/fixtures/panda_single_door/` and `src/active_sysid/robot/fixture_backend.py`; not used in source-World R1.

## Environment and commands

From `active_sysid/`:

```bash
# Recommended consistent environment; source assets are already local.
uv sync --locked --extra sim --extra robot --extra opt --extra dev

# Compile native World and inspect the exact actuator/robot/door bindings.
.venv/bin/python -m active_sysid.robot world-inspect --out runs/m3r_real_world/inspection

# R0: approach, reach and bilateral grip.
.venv/bin/python -m active_sysid.robot world-smoke --out runs/m3r_real_world/r0

# R1: real joint execution plus native contact interaction, dual camera video.
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot world-baseline --out runs/m3r_real_world/r1 --video

# Controlled negative result: omit handle rotation while keeping grasp and pull.
.venv/bin/python -m active_sysid.robot world-ablate-no-rotate --out runs/m3r_real_world/ablation_no_rotate

# CaP entry point (user module must expose run(api)), robot observations by default.
.venv/bin/python -m active_sysid.robot world-evaluate --bundle /absolute/path/to/frozen_policy \
    --out runs/m3r_real_world/frozen_eval
```

The fixture stays available as `python -m active_sysid.robot smoke` and `... baseline`, without `world-` prefix. The original World is explicitly labeled `reference_world` in every World evaluation result. `world-evaluate` accepts only frozen Sim visual-policy bundles; privileged World evaluation is disabled for CaP.

## R1 acceptance and recorded evidence

Measured on 2026-11-10 (MuJoCo 4.0.0, timestep 0.002 s):

| Quantity | World R1 | Required |
|---|---:|---:|
| Final door angle | 0.399582 rad | >= 0.30 rad |
| Handle rotation / bolt motion | peak 0.684799 rad / 0.027288 m | >= 0.55 rad / >= 0.020 m |
| Bilateral finger contact during rotate and pull | 100% | >= 95% |
| Handle-in-TCP translation drift | 0.002388 m | <= 0.010 m |
| Handle-in-TCP orientation drift | 0.122478 rad | <= 0.150 rad |
| Aggregate finger-handle contact-force magnitude | peak 21.066 N | <= 80 N (experiment cutoff) |
| Unexpected robot/world or self contact | 0 N | <= 5 N |
| Direct applied object torque/force | 0 | zero |
| Time | 10,550 physics steps = 21.1 s | step budget <= 30,000 |
| Acceptance | PASS | all required gates |

These force limits are **experimental cutoffs, not established safe damage thresholds**. The original World has finite-compliance limits: its handle-bolt equality can lead to small bolt-travel overshoot beyond the nominal `.025 m` joint range. See the recorded `coupling_residual_m` and joint motion when making stronger safety claims.

Outputs under `runs/m3r_real_world/r1/` are `metrics.json`, `actions.json`, every-physics-step `trajectory.npz`, `overview.mp4`, `closeup.mp4`, and first/last PNGs. Video: 528 frames per camera at 25 fps, 640x480; the 21.12 s media duration agrees with 21.1 s simulation time to within one frame. Original scene XML/resource files were never changed by control or evaluation.

Two independent negative-control checks:

- Omitting handle rotation yields a final door angle effectively zero, with R1 FAIL.
- Disabling handle collision **in memory for a test only** prevents the two-finger grasp check; no source asset is modified.

## Scope and next stage

This is **R0/R1 privileged fixed-program execution**, not yet the new Coding-Agent CaP / visual randomized stage. Source-World robot joint control and physical contact are established. Next: freeze this policy and independently evaluate `privileged_exact` versus `visual_exact` versus `visual_randomized` with identical robot, hardware/contact settings, and no privileged object angles in visual policies. Arbitrary dynamically imported CaP code is not yet sandboxed and should not be treated as trusted third-party code.

## Frozen CaP handoff (new session)

The native-World R1 execution baseline has been frozen into `configs/robot/frozen_r1.json`, as historical World-only v1. It has been superseded by the explicitly separated Sim/World v2 stage runner. Do not modify the original World or the fixed reference policy to write new CaP. **Read [ROBOT_SPACE_CAP_HANDOFF.md](ROBOT_SPACE_CAP_HANDOFF.md) before building CaP.** The three Stage runners and exact experimental contract are documented there.
