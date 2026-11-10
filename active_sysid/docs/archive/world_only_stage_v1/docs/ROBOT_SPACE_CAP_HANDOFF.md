# Handoff to a new DevSpace session: Robot-Space CaP

**Task for the next session:** write and evaluate CaP strategies. The R0/R1 native double-door execution layer is complete. Do not rebuild cameras, swap robots, edit the World XML, or modify the fixed control/evaluation baseline to improve a policy score.

Project root: `/home/wen/experiment/PhyAgent-RSR`
Python package root: `/home/wen/experiment/PhyAgent-RSR/active_sysid`

## Frozen execution contract

- Source double-door World + all local mesh/texture assets: `assets/worlds/unidoor_reference/scene.xml` and `resources/`
- Native robot, gripper, handle/bolt, and door mapping: `configs/robot/world_bindings/unidoor_reference.yaml`
- Shared controller, runtime, simulator binding, evaluator: `src/active_sysid/robot/{controller,adapter,runtime,world_backend,world_runtime,world_evaluator}.py`
- Reference fixed program: `src/active_sysid/robot/world_fixed_policy.py` (privileged fixed R1; **not a visual CaP**)
- SHA256 source freeze: `configs/robot/frozen_r1.json`, baseline id `unidoor_r1_native_robot_v1`
- Stage configs: `configs/robot/stages/{privileged_exact,visual_exact,visual_randomized}.yaml`
- Runner: `src/active_sysid/robot/stages.py`
- New strategy files: **only** `policies/robot_space/*.py`

Use these commands from `active_sysid/`:

```bash
uv sync --locked --extra sim --extra robot --extra opt --extra dev
.venv/bin/python -m active_sysid.robot.stages check
.venv/bin/python -m active_sysid.robot world-inspect
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot world-baseline \
    --out runs/m3r_real_world/r1_recheck --video
```

`stages check` verifies 126 SHA256-pinned files and the Python/MuJoCo/NumPy runtime. Changed baseline code, assets, stage configs, or runtime versions cause the stage runner to reject execution. To intentionally release a **new version** of the baseline, run the physical R1 tests and controls first, then explicitly regenerate the manifest using `python -m active_sysid.robot.stage_freeze write`; retain the prior manifest and evidence for comparability. The current worktree has not been Git-committed.

## The exact three-stage contract

| Stage | Policy observation | World model | Default seeds |
|---|---|---|---|
| `privileged_exact` | robot proprioception/contact + `observation['privileged']` object poses/joints | Original native World, exact | `[0]` |
| `visual_exact` | RGB on request, robot proprioception, measured contacts, **no privileged state** | Same native World, exact | `[0]` |
| `visual_randomized` | Same visual-only interface as `visual_exact` | Same geometry/robot/controller; seed-dependent damping, friction, stiffness and light intensity | `[0..7]` |

Stage 3 uses deterministic `numpy.random.default_rng(seed)` draws only on the in-memory model. The exact ranges are recorded in `configs/robot/stages/visual_randomized.yaml`. Evaluator-private draws are saved under each episode's `metrics.json → provenance`, **not included in policy observations**. Stage 3 is a modest physics + illumination variation, not a large-scale scene-layout randomization benchmark. Override the default seeds with `--seeds 0 1 ...` for a larger predeclared evaluation set.

## Policy interface

Each new CaP file must expose `def run(api): ...`. `api` provides:

```python
api.observe(rgb=False, camera='overview')       # arrays of q, dq, ee pose, wrench, gripper, contacts
api.observe(rgb=True, camera='overview')        # adds rgb: H×W×3 uint8
api.observe(rgb=True, camera='robot0_eye_in_hand')
api.phase('approach')                            # label only; not a robot macro
api.move_joints(positions_rad, duration_s=2.0)  # 7 arm joint targets, radians
api.move_ee(position_m, quaternion_wxyz, duration_s=1.0)
api.move_arc(center_m, axis_world, angle_rad, duration_s=2.0)
api.set_gripper(width_m, duration_s=0.6)
api.hold(duration_s)
```

All distances in meters, times in seconds, orientations in wxyz quaternion order, and Cartesian positions in World coordinates. Allowed visual cameras: `overview` and `robot0_eye_in_hand`. `visual_exact` and `visual_randomized` reject other camera requests and omit `observation['privileged']`. The evaluator privately reads real joint states and contact forces to score the task, including physical handle contact, unlocking, door angle, force constraints, and no direct object actuation. Do not trust a policy's own `success` flag.

Visual-stage task acceptance also requires **at least two RGB observations at distinct simulation times**. Their camera names and timestamps are recorded in `metrics.json → rgb_observations`. This prevents purely open-loop scripts with zero image reads from being counted as vision-based; it does not prove that a strategy meaningfully used the pixels.

No direct `door_hinge` / `handle_hinge` / `bolt_joint` command exists in the Robot API. Coding Agent programs must not use `api._backend`, `api._adapter`, MuJoCo model/data, private evaluation outputs, stage randomization draws or source-World hardcoded ground-truth poses in the visual stages. **This is an interface-level research contract, not an isolated process security sandbox:** Python policy files are imported into the evaluator process. A stronger adversarial isolation layer is future work.

## Evaluating new CaP files

Write only the new policy modules, e.g.:

```text
policies/robot_space/privileged_policy.py
policies/robot_space/visual_policy.py
```

First, run each stage independently:

```bash
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run \
  --stage privileged_exact --policy policies/robot_space/privileged_policy.py \
  --out runs/robot_cap/privileged_exact
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run \
  --stage visual_exact --policy policies/robot_space/visual_policy.py \
  --out runs/robot_cap/visual_exact
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run \
  --stage visual_randomized --policy policies/robot_space/visual_policy.py \
  --out runs/robot_cap/visual_randomized
```

Then use **one comparison command**, which defaults to reusing the *identical* visual policy SHA256 in stages 2 and 3:

```bash
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.stages run-all \
  --privileged-policy policies/robot_space/privileged_policy.py \
  --visual-policy policies/robot_space/visual_policy.py \
  --out runs/robot_cap/comparison_v1
```

Only use `--randomized-policy` if explicitly testing a revised randomized-stage strategy; the comparison records `visual_policy_unchanged_in_randomized: false` in that case. Do not silently mix strategy revisions. Add `--video` to record every episode; by default, only numeric telemetry is recorded to keep evaluations fast. `summary.json` reports seed-level results and success rate; `comparison.json` reports stage-level comparison; individual `seed_0000/metrics.json`, `actions.json` and `trajectory.npz` preserve evidence and hashes.

The default randomized seed list is a starting harness, **not** sufficient by itself to support claims of robust statistical superiority. Predeclare 32+ seeds and multiple generation replicates before final comparison.

## Evidence already established

Reference fixed baseline in exact native World: final lever door angle **0.399582 rad** after **21.1 s**, measured two-finger contact during rotate/pull **100%**, direct object actuation **zero**; all R1 required gates pass. No-rotate control leaves door shut. The fixed program is deliberately privileged and contains an exact-World pregrasp waypoint; do not use its output as proof of visual robotics capability.

## Prompt for the next session

> Use DevSpace MCP at `/home/wen/experiment/PhyAgent-RSR`. Read `active_sysid/docs/ROBOT_SPACE_CAP_HANDOFF.md` and verify `python -m active_sysid.robot.stages check`. Write only new Robot-Space Code-as-Policy strategies in `active_sysid/policies/robot_space/` implementing `run(api)`. Start with `privileged_exact`, then develop a genuine RGB-feedback `visual_exact` strategy, finally run the same visual policy unmodified on `visual_randomized` seeds. Do not modify World assets, frozen R1 controller/evaluator, fixed reference baseline, or stage configurations merely to make policies pass. Record execution traces and stage result SHA256. Treat privileged geometry as unavailable to visual CaP.
