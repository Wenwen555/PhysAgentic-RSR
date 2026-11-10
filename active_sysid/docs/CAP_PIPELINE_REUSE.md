# Reusing the previous CaP Stage 1–3 work

## Decision

**Reuse the experimental protocol, not the old object-space policy code.** The earlier deliveries live in local ignored `runs/policy-develop/` and were written against `api.step(handle=(...), door=(...))` with directly actuated hinges and an already-extracted `obs['vision']['door']['angle_rad']`. The current Robot-Space API instead uses seven arm joints, real gripper contacts, RGB frames, wrist wrench and `move_ee / move_arc / move_joints`. The old policy cannot run unchanged or demonstrate robot-space transfer.

| Old capability | Decision | Current Robot-Space implementation |
|---|---|---|
| Stage 1 bounded development revisions | Reuse | `robot/cap_protocol.py` tracks numbered revisions; configurable maximum 3 |
| Stage 2 RGB vs privileged observation boundary | Reuse the idea | `boundary_runtime.py`, isolated `policy_worker.py` / `policy_rpc.py` |
| Stage 3 deterministic seed draws | Reuse | `stage_randomization.py` and `stages.py`, only in reconstructed Sim |
| Separate development vs independent holdout seeds | Reuse | `configs/robot/cap_protocol.yaml`: 20 dev, 30 held-out seeds |
| Frozen policy and SHA/provenance | Reuse | `cap_protocol freeze`, `sim2sim freeze`, `stage_freeze.py` |
| Stage 2 blue-door RGB angle estimator | Do not transplant | Wrong scene/model; new visual CaP must estimate actual task state from rendered robot scene |
| Stage 1/3 direct handle-position + door-force control | Do not transplant | Robot actions are physical joint/EE/gripper commands; no object actuators |
| Object-Space Stage 3 randomized physics and old acceptance metrics | Keep as historical | Robot-Space randomization and private task evaluator operate on task-grounded Sim |

Earlier Stage 2 delivery explicitly reported that it had **not** passed M2 acceptance in that environment. Stage 3 was a candidate with no validated 30-seed acceptance at delivery. Preserve those distinctions rather than assigning them new successful run labels.

## Stage 1–3 Robot-Space CaP development protocol

Code authors should create **new** self-contained `run(api)` files in `policies/robot_space/`, then use the existing runner:

```bash
cd active_sysid
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.cap_protocol develop \
  --experiment runs/cap_robot_v1 --revision 0 \
  --policy policies/robot_space/visual_policy.py

.venv/bin/python -m active_sysid.robot.cap_protocol freeze \
  --experiment runs/cap_robot_v1 --revision 0

MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.cap_protocol accept \
  --experiment runs/cap_robot_v1

# After independent Sim acceptance, explicitly request World evaluation.
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot.sim2sim evaluate-world \
  --bundle runs/cap_robot_v1/frozen --out runs/cap_robot_v1/world_final
```

`develop` can be called at most 3 times for revisions 0–2. It stores each revision's policy snapshot and per-seed Sim trace; `freeze` takes the chosen revision and its exact SHA; `accept` uses a disjoint 30-seed holdout in **Sim only**. No World evaluation occurs as part of `develop`, `freeze` or `accept`, and no policy is generated automatically.

A robust CaP method comparison will still need agent-generated samples, camera/geometry failures and independent repeated World tests. A 30-seed Sim acceptance by itself does not establish World success.
