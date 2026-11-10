# Repository layout and roles (Robot-Space Sim2Sim)

This page is the canonical file map after the November 2026 cleanup. **Names now describe the source of truth and experiment role, not just the model's appearance.**

```text
active_sysid/
├── assets/
│   ├── reference_world/unidoor_native/     # private World truth; 111 pinned source files
│   ├── reconstructed_sim/
│   │   ├── visual_stage3_v1/                # older vision reconstruction (not World)
│   │   └── task_grounded_v1/                # mainline robot_m0, robot_m1, robot_m2
│   └── fixtures/panda_single_door/         # separate single-door actuator smoke model
├── configs/
│   ├── robot/simulators/
│   │   ├── robot_grounding_m0_m2.yaml      # mainline
│   │   └── legacy_visual_stage3.yaml       # visual-only historical baseline
│   ├── robot/world_bindings/unidoor_native.yaml
│   ├── robot/cap_protocol.yaml            # 3 revisions, 20 development/30 holdout seeds
│   ├── robot/stages/                      # privileged_exact, visual_exact, visual_randomized
│   └── worlds/                            # legacy object-space experiment configuration
├── src/active_sysid/
│   ├── robot/
│   │   ├── controller.py, adapter.py, runtime.py
│   │   ├── fixture_*                       # Panda single-door debug-only
│   │   ├── reconstructed_sim_backend.py   # Sim-only development
│   │   ├── reference_world_*               # World-only execution + private evaluator
│   │   ├── boundary_runtime.py, policy_rpc.py, policy_worker.py
│   │   ├── stages.py, cap_protocol.py, sim2sim.py
│   │   └── stage_freeze.py                # SHA256 manifest verification
│   ├── grounding/task_interaction.py      # public probe → task-conditioned Sim update
│   └── backends/, identification/, ...    # existing legacy object-space experiments
├── policies/robot_space/
│   ├── reference_initial_direct_pull.py
│   └── reference_grounded_turn_pull.py   # fixed programs, NOT Coding-Agent CaPs
├── worlds/legacy_object_space/           # earlier object-space oracle/reconstruction
├── scripts/run_robot_grounding.py
├── docs/ROBOT_SPACE_M0_M2_IMPLEMENTATION.md
├── docs/SIM_WORLD_BOUNDARY.md
├── docs/CAP_PIPELINE_REUSE.md
└── runs/                                 # local evidence; gitignored, not versioned
```

## Important naming boundaries

- `reference_world` is **only** the native evaluation environment. The legacy object-space `worlds/legacy_object_space/oracle` is **not** this reference World.
- `robot_m0` / `robot_m1` / `robot_m2` are Robot-Space models. Legacy M0/M1/M2 data under `worlds/legacy_object_space` and older `runs/` are not silently swapped in.
- `fixture_*` names mean **debug fixtures**. Their success is not World transfer evidence.
- `reference_*` policy names mean manually authored, fixed baseline demonstrations; they are **not** agent-generated visual CaPs.
- Previous object-space Stage 1–3 CaP delivery archives remain local under `runs/policy-develop/` (gitignored). See [CAP_PIPELINE_REUSE.md](CAP_PIPELINE_REUSE.md) for what was reused.

## Migration

The active files were moved, and executable imports and config paths were updated. Historical files inside `docs/archive/` retain their original snapshot names and paths and are clearly archival. The frozen current execution manifest is `configs/robot/frozen_sim2sim.json`; historical `docs/archive/world_only_stage_v1/frozen_r1.json` is unchanged.

Existing local run evidence under `runs/` retains its original directory names, SHA256 and meanings. It is not reclassified or rewritten just to match the cleaned source layout.
