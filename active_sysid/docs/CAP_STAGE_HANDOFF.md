# Codex Astra CaP Stage 1–3: input/output handoff

**Status: contexts prepared; no Coding Agent run, no CaP generation, no Sim evaluation, no World evaluation.** This is the development contract for the next DevSpace session. The user requested `gpt-6-astra`; Codex CLI is installed and logged in with ChatGPT, but model invocation has deliberately not been attempted.

## Context location

`active_sysid/runs/cap_context_preparation_v1/` contains generated stage inputs and blank status records. This path is local and gitignored (experiment runtime context). Version-controlled templates and preparation code are in `cap_development/` and `scripts/prepare_cap_context.py`.

```text
runs/cap_context_preparation_v1/
├── pipeline.json                         # prepared_not_executed
├── stage1_privileged_exact/
│   ├── input/{manifest.json,task.md,robot_api.md,grounding_summary.md,stage_specific.md}
│   └── output/status.json                # not_started
├── stage2_visual_exact/
│   ├── input/{...previous_stage.md}
│   └── output/status.json                # not_started
└── stage3_visual_randomized/
    ├── input/{...previous_stage.md}
    └── output/status.json                # not_started
```

## Stage I/O contract

| Stage | Inputs | CaP output | Acceptance and handoff |
|---|---|---|---|
| 1 `privileged_exact` | task, Robot API, `robot_m2` model identity, Sim-only privileged object observations | `policies/robot_space/cap_stage1_privileged.py` | Exact-Sim run, source hash, failure notes, Stage 2 privilege-removal handoff |
| 2 `visual_exact` | Stage 1 accepted source and evidence, same Sim, RGB + robot sensors | `policies/robot_space/cap_stage2_visual.py` | Exact-Sim run, visual estimator description, source hash, Stage 3 perturbation handoff |
| 3 `visual_randomized` | Stage 2 accepted visual source, fixed 20 dev seeds, 30 holdout seeds, unchanged Sim2Sim boundary | `policies/robot_space/cap_stage3_robust.py` | Revision logs, frozen source, independent Sim acceptance, final frozen World transfer **only after acceptance** |

Each stage's `input/manifest.json` records its target policy path, observation permissions, predecessor, model, development/holdout seeds, input SHA256 and expected outputs. Each `output/status.json` starts `not_started`; the next session must mark a stage `accepted` **only after its independent evaluation passes**, never when code merely exists. Later stages refuse launch if the previous stage status is not `accepted`.

The output contract anticipates `policy.py`, `agent_trace.jsonl`, `development_summary.json`, `acceptance_summary.json`, `handoff.md`. They are **not prepopulated**; only a blank `status.json` is prepared. The actual robot stage evaluator produces `metrics.json`, `actions.json`, `trajectory.npz`, RGB frames, per-seed summaries and source SHA256 under `runs/`.

## Source reused from agentic_rsr and previous CaP deliveries

- `agentic_rsr/src/real2sim/stage2/agent_runner.py`: bounded Codex reasoning and explicit agent prompt, handoff records.
- `agentic_rsr/src/real2sim/stage3/agent_runner.py`: default `gpt-6-astra`, structured result, per-command trace, independent verification.
- Previous local `runs/policy-develop/stage1_delivery`, `stage2_delivery`, `stage3_visual_randomized_delivery`: revision budgets, candidate freeze, 20 dev + 30 held-out seeds.
- Current Robot-Space `robot/cap_protocol.py`, `robot/stages.py`, `robot/sim2sim.py`: the real evaluation and frozen World transfer interfaces.

Do not copy previous `api.step(handle,door)` policy implementations: those are object-space direct-actuator programs and cannot operate the current robot.

## Next-session commands

```bash
cd /home/wen/experiment/PhyAgent-RSR/active_sysid
.venv/bin/python -m active_sysid.robot.stages check

# Print (DO NOT RUN) the Stage 1 Codex command:
.venv/bin/python scripts/launch_cap_agent.py \
  --context runs/cap_context_preparation_v1 --stage privileged_exact

# When ready to actually start the coding agent:
.venv/bin/python scripts/launch_cap_agent.py \
  --context runs/cap_context_preparation_v1 --stage privileged_exact --execute
```

Codex is configured as `gpt-6-astra`, `high` reasoning, `workspace-write`. Each stage is a **new** Codex coding-agent run; the previous stage's *accepted* source and evaluation report are explicit handoff inputs. The preparation script does not call Codex or MuJoCo. Do not assume the model is accessible until the first explicit run succeeds.

After writing Stage 1 policy, evaluate it with `python -m active_sysid.robot.stages run --stage privileged_exact --sim-variant robot_m2 --policy ... --out runs/...`; complete the status and handoff before Stage 2. Stage 2 similarly uses `visual_exact`. Stage 3 uses `python -m active_sysid.robot.cap_protocol develop/freeze/accept` with its predeclared seed budgets, then explicitly `sim2sim evaluate-world` on the frozen visual bundle.

**Scientific boundary:** Stage 1 may use Sim-only privileged object geometry, not World truth. Stages 2/3 may use only RGB, q/dq, TCP, wrist wrench, gripper and finger contacts. `robot_m2` contains a task-conditioned contact model; its success does not prove RGB-based CaP generation. No stage may read World evaluator-private telemetry to optimize code.
