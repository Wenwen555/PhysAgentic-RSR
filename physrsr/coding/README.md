# Codex Astra low — four-arm CaP generation and revision

The redesigned shared-static / feedback / verification experiment lives in
[M2.1-S](../m21s/README.md). This directory retains the original frozen protocol
and shared Codex/RPC backend; historical results are not rewritten.

This is an extension alongside the frozen M2.1-R template experiment. `agent.py`
invokes the installed Codex CLI with **`gpt-6-astra`**, reasoning effort **`low`**,
structured JSON output, ephemeral sessions and existing local Codex login.
It does not substitute another model or manufacture code when inference fails.

Minimal call:

```python
from physrsr.paths import output_path
from physrsr.coding.agent import CodexAgent
from physrsr.coding.prompts import initial_messages
answer = CodexAgent(model="gpt-6-astra", effort="low").generate(
    initial_messages(), output_path("new_codex_program")
)
print(answer["code"])
```

Requires `codex`, an existing Codex login, `bwrap`, and the project `.venv`.
Codex runs without the repository, experiment outputs, user config or skills
mounted. Native shell/apps/multi-agent/code-mode tools are disabled. It returns
Python and rationale, not a fixed template or a candidate ID. See the
[official non-interactive CLI documentation](https://developers.openai.com/codex/noninteractive/).

## Paired protocol

Each case/generation first gets one common no-physics program. From that exact
source, the same Agent makes three information-stage calls: no-physics static
review, identified physics, and Oracle physics. The identified output is copied
as the **identical starting program** for Physics-informed CaP and Sim-CaP.

All four final paths get one additional Codex call:

| Arm | Physical information | Final call |
|---|---|---|
| Vanilla | None | Static review |
| Physics-informed | Shared Probe estimate/support | Static review |
| Sim-CaP | Exactly same estimate and initial code | Actual internal execution logs |
| Oracle Sim-CaP | True CoM/friction, reference only | Oracle-model execution logs |

Each logical path has 3 Codex invocations with the same model/effort; shared
prefixes result in 8 unique call nodes per case/generation, not 12. Realized token
usage is logged, not assumed equal. There is one simulation-guided revision;
initial and final Sim/Oracle programs each receive two Model-seed evaluations.
No World task outcome is sent to the Agent. All programs freeze before any
World task scoring. World scores also evaluate the shared identified initial
program to isolate revision effects from the matched static-review call.

All arms may react to pose observations, regrasp or terminate; Vanilla is not
denied normal feedback. Costs include command counts/retries and elapsed
physical time, not just top-level trial count. Oracle has no stronger controller.

## Generated code isolation

`runtime.py` runs generated Python in a separate PID/user/network/filesystem
namespace. It sees its code, standard Python and system libraries, not the repo,
MuJoCo, credentials or hidden physics. The only actuation is the same four
robot RPC methods listed in `prompts.py`. Workspace/quaternion/speed checks,
96-RPC budget, 5-second guest CPU and 120-second host wall budget apply equally.
Python `random` is seeded deterministically. Isolated Python ignores environment hash seeds; arbitrary hash-order/entropy/clock-driven
policies are not claimed reproducible; generated sources and execution traces
are retained for inspection. Actual hardware tracking/actuation remains untested.

World and Model use different spawned trusted workers. `case_id` and MJCF data
never enter ordinary prompts or guest observations. Code errors are recorded as
`error_kind=program`; infrastructure errors stop the run. Task failures remain
`status=fail`; error counts are reported separately from intention-to-treat SR.

The legacy Robosuite 1000-step termination is disabled **only in the generated
code worker and equally for all arms**, since long code otherwise causes
`executing action in terminated episode`. This does not change the rod success
criterion: centre above table+.04 m, grasped/level for ten hold steps. Nominal
motion target .10 m, actual rise and target error are separately reported.
The first controller run encountering the horizon was aborted, not scored.

## Commands

The existing IK service must be running on port 8116. A fresh controlled run:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m physrsr.coding.experiment \
  --out physics_cap_coding_new --cases 4 --generations 2 --world-seeds 4 --jobs 4
```

Default physics seed=20261105; Probe=7901, Model=7201/7202, World=8101–8104.
The full prespecified sequence has 8 nominal and 2 low-friction controls;
`--cases 4` evaluates only its first four nominal cases, not those negative
controls. Use `--cases 10` for the full sequence. Small case counts cannot
support broad success claims.

Every output root refuses overwrite. `--reuse-generations` reuses a completed
inference only when the entire input conversation, model and reasoning effort
match; changed simulator feedback requires a new call. Reuse metadata and
logical versus newly incurred tokens are reported separately. It never reuses
World outcomes as feedback. Obsolete pilot/aborted outputs and their logs have been deleted; valid comparisons and their copied generation records remain.

Saved artifacts: every prompt, full code/rationale, event/usage log, before/after
Model execution, code diff, frozen source hashes, paired World rollout, error
kind, RPC usage, case-cluster CI and Model false-pass analysis. AST counts are
only descriptive; semantic logic changes must be checked against code/diffs.

## Final information-matched evaluation

The final comparison additionally supplies **both** identified arms with the
same known model assumptions and derived estimated mass/approximate inertia.
Otherwise the simulator would implicitly know the counterweight mass model
while the direct Agent saw only CoM/friction. These values are derived from the
shared estimate, never from World truth. Oracle receives the same representation
computed from its reference true CoM/friction.

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m physrsr.coding.experiment \
  --out physics_cap_coding_information_matched_new --cases 4 --generations 2 --world-seeds 4 --jobs 4 \
  --physics-seed 20261106 --seed-base 9101
```

This uses Probe=8901, Model=8201/8202, World=9101–9104. The archived final run
contains byte-for-byte copies of reused no-information generations, without
outcome-dependent selection. The example above creates a fresh run. Changed physical information or
simulator feedback always requires a new inference. Logical call count and new
versus reused token usage are separated. Development runs are never pooled into
this final estimate. Obsolete development outputs have been removed; final results and their actual
inputs, generated programs and recorded costs remain available.

`mean_close_commands` counts closing commands including extra hold commands; it
is not a count of distinct successful grasps. Legacy phase attribution is a
heuristic for simple programs and may be ambiguous for general retry loops.
Terminal success, raw pose/tilt data and code are the evidence for conclusions.

## Recorded outcome

Information-matched run `outputs/physics_cap_coding_final/`: 4 physical cases,
2 code generations, 4 World seeds, 32 trials/arm. SR: Vanilla .21875,
Physics .81250, Sim .78125, Oracle Sim .81250. Paired Physics−Vanilla
+.59375 [.375,.8125]; Sim−Physics −.03125 [−.09375,0]. Sim changes 6/8 ASTs,
but does not improve World over the shared identified initial code. The final
revision is executed unconditionally; after-model scoring is not a rollback rule.

[Read-only revision and failure audit](../../docs/2026-11-01-coding-revision-audit.md):
only 4 of the 6 AST changes alter control behavior. Sim repairs 2 Model failures
but 0 World failures; Oracle Model repairs 1 and breaks 1. The audit distinguishes
terminal feedback from full failure trajectories and documents the missing
Oracle-initial World counterfactual. No new rollouts or inference were used.

[Full protocol, failure diagnosis and results](../../docs/2026-11-01-coding-agent.md).
`protocol_checks.json` checks exact paired contexts, source hashes, real model
settings and deterministic replay. CLI `error` items saying code-mode host is
disabled are capability notifications, not failed inference; return codes and
valid schemas are checked. No native CLI tool calls occurred in the final run.
