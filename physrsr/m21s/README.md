# M2.1-S: static reasoning plus simulation evidence and revision gating

This version leaves all previous experiments unchanged. The shared reviewed
program is the baseline. A, B and D use the same Agent, physical information,
conversation prefix, static requirements and one final revision call. C reuses
B's exact candidate and deterministic check. All arms use the existing isolated
`run(robot)` RPC controller with unchanged task grading and 96-call limit.

The common review itself is statically gated: if it introduces a deterministic
counterexample and the original initial code passes, all arms revert to that
same initial code with an explicit rejection notice in their common context.
If neither passes, the run aborts. No extra inference is granted to replace a
bad review. Rejected review code is retained alongside `reviewed_baseline.py`.

| Arm | Final revision evidence | Deployment decision |
|---|---|---|
| A | Static reasoning only | Deterministic static check |
| B | Static + legacy last-four phase feedback | Same static check, no physical gate |
| C | Exactly B's candidate | Same static check + paired Model validation |
| D | Static + structured trajectory evidence | Same gate and seeds as C |

All feedback geometry is robot-base. B/C retain the original tail representation
but its frame bug is corrected universally; C/D isolate the richer representation
rather than also changing the control API or revision instructions. The prompts
have the same static review instructions; evidence length/token cost differs and
is reported. There is one feedback-guided revision, no iterative optimization on
validation or World outcomes.

`static.py` combines AST syntax/API/import checks with three deterministic,
nonphysical kinematic contract scenarios (positive/negative tilt and hold sag).
These expose concrete counterexamples, not a proof of arbitrary Python safety.
All generated programs run inside the existing OS sandbox. Hardware uses the
same robot API; simulator contact diagnostics are developer evidence, not added
runtime capabilities for policies.

`trace.py` records actual API boundaries and every underlying Robosuite control
step, including steps inside blocking motion. `feedback.py` distinguishes observed
tilt/loss/drop/release events from unverified causes and retains continuous event
windows. Relative displacement contains rotation, so it is not called pure slip.
Full trajectories and legacy phase records (explicit raw-frame metadata) remain
in rollout JSONL. Policies/agents never receive case IDs or hidden parameters.

The gate uses four paired validation seeds disjoint from the two development
seeds and World seeds. Any paired success regression or new severe event rejects
replacement. A repair needs minimum two rescues and a one-sided sign p<=.10
(with four seeds this requires four wins). All-success efficiency acceptance
requires >=10% improvement on every seed, no material target-error regression,
and the same sign criterion. Alternatively every seed must improve nominal
target error by >=5mm with <=5% time penalty. Equality or insufficient evidence
keeps the baseline. These are decision heuristics, not reliability guarantees.
The thresholds are fixed before World evaluation and saved in config.

## Reproduction

Requires the existing `.venv`, Codex Astra low login, bubblewrap and IK server on
127.0.0.1:8116. Output roots refuse overwrite.

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m pytest \
  tests/test_physics_cap_m21s.py tests/test_physics_cap_coding.py \
  tests/test_physics_cap_metrics.py tests/test_robosuite_rod_lift.py \
  tests/test_robosuite_physics_lift.py -q

MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m physrsr.m21s.regressions \
  --out physics_cap_m21s_regressions

MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m physrsr.m21s.experiment \
  --cohort regression --out physics_cap_m21s_regression_new \
  --cases 4 --generations 2 --world-seeds 4 --jobs 4 --seed-base 12001

MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m physrsr.m21s.experiment \
  --cohort independent --out physics_cap_m21s_independent \
  --cases 4 --generations 2 --world-seeds 4 --jobs 4 \
  --physics-seed 20261107 --seed-base 16001
```

The regression command creates a fresh run. The archived controlled-v3 result
uses its own source snapshot and contains copies of the reused programs and
trajectories. Obsolete interrupted runs and their logs have been deleted.
Do not mix feedback versions when reusing generations. Use each valid run's
source archive to reproduce its exact recorded protocol.

The regression cohort reuses ALL eight historical initial identified programs,
adds the new shared review and three final calls (A/B/D). It is a known-case
protocol check. The independent cohort samples four continuous physical cases,
uses the unchanged fixed probes/identifier, and makes fresh initial/shared/final
calls. Physics identification itself is not retuned. Cohorts and generation
counts are never pooled as independent physical cases. All sources freeze before
World scoring; no stopping or seed expansion based on an effect's significance.

The report includes paired case-cluster bootstrap intervals, repairs/breaks,
gate decisions, Model-to-World deployed repair transfer, false-pass counts,
new inference usage and actual rollout costs. Unaccepted raw D candidates have
no World counterfactual; that transfer is explicitly unknown. Generic diagnostic
accuracy and semantic code-change rates need independent labeling, not AST counts.

## Frozen measured results

The known-configuration check completed 160 World trials: A/B/C/D all 30/32.
D accepted one efficiency revision. Its World mean time difference against A
was -4.280s, case-cluster CI [-17.527,+4.688]; it is not a broad efficiency claim.

Independent continuous configurations, 2 generations and 4 World seeds:

| Arm | World successes | Case-cluster 95% CI | Mean task time (s) |
|---|---|---|---|
| A | 15/32 | [.125,.84375] | 41.658 |
| B | 15/32 | [.125,.84375] | 37.530 |
| C | 15/32 | [.125,.84375] | 47.191 |
| D | 15/32 | [.125,.84375] | 47.191 |

No primary arm repaired or broke a World outcome. C/D accepted 0/8, rolled back
2/8 for paired Model regressions, and kept 6/8 for insufficient evidence. All
successful trials have the same mean time (28.52s); overall timing differences
come from failing trials. Same-source/seed complete trajectories replay exactly
for 64 pairs in the independent run, and 60 in the known-case run.

An evaluator-only post-hoc replay of ALL frozen D proposals on the SAME World
states gives 14/32: 0 repairs and 1 broken success. Gate retained the successful
baseline. This does not replace the primary arm or tune its threshold. The known
Oracle bug additionally breaks 3/4 World successes into 0/4; the new shared
static review repairs the specified old ordinary program from 0/4 to 4/4 on
paired known-case states. Known regressions are not generalization evidence.

The independent core uses 40 new LLM calls, 144 Model rollouts, 160 World task
trials and 8 fixed probes. Full-session development/diagnostic costs and tokens
are separated in `outputs/physics_cap_m21s_logs/session_costs.json`. Final tests:
**40 passed**, two pre-existing dependency deprecation warnings.

Recompute statistics and verify artifacts without new inference or physics:

```bash
src/cap-x/.venv/bin/python -m physrsr.m21s.report physics_cap_m21s_independent
src/cap-x/.venv/bin/python outputs/physics_cap_m21s_logs/verify_run.py outputs/physics_cap_m21s_independent
```

[Full protocol, results, failure mechanisms and freeze decision](../../docs/2026-11-01-m21s.md).
The executable protocol is implemented, but no additional independent World
success benefit over static reasoning is supported. Keep Rod Lift as a harness
regression/identification benchmark; do not add seeds or revisions to seek a
positive result. Active probing and new manipulation tasks were not implemented.
