# Stage 3 — visual_randomized

Goal: make the Stage 2 **visual Robot-Space policy** robust to the existing task-Sim physical and illumination randomization. This is not the old Object-Space Stage 3 benchmark.

Inputs: current input package plus **accepted Stage 2** visual policy and summary; same robot-only API and RGB cameras; fixed development seeds and disjoint independent acceptance seeds in `cap_development/pipeline.yaml`.

Preserve a copy of the Stage 2 policy as the unchanged baseline. Improve observation feedback, contact handling, stall detection and recovery only when supported by development evidence. The maximum authored revisions is three. Never read randomized draws from private evaluator provenance in order to specialize by seed. Do not tune on the 30 acceptance seeds.

Output contract: one self-contained robust visual policy, per-revision hashes and development results, frozen policy manifest, 30-seed independent acceptance, and a final comparison against the unchanged Stage 2 policy on the same seed set. After Sim acceptance, the frozen visual policy may be transferred to World by the separate evaluator; do not trigger World during development.
