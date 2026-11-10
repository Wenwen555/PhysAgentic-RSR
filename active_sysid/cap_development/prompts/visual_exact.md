# Stage 2 — visual_exact

Goal: replace **all** Stage 1 privileged object observations with RGB and robot-only feedback while preserving the manipulation strategy. Stage 2 uses the same `robot_m2` Sim, no randomization.

Inputs: current stage input package plus the **accepted** Stage 1 policy, Stage 1 summary, and permitted Sim observations. Read the Stage 1 outputs for strategy intent, not for hidden object pose or evaluator values. Do not import Stage 1 policy as a runtime dependency; emit one self-contained policy file.

`api.observe()` contains no `privileged` key. Request `rgb=True` on the allowed cameras, infer task progress from RGB and robot contact/wrench feedback, and verify at different times. A simple image-read count is not proof of visual reasoning; explicitly document what visible cue you use. Do not copy an exact reference-World pregrasp target as a claimed visual solution.

Output contract: self-contained visual policy, visual estimation/feedback notes, Sim exact results (after evaluation), stage acceptance record, policy SHA256, and a Stage 3 handoff describing expected sensitivity to camera/physics perturbations.
