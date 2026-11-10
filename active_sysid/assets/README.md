# Assets organized by experimental role

- `reference_world/unidoor_native/`: **Ground-Truth World**. Only the limited probe endpoint and final evaluator load it; no model update should inspect its hidden dynamics or native latch.
- `reconstructed_sim/visual_stage3_v1/`: previous Stage 3 vision-reconstructed geometry and shared robot. Kept as provenance, **not** treated as ground-truth geometry.
- `reconstructed_sim/task_grounded_v1/`: Robot-Space contact-feasible `robot_m0`, task-condition `robot_m1`, and response-calibrated `robot_m2`; model sources and public-probe evidence explained in `docs/ROBOT_SPACE_M0_M2_IMPLEMENTATION.md`.
- `fixtures/panda_single_door/`: isolated single-door Panda actuator smoke fixture. **Not** reference World or the Sim2Sim environment.

`assets/reference_world/unidoor_native/scene.xml` contains 111 unmodified original asset files (excluding locally authored README). The other models are independent derived models with separate provenance; never exchange these roles by merely renaming output labels.
