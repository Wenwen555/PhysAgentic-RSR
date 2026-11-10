# Stage 1 — privileged_exact

Goal: establish a working Robot-Space CaP in the **reconstructed Sim** using its internal object state. This is a privileged Sim development upper bound, **not** a World-truth policy.

Inputs: `input/manifest.json`, `input/robot_api.md`, `input/task.md`, `input/grounding_summary.md`, and `input/stage_specific.md`. `api.observe()['privileged']` is available here only, including Sim door/handle angles, handle grasp point/pivot/axis, and door pivot/axis.

Develop a complete reach → grasp → rotate → pull → verify sequence. The policy must use physical robot motion and contact; no direct object-joint control. Prefer observations over hard-coded World poses. Write the output file named by the manifest.

Output contract: policy source, Sim run evidence (only after human runs it), a summary of action phases, policy SHA256, failure categories, and Stage 2 transfer notes describing which privileged fields must be replaced by vision or robot feedback.
