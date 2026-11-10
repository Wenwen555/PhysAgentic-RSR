# Robot-Space Code-as-Policy — shared development contract

You are the **Coding Agent**, powered by Codex `gpt-6-astra`. Build a self-contained Python `def run(api):` policy that physically opens the lever door with the robot. You write policy code, not MuJoCo patches. The simulator is the **task-grounded robot_m2** reconstruction, not the original World. Task success: physically open the door to at least **0.30 rad**, with robot-only actuation and real contacts.

## Available robot API

- `api.observe(rgb=False, camera='overview')`: robot `time_s`, 7-joint `q/dq`, `joint_torque_nm`, `ee_pos_m`, `ee_quat_wxyz`, `gripper_width_m`, `wrist_wrench_ee` (6-vector), `finger_contact_normal_n` (2-vector); `rgb=True` adds uint8 `rgb` (480×640×3). Cameras: `overview`, `robot0_eye_in_hand`.
- `api.phase(name)`: labels execution phase, does not move anything.
- `api.move_joints(positions_rad, duration_s=2.0)`; `api.move_ee(position_m, quaternion_wxyz, duration_s=1.0)`; `api.move_arc(center_m, axis_world, angle_rad, duration_s=2.0)`; `api.set_gripper(width_m, duration_s=0.6)`; `api.hold(duration_s)`.
- Units: meters, radians, seconds, wxyz quaternion. Actions return robot-only observation.
- Source must be a **single self-contained file** (NumPy allowed); runner mounts only policy and numeric library, not repository helper modules. No direct object actuator or private evaluator state.

## Task and evidence

The M0–M2 Grounding chain established an actionable dependency: initial direct pulling did not transfer to World; a rotation-then-pull reference program did. This is a **task-level prior**, not a hidden World mechanism description. Public robot contact demonstration and Sim model are available; do not read `assets/reference_world`, `evaluator_private`, source-World mechanism or old oracle scripts to tune your policy. Do not copy the fixed reference program as a submitted CaP: it is a diagnostic and includes a demonstrated pose. Your code should use the stage's allowed observations to drive and verify actions.

## Output

Write only the assigned `output_policy` file. Include `run(api)` and any local helper functions in the same file. Record your design choices and unresolved assumptions in the stage's output summary after evaluation. Do not alter frozen Sim/World, stage configs, scoring gates or robot control code. The human orchestrator, not your generated policy, executes stage evaluation and freezes source hashes.
