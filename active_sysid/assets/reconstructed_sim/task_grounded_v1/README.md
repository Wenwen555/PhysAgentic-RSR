# Task-grounded Robot-Space simulation variants

- `robot_m0`: contact-feasible initial task prior, assumes direct mobility.
- `robot_m1`: public-probe-supported task precondition, rotation before pulling.
- `robot_m2`: M1 plus a probe-fitted effective pull-radius response.
- `robot_contact_demonstration.json`: robot q/TCP/gripper/contact-only initialization, **not** hidden World latch truth.
- `grounding_provenance.json`: inputs and task-model updates.

Rebuild via `scripts/run_robot_grounding.py` using recorded public probes or new sensor-only probes. `robot_m2` is not a detailed recreation of the native World door-lock mechanism.
