# Original Ground-Truth World (evaluation only)

`unidoor_native/scene.xml` and its complete `resources/` tree are the original double-door World export. The 111 source files retain their original hashes across directory moves. No source model edit or hidden-parameter import is needed for Robot-Space Grounding.

Code: `src/active_sysid/robot/reference_world_backend.py` and `reference_world_evaluator.py`; explicit entrypoints: `python -m active_sysid.robot.sim2sim probe-world` for sensor-only interaction evidence, `evaluate-world` for frozen visual CaP. The old privileged `reference_oracle_policy.py` remains a diagnostic capability reference, not a CaP-development route.
