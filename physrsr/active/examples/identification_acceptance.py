"""Benchmark fixture only: public task data and separately owned private Worlds.

The identification agent/workflow imports none of this module. Other tasks
provide their own semantic goals, perception and public evidence in this format.
"""
from pathlib import Path
import json

from ..core.artifacts import save_json


def prepare(parent, preflight):
    parent, preflight = Path(parent), Path(preflight)
    prior = json.loads((parent / "initial_cap/physic_input.json").read_text())
    initial = []
    for e in prior["episodes"]:
        # Poses only, with no benchmark names, case labels, hidden mass/inertia,
        # success-function implementation or task-specific candidate programs.
        samples = []
        for row in e["trajectory"]:
            if row["event"] not in ("initial", "close_gripper", "goto_pose", "terminal"):
                continue
            samples.append(dict(evidence_id=row["evidence_id"], event=row["event"], time_s=row["time_s"],
                object_position_m=row["rod_position_m"], object_axis_base=row["rod_axis"],
                tcp_position_m=row["tcp_position_m"]))
        initial.append(dict(episode_id=e["episode_id"], evidence_id=e["summary"]["evidence_id"],
            external_diagnostic=e["summary"]["diagnostic_task_grade"], samples=samples))
    context = dict(goal=dict(instruction="Pick up the observed rigid object, raise its geometric centre by 0.10 m while retaining a near-horizontal major axis and stable grasp; reduce execution time subject to reliable completion.",
            success=dict(min_height_above_support_m=.04, max_abs_major_axis_vertical=.25, external_hold_steps=10),
            constraints=dict(unchanged_gripper_force_limit=True, nominal_rise_m=.10)),
        capabilities=json.loads((preflight / "capabilities.json").read_text()),
        current_observation=json.loads((preflight / "observation.json").read_text()),
        initial_evidence=initial,
        evidence_contract=dict(episodes_are_distinct_instances=True,
            samples_are_api_boundaries=True, episode_id_has_no_physical_meaning=True,
            hidden_object_parameters_omitted=True, source_benchmark_identifier_omitted=True))
    cases = []
    for i, offset in enumerate((0., -.04, .08), 1):
        cases.append(dict(episode_id=f"episode_{i:03d}", world=dict(physics=dict(
            com_offset=offset, friction=1.5, grip_force=2., mass_kg=.27, inertial_mode="fixed_mass_com"), seeds=[19001])))
    save_json(preflight / "public_context.json", context)
    save_json(preflight / "evaluator_only_cases.json", cases)
    return context, cases
