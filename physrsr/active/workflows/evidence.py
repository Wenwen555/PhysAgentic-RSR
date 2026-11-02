"""Organize initial-CaP sensor evidence without exposing evaluator parameters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core.contracts import API_VERSION, ROBOT_CONTRACT
from ..core.policy import Policy, execute
from ..runtime.robot import public_observation
from ..core.artifacts import save_json


def replay_commands(policy, samples, runner=execute):
    """Reconstruct calls using recorded replies; no robot/simulator is reachable.

    These are replay-derived arguments, not original command telemetry. Method
    order must exactly match every archived API boundary or evidence prep fails.
    """
    if len(samples) < 2 or samples[0]["event"] != "initial" or samples[-1]["event"] != "terminal":
        raise ValueError("Expected initial and terminal samples")
    commands, cursor = [], 1

    def dispatch(method, params):
        nonlocal cursor
        if cursor >= len(samples)-1 or samples[cursor]["event"] != method:
            raise ValueError(f"Frozen policy replay diverged at sample {cursor}: {method}")
        observation = public_observation(samples[cursor]["observation"])
        commands.append(dict(call_index=len(commands)+1, post_sample_index=cursor,
            method=method, params=params,
            time_before_s=public_observation(samples[cursor-1]["observation"])["time_s"],
            time_after_s=observation["time_s"], arguments_source="deterministic_sensor_log_replay"))
        cursor += 1
        return observation if method == "observe" else None

    runner(policy, dispatch)
    if cursor != len(samples)-1:
        raise ValueError("Frozen policy replay did not consume all policy API samples")
    return commands


def episode_evidence(episode_id, policy, outcome, runner=execute):
    samples = [dict(event=s["event"], observation=public_observation(s["observation"]))
               for s in outcome["observations"]]
    commands = replay_commands(policy, samples, runner=runner)
    command_by_index = {c["post_sample_index"]: c for c in commands}
    start = samples[0]["observation"]
    trajectory, compact = [], []
    zero_time_observe = 0
    for index, sample in enumerate(samples):
        obs = sample["observation"]
        eid = f"{episode_id}:sample_{index:03d}"
        derived = dict(rise_m=obs["rod_position"][2]-start["rod_position"][2],
            abs_axis_z=abs(obs["rod_axis"][2]), signed_axis_z=obs["rod_axis"][2],
            rod_minus_tcp_m=[obs["rod_position"][i]-obs["tcp_position"][i] for i in range(3)])
        row = dict(evidence_id=eid, sample_index=index, event=sample["event"],
            observation=obs, derived=derived, command=command_by_index.get(index))
        trajectory.append(row)
        unchanged = index and obs == samples[index-1]["observation"]
        if sample["event"] == "observe" and index and obs["time_s"] == samples[index-1]["observation"]["time_s"]:
            zero_time_observe += 1
        # Duplicate observe calls stay in the complete archive. The author gets
        # every action boundary, terminal sample and changed observation.
        if sample["event"] == "observe" and unchanged:
            continue
        compact.append(dict(evidence_id=eid, event=sample["event"],
            time_s=round(obs["time_s"]-start["time_s"], 6),
            rod_position_m=[round(v, 6) for v in obs["rod_position"]],
            rod_axis=[round(v, 6) for v in obs["rod_axis"]],
            tcp_position_m=[round(v, 6) for v in obs["tcp_position"]],
            rise_m=round(derived["rise_m"], 6),
            command=(dict(method=row["command"]["method"], params=row["command"]["params"])
                     if row["command"] else None)))
    first_tilt = next((r for r in trajectory if r["derived"]["rise_m"] > .005 and r["derived"]["abs_axis_z"] > .25), None)
    peak_rise = max(trajectory, key=lambda r: r["derived"]["rise_m"])
    peak_tilt = max(trajectory, key=lambda r: r["derived"]["abs_axis_z"])
    terminal = trajectory[-1]
    summary = dict(evidence_id=f"{episode_id}:summary", episode_id=episode_id,
        diagnostic_task_grade=dict(status=outcome["status"], terminal_reason=outcome["reason"],
            source="external_simulator_task_check; not a runtime sensor"),
        max_observed_rise_m=peak_rise["derived"]["rise_m"], max_rise_evidence=peak_rise["evidence_id"],
        max_abs_axis_z=peak_tilt["derived"]["abs_axis_z"],
        signed_axis_z_at_max_tilt=peak_tilt["derived"]["signed_axis_z"], max_tilt_evidence=peak_tilt["evidence_id"],
        first_excessive_tilt=None if first_tilt is None else first_tilt["evidence_id"],
        final_rise_m=terminal["derived"]["rise_m"], final_evidence=terminal["evidence_id"],
        observed_simulation_duration_s=terminal["observation"]["time_s"]-start["time_s"],
        rpc_count=len(commands), zero_time_observe_calls=zero_time_observe,
        samples_source="Archived public API boundary poses; not continuous control-step telemetry",
        command_arguments_source="Offline deterministic replay; not original command telemetry",
        relative_motion_caveat="Rod-minus-TCP displacement also changes under rotation; it is not pure slip")
    return dict(episode_id=episode_id, summary=summary, trajectory=trajectory,
                commands=commands, author_trajectory=compact)


def prepare_initial_evidence(experiment, runner=execute):
    experiment = Path(experiment).resolve()
    config = json.loads((experiment / "config.json").read_text())
    initial = experiment / "initial_cap"
    manifest = json.loads((initial / "manifest.json").read_text())
    policy = Policy.from_wire(dict(source=(initial / "bundle/policy.py").read_text(),
        sha256=manifest["sha256"], api_version=manifest["api_version"]))
    if config["policy_sha256"] != policy.sha256:
        raise ValueError("Experiment policy hash mismatch")
    task = {key: config["task"][key] for key in
            ("id", "instruction", "success", "constraints", "api_version", "robot_api")}
    task["robot_api"] = ROBOT_CONTRACT
    if task["api_version"] != API_VERSION:
        raise ValueError("Unsupported evidence API")
    episodes, private_index = [], {}
    for index, case in enumerate(config["cases"], 1):
        name = case["name"]
        if Path(name).name != name:
            raise ValueError("Invalid archived episode directory")
        response = json.loads((experiment / name / "rollout.json").read_text())
        row = response["rows"][0]
        if row["policy_sha256"] != policy.sha256 or len(row["feedback"]) != 1:
            raise ValueError("Evidence must use one common policy and one rollout per episode")
        episode_id = f"episode_{index:03d}"
        episodes.append(episode_evidence(episode_id, policy, row["feedback"][0], runner=runner))
        private_index[episode_id] = dict(source_directory=name,
            source_sha256=hashlib.sha256((experiment / name / "rollout.json").read_bytes()).hexdigest())
    organized = initial / "organized_trajectories"
    if organized.exists():
        cached = json.loads((initial / "physic_input.json").read_text())
        recorded_index = json.loads((experiment / "evaluator_only_episode_index.json").read_text())
        if recorded_index != private_index or cached["initial_policy"]["sha256"] != policy.sha256:
            raise ValueError("Refusing to reuse changed initial evidence")
        # The archival observations are unchanged. API documentation can gain
        # additive public fields without rewriting the original authoring input.
        cached["task"] = task
        save_json(initial / "task.json", task)
        save_json(initial / "physic_input.json", cached)
        return cached
    organized.mkdir(exist_ok=False)
    for episode in episodes:
        save_json(organized / f"{episode['episode_id']}.json", episode)
    save_json(initial / "task.json", task)
    save_json(initial / "trajectory_summary.json", [e["summary"] for e in episodes])
    save_json(experiment / "evaluator_only_episode_index.json", private_index)
    public = dict(task=task, initial_policy=dict(evidence_id="initial_policy", **policy.wire()),
        evidence_contract=dict(coordinate_frame="robot_base", position_unit="m", time_unit="s",
            episodes_are_distinct_physical_instances=True, do_not_pool_into_one_parameter_estimate=True,
            total_mass_and_com_frame_inertia_held_constant=True,
            hidden_parameter_values_and_case_labels_omitted=True,
            terminal_sample_is_after_external_task_hold=True,
            commands_reconstructed_by_offline_replay=True, no_new_physical_execution=True),
        episodes=[dict(episode_id=e["episode_id"], summary=e["summary"], trajectory=e["author_trajectory"]) for e in episodes])
    save_json(initial / "physic_input.json", public)
    return public
