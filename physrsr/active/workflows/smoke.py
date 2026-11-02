"""One frozen task policy, three CoM configurations, constant mass and inertia."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from ..core.contracts import Task
from .initial_cap import generate_initial_cap
from ..api.process import LoggedAPI, service
from ..core.artifacts import save_json as save
from ..core.paths import PACKAGE_ROOT, PROJECT_ROOT, PROMPTS_ROOT
from ..core.policy import Policy
from ..runtime.simulator import FIXED_MASS_MECHANISM

CASES = (("com_0cm", 0.), ("com_minus4cm", -.04), ("com_plus8cm", .08))


def summarize(case, offset, policy, response):
    row = response["rows"][0]
    outcome = row["feedback"][0]
    samples = outcome["observations"]
    first, last = samples[0]["observation"], samples[-1]["observation"]
    initial_z = first["rod_position"][2]
    return dict(case=case, com_offset_cm=100*offset, policy_sha256=policy.sha256,
        status=outcome["status"], reason=outcome["reason"], error_kind=outcome["error_kind"],
        success_rate=row["success_rate"], execution_s=row["duration_s"],
        initial_position=first["rod_position"], final_position=last["rod_position"],
        final_rise_m=last["rod_position"][2]-initial_z,
        max_rise_at_api_boundaries_m=max(s["observation"]["rod_position"][2]-initial_z for s in samples),
        final_abs_axis_z=abs(last["rod_axis"][2]),
        nominal_target_error_m=abs(.1-(last["rod_position"][2]-initial_z)),
        policy_rpc_calls=len(samples)-2, task_evaluation=outcome["task_evaluation"],
        policy_path="initial_cap/bundle/policy.py")


def load_initial(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "authored":
        raise ValueError("Expected an authored initial CaP manifest")
    return Policy.from_wire(dict(source=(root / "bundle/policy.py").read_text(),
        sha256=manifest["sha256"], api_version=manifest["api_version"]))


def write_report(root, config, results):
    lines = ["# 同一份 policy：固定质量，仅改变质心", "",
        f"唯一 policy：[policy.py](initial_cap/bundle/policy.py)，SHA-256 `{config['policy_sha256']}`。",
        f"总质量固定 {config['mass_kg']} kg，质心处的主惯量固定，摩擦 {config['friction']}，每指上限 2 N，初始化种子 {config['seed']}。",
        "同一源码分别执行三次；没有按配置重新生成或修订程序。", "",
        "| 质心 | 结果 | 原判定原因 | 观测最大升高 | 最终升高 | 最大 abs(axis_z) | 仿真时间 | 编译质量 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['com_offset_cm']:+g} cm | {r['status']} | {r['reason'] or '—'} | "
            f"{100*r['max_rise_at_api_boundaries_m']:.2f} cm | {100*r['final_rise_m']:.2f} cm | "
            f"{r['max_abs_axis_z_at_api_boundaries']:.4f} | {r['execution_s']:.2f} s | {r['compiled_inertial_check']['mass_kg']:.6f} kg |")
    lines += ["", "成功标准保持原样：中心高于桌面 4 cm、抓持且近似水平，并连续保持十个额外控制步。",
        "最大升高和倾斜来自 API 边界采样，不能当作连续控制步极值。每个配置只有一次执行。", "",
        "显式 MJCF inertial 模式固定总质量和质心处惯量，只移动惯性中心 body_ipos；不添加额外配重。",
        "[原始结果](results.json) / [物理参数与源码一致性检查](protocol_checks.json) / [冻结输入](frozen.json)。", ""]
    (root / "report.zh.md").write_text("\n".join(lines))


def run(root, *, initial_cap=None, seed=19001, friction=1.5, mass_kg=.27, base_port=18121):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_root = PROJECT_ROOT
    task = Task(**json.loads((PACKAGE_ROOT / "examples/rod_lift.json").read_text())["task"])
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol="One frozen initial policy; three scored CoM interventions; fixed total mass and inertia",
        cases=[dict(name=name, com_offset=offset) for name, offset in CASES],
        seed=seed, friction=friction, mass_kg=mass_kg, grip_force_per_finger_n=2.,
        mechanism=FIXED_MASS_MECHANISM, task=task.public(),
        initial_cap_origin=str(Path(initial_cap).resolve()) if initial_cap else "generated_once_before_scoring",
        new_code_generations=0 if initial_cap else 1)
    save(root / "config.json", config)
    results = []
    try:
        if initial_cap:
            policy = load_initial(initial_cap)
            shutil.copytree(initial_cap, root / "initial_cap", ignore=shutil.ignore_patterns("simulation.json"))
        else:
            private = dict(physics=dict(com_offset=0., friction=friction, grip_force=2.,
                inertial_mode="fixed_mass_com", mass_kg=mass_kg), seeds=[seed])
            save(root / "authoring_world.json", private)
            with service(["serve-agent", "--out", str(root / "coding_service")], base_port+1,
                    root / "coding_service.log") as coding:
                with service(["serve-sim-target", "--private-config", str(root / "authoring_world.json")],
                        base_port+2, root / "target_service.log") as target:
                    policy = generate_initial_cap(task, target=LoggedAPI(target, root / "authoring_api.jsonl"),
                        agent=LoggedAPI(coding, root / "authoring_api.jsonl"), output=root / "initial_cap")
        config["policy_sha256"] = policy.sha256
        save(root / "config.json", config)
        save(root / "frozen.json", dict(policy=policy.wire(), cases=config["cases"]))
        sources = list(PACKAGE_ROOT.rglob("*.py"))
        sources += list(PROMPTS_ROOT.glob("*.txt"))
        sources += [source_root / p for p in ("physrsr/physics_spec.py", "physrsr/worker.py",
            "physrsr/phase_tracker.py", "physrsr/candidate_programs.py",
            "physrsr/coding/runtime.py", "physrsr/coding/guest.py",
            "physrsr/paths.py", "physrsr/__init__.py", "pyproject.toml",
            "src/cap-x/capx/envs/simulators/robosuite_rod_lift.py", "src/cap-x/capx/integrations/franka/control_privileged.py")]
        save(root / "source_manifest.json", {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
            for p in sources:
                archive.add(p, arcname=str(p.relative_to(source_root)))
        print(f"One frozen policy: {policy.sha256}; mass={mass_kg} kg", flush=True)
        with service(["serve-legacy-simulator"], base_port, root / "evaluation_service.log") as simulator:
            for name, offset in CASES:
                folder = root / name
                folder.mkdir()
                print(f"Evaluating {name}: same policy, mass={mass_kg} kg, CoM={100*offset:+g} cm", flush=True)
                params = dict(task=task.public(), models=[dict(id="evaluation_world", mechanism=FIXED_MASS_MECHANISM,
                    parameters=dict(com_offset=offset, friction=friction, mass_kg=mass_kg), weight=1.)],
                    seeds=[seed], policies=[policy.wire()])
                response = LoggedAPI(simulator, folder / "evaluation_api.jsonl").call("sim.evaluate", params)
                save(folder / "rollout.json", response)
                result = summarize(name, offset, policy, response)
                result["policy_path"] = "initial_cap/bundle/policy.py"
                outcome = response["rows"][0]["feedback"][0]
                result["compiled_inertial_check"] = outcome["compiled_inertial_check"]
                result["max_abs_axis_z_at_api_boundaries"] = max(abs(s["observation"]["rod_axis"][2]) for s in outcome["observations"])
                results.append(result)
                save(root / "results.json", results)
                write_report(root, config, results)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        inertias = [r["compiled_inertial_check"] for r in results]
        checks = dict(unique_policy_hashes=sorted({r["policy_sha256"] for r in results}),
            all_source_hashes_match=len({r["policy_sha256"] for r in results}) == 1,
            compiled_mass_kg=[v["mass_kg"] for v in inertias],
            compiled_com_body_m=[v["com_body_m"] for v in inertias],
            compiled_principal_inertia_kg_m2=[v["principal_inertia_kg_m2"] for v in inertias],
            mass_constant=all(abs(v["mass_kg"]-mass_kg)<1e-10 for v in inertias),
            inertia_constant=all(v["principal_inertia_kg_m2"] == inertias[0]["principal_inertia_kg_m2"]
                                 and v["inertia_quaternion_wxyz"] == inertias[0]["inertia_quaternion_wxyz"] for v in inertias))
        save(root / "protocol_checks.json", checks)
        if not all(checks[k] for k in ("all_source_hashes_match", "mass_constant", "inertia_constant")):
            raise RuntimeError("Controlled CoM comparison failed protocol checks")
        save(root / "completion.json", dict(status="completed", scored_rollouts=3, unique_policies=1,
            new_code_generations=config["new_code_generations"], successes=sum(r["status"] == "success" for r in results),
            policy_sha256=policy.sha256, hardware_connected=False))
        return results
    except Exception as exc:
        save(root / "error.json", dict(error_kind="infrastructure", error=f"{type(exc).__name__}: {exc}",
            completed_scored_rollouts=len(results)))
        raise
