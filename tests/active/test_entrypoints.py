"""Package migration and read-only service startup, without model/task execution."""
import importlib
from pathlib import Path
import pkgutil
import socket
import pytest

import physrsr.active as active
from physrsr.active.__main__ import main
from physrsr.active.api.process import service
from physrsr.active.core.contracts import IDENTIFICATION_API_VERSION
from physrsr.active.core.paths import PACKAGE_ROOT, PROJECT_ROOT, PROMPTS_ROOT


def test_all_grouped_modules_import_and_package_resources_resolve():
    modules = [importlib.import_module(item.name) for item in
        pkgutil.walk_packages(active.__path__, prefix=active.__name__+".")]
    assert {p.name for p in (PACKAGE_ROOT/"workflows").glob("*.py")}=={"__init__.py","loop.py"}
    assert not any(module.__name__.endswith((".workflows.probe_authoring", ".agents.identification")) for module in modules)
    retired={"physrsr.active.workflows.legacy_loop","physrsr.active.workflows.batch",
        "physrsr.active.workflows.scoped_loop","physrsr.active.legacy"}
    assert not retired.intersection(module.__name__ for module in modules)
    assert {module.__name__ for module in modules if module.__name__.endswith("loop")}=={
        "physrsr.active.workflows.loop"}
    assert PACKAGE_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROMPTS_ROOT / "identified_policy.txt").is_file()
    assert (PROMPTS_ROOT / "physical_identification.txt").is_file()
    assert not (PROMPTS_ROOT / "physic_probe.txt").exists()




@pytest.mark.parametrize("working_directory", ["workspace", "dependency", "unrelated"])
def test_http_service_starts_and_describes_without_loading_a_world(tmp_path, monkeypatch, working_directory):
    directory = {"workspace": PROJECT_ROOT, "dependency": PROJECT_ROOT / "src/cap-x", "unrelated": tmp_path}[working_directory]
    monkeypatch.chdir(directory)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with service(["serve-simulator"], port, tmp_path / "service.log") as endpoint:
        description = endpoint.call("sim.describe", {})
    assert description["api_version"] == IDENTIFICATION_API_VERSION
    assert "rod_fixed_mass_com_v1" in description["mechanisms"]
    assert "Loading controller" not in (tmp_path / "service.log").read_text()


def test_world_controller_resources_do_not_depend_on_cwd(tmp_path, monkeypatch):
    import physrsr.worker as worker
    from physrsr.physics_spec import PhysicsSpec
    calls = []
    class Environment:
        def __init__(self, **kwargs):calls.append(kwargs)
    monkeypatch.setattr(worker, "FrankaRobosuiteRodLiftLowLevel", Environment)
    monkeypatch.setattr(worker, "FrankaControlPrivilegedApi", lambda env: object())
    monkeypatch.chdir(tmp_path)
    worker.RolloutWorker(PhysicsSpec())._ensure_env()
    assert Path(calls[0]["controller_cfg"]).is_absolute()
    assert Path(calls[0]["controller_cfg"]).is_file()


def test_run_has_one_predictive_loop_and_requires_all_services(tmp_path, monkeypatch):
    import json
    import physrsr.paths as paths
    from physrsr.active.workflows import loop
    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path)
    config = tmp_path / "config.json"
    payload = dict(task=dict(id="scene", instruction="Manipulate all three rods", success={}, constraints={}),
        endpoints={key: "http://127.0.0.1:1" for key in ("agent", "physic", "simulator", "target")})
    config.write_text(json.dumps(payload))
    calls = []
    class Loop:
        def __init__(self, task, cfg, models, **kwargs): calls.append(kwargs)
        def run(self): return dict(status="task_sufficient")
    monkeypatch.setattr(loop, "ActiveLoop", Loop)
    assert main(["run", "--config", str(config), "--out", "run"]) == 0
    assert set(calls[0]) == {"output", "agent", "physic", "simulator", "target"}
    payload["endpoints"].pop("simulator")
    config.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", str(config), "--out", "missing_sim"])
    assert exc.value.code == 2 and len(calls) == 1


@pytest.mark.parametrize("flags", [["--cases", "missing-cases.json"], ["--resume"]])
def test_run_rejects_legacy_switches_before_reading_files_or_creating_outputs(tmp_path, monkeypatch, capsys, flags):
    import physrsr.paths as paths
    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", "missing-config.json", "--out", "new", *flags])
    assert exc.value.code == 2
    error=capsys.readouterr().err
    assert ("no longer supported" if "--cases" in flags else "resume_probe") in error
    assert not (tmp_path / "new").exists()


def test_removed_batch_command_is_not_available(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["legacy-batch"])
    assert exc.value.code==2 and "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize('command',['author-initial','identification','probe-world','revise-cap','compare-caps','smoke'])
def test_removed_experiment_commands_are_not_available(command,capsys):
    with pytest.raises(SystemExit) as exc: main([command])
    assert exc.value.code==2 and 'invalid choice' in capsys.readouterr().err
