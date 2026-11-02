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
    assert any(module.__name__.endswith(".workflows.smoke") for module in modules)
    assert any(module.__name__.endswith(".workflows.probe_authoring") for module in modules)
    assert PACKAGE_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROMPTS_ROOT / "initial_cap.txt").is_file()
    assert (PROMPTS_ROOT / "physic_probe.txt").is_file()


def test_unified_smoke_command_selects_single_policy_workflow(monkeypatch, tmp_path):
    import physrsr.paths as paths
    from physrsr.active.workflows import smoke
    calls = []
    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(smoke, "run", lambda root, **kwargs: calls.append((root, kwargs)))
    assert main(["smoke", "--out", str(tmp_path / "run"), "--initial-cap", str(tmp_path / "policy")]) == 0
    assert len(calls) == 1
    assert calls[0][1]["initial_cap"] == tmp_path / "policy"
    assert calls[0][1]["mass_kg"] == .27


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
