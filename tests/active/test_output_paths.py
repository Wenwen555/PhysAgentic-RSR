"""CLI outputs stay in the shared root without launching any experiment."""
import argparse

import pytest

import physrsr.paths as paths
from physrsr.active.__main__ import main


def test_output_name_resolves_independently_of_cwd(tmp_path, monkeypatch):
    root = tmp_path / "output"
    monkeypatch.setattr(paths, "OUTPUT_ROOT", root)
    monkeypatch.chdir(tmp_path)
    assert paths.output_path("run/active_probe_1") == root / "run/active_probe_1"
    assert paths.output_path(root / "run") == root / "run"
    assert not root.exists()


def test_output_rejects_parent_absolute_and_symlink_escapes(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(paths, "OUTPUT_ROOT", root)
    for value in ("../bad", tmp_path / "bad", "outside/bad", root):
        with pytest.raises(argparse.ArgumentTypeError):
            paths.output_path(value)


def test_cli_rejects_old_output_before_work_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path / "output")
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", "missing.json", "--out", str(tmp_path / "old")])
    assert exc.value.code == 2
    assert not (tmp_path / "output").exists()
