"""Portable source and runtime provenance; no experiment results in source docs."""
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess

from physrsr.paths import PACKAGE_ROOT, CAPX_ROOT


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(root):
    if not (Path(root)/".git").exists():return dict(commit=None,dirty=None)
    try:
        sha=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
        status=subprocess.check_output(["git","-C",str(root),"status","--porcelain","--untracked-files=no"],stderr=subprocess.DEVNULL,text=True)
        return dict(commit=sha,dirty=bool(status),status_sha256=hashlib.sha256(status.encode()).hexdigest())
    except (OSError,subprocess.CalledProcessError):return dict(commit=None,dirty=None)


def snapshot(recorder):
    sources={}
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if path.suffix not in (".py",".txt",".json") or "__pycache__" in path.parts:continue
        if not path.is_file():continue
        relative=path.relative_to(PACKAGE_ROOT).as_posix()
        sources[relative]=file_hash(path)
        recorder.write_text("task_context/source/physrsr/"+relative,path.read_text())
    versions={distribution.metadata["Name"]:distribution.version for distribution in metadata.distributions()
        if distribution.metadata.get("Name")}
    controllers={}
    config=CAPX_ROOT/"capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json"
    if config.is_file():
        controllers[config.name]=file_hash(config)
        recorder.write_text("task_context/source/controllers/"+config.name,config.read_text())
    manifest=dict(manifest_version="physrsr.run.v2",python=platform.python_version(),
        project=git_revision(PACKAGE_ROOT.parent),capx=git_revision(CAPX_ROOT),
        source_sha256=sources,controller_sha256=controllers,packages=versions)
    lock=PACKAGE_ROOT.parent/"dependencies.lock.json"
    if lock.is_file():
        import json
        manifest["dependency_lock"]=json.loads(lock.read_text())
    delivery=PACKAGE_ROOT.parent/"source_manifest.json"
    if delivery.is_file():
        import json
        manifest["source_delivery"]=dict(manifest_sha256=file_hash(delivery),project=json.loads(delivery.read_text()).get("project"))
    for name,root in (("project",PACKAGE_ROOT.parent),("capx",CAPX_ROOT),
            ("robosuite",CAPX_ROOT/"capx/third_party/robosuite")):
        revision=git_revision(root)
        if revision["commit"] is None:continue
        diff=subprocess.check_output(["git","-C",str(root),"diff","HEAD","--binary"],stderr=subprocess.DEVNULL)
        revision["diff_sha256"]=hashlib.sha256(diff).hexdigest()
        recorder.write_text(f"task_context/source/{name}.patch",diff.decode())
        manifest[name]=revision
    recorder.write_json("run_manifest.json",manifest)
    return manifest


def verify_snapshot(manifest):
    for name,digest in manifest["source_sha256"].items():
        path=PACKAGE_ROOT/name
        if not path.is_file() or file_hash(path)!=digest:
            raise ValueError("Source/prompt changed during the run: "+name)
    config=CAPX_ROOT/"capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json"
    if config.name in manifest["controller_sha256"] and file_hash(config)!=manifest["controller_sha256"][config.name]:
        raise ValueError("Controller configuration changed during the run")
