"""Immutable policy revisions and one execution path for both backends."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import hashlib
from pathlib import Path

from .contracts import API_VERSION, IDENTIFICATION_API_VERSION
from .recording import SessionRecorder


@dataclass(frozen=True)
class Policy:
    source: str
    rationale: str = ""
    parent: str | None = None
    api_version: str = API_VERSION

    def __post_init__(self):
        if self.api_version not in (API_VERSION, IDENTIFICATION_API_VERSION):
            raise ValueError("Unsupported policy API version")
        if not isinstance(self.source, str) or len(self.source.encode()) > 64000:
            raise ValueError("Policy must be Python source of at most 64KB")
        tree = ast.parse(self.source)
        entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
        if len(entries) != 1:
            raise ValueError("Policy must define run(robot)")
        args = entries[0].args
        if len(args.args) != 1 or args.posonlyargs or args.defaults or args.vararg or args.kwarg or args.kwonlyargs:
            raise ValueError("Policy entrypoint must be run(robot)")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name != "math" for a in node.names):
                raise ValueError("Only math imports are allowed")
            if isinstance(node, ast.ImportFrom) and (node.module != "math" or node.level):
                raise ValueError("Only math imports are allowed")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError("Private attributes are unavailable to policies")
            if isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "getattr", "setattr", "vars", "input", "breakpoint"}:
                raise ValueError("Policy must use the public robot API")

    @property
    def sha256(self):
        return hashlib.sha256(self.source.encode()).hexdigest()

    @property
    def behavior_sha256(self):
        """Only an obvious-duplicate check; executed interventions certify coverage."""
        tree=ast.parse(self.source)
        names={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):
                names.setdefault(node.id,f"local_{len(names)}")
        for node in ast.walk(tree):
            if isinstance(node,ast.Name) and node.id in names:node.id=names[node.id]
        normalized = ast.dump(tree, include_attributes=False)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def wire(self):
        return dict(source=self.source, sha256=self.sha256, api_version=self.api_version)

    @classmethod
    def from_wire(cls, value):
        policy = cls(value["source"], api_version=value.get("api_version"))
        if policy.sha256 != value.get("sha256"):
            raise ValueError("Policy hash/API mismatch")
        return policy


class PolicyStore:
    def __init__(self, root, *, recorder=None):
        self.root = Path(root)
        self.recorder = recorder if recorder is not None else SessionRecorder(self.root)
        self.policies = {}

    def add(self, policy, round_index):
        if policy.sha256 in self.policies:
            return
        recorder = self.recorder.scope(policy.sha256, exclusive=True)
        recorder.write_text("policy.py", policy.source)
        recorder.write_json("manifest.json", dict(
            sha256=policy.sha256, parent=policy.parent, round=round_index,
            rationale=policy.rationale, api_version=policy.api_version))
        if policy.parent in self.policies:
            old = self.policies[policy.parent].source
            recorder.write_text("revision.diff", "".join(difflib.unified_diff(
                old.splitlines(True), policy.source.splitlines(True), fromfile=policy.parent, tofile=policy.sha256)))
        self.policies[policy.sha256] = policy

    def freeze(self, policy, destination, evidence, *, recorder=None, execution=None):
        recorder = recorder if recorder is not None else SessionRecorder(destination)
        binding={}
        if execution is not None:
            from .bundle import BUNDLE_VERSION, content_hash
            recorder.write_json("identification.json",execution["identification"])
            recorder.write_json("task.json",execution["task"])
            recorder.write_json("models.json",execution["models"])
            recorder.write_json("config.json",execution["config"])
            recorder.write_json("run_manifest.json",execution["run_manifest"])
            binding=dict(bundle_version=BUNDLE_VERSION,world_identity=execution["world_identity"],
                identification_sha256=content_hash(execution["identification"]),task_sha256=content_hash(execution["task"]),
                models_sha256=content_hash(execution["models"]),model_version=execution["model_version"],
                config_sha256=content_hash(execution["config"]),run_manifest_sha256=content_hash(execution["run_manifest"]),
                runtime_contract=execution["runtime_contract"])
        # Copy exactly the bytes evaluated by the simulator; no sim/real branch.
        recorder.write_text("policy.py", policy.source)
        recorder.write_json("manifest.json", dict(
            sha256=policy.sha256, api_version=policy.api_version, entrypoint="run(robot)",
            status="task_sufficient", evidence=evidence,**binding))


def execute(policy, dispatch):
    from physrsr.execution import execute_source
    if policy.api_version == IDENTIFICATION_API_VERSION:
        # The shared physical budget is 360 s; allow controller/transport
        # overhead on hardware without imposing a shorter hidden wall deadline.
        return execute_source(policy.source, dispatch, timeout=480, capture_return=True, max_calls=192)
    return execute_source(policy.source, dispatch, capture_return=True)
