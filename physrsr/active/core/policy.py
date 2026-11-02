"""Immutable policy revisions and one execution path for both backends."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path

from .contracts import API_VERSION, IDENTIFICATION_API_VERSION


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
        """Ignore comments/formatting when separating fit/validation actions."""
        normalized = ast.dump(ast.parse(self.source), include_attributes=False)
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
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.policies = {}

    def add(self, policy, round_index):
        if policy.sha256 in self.policies:
            return
        folder = self.root / policy.sha256
        folder.mkdir()
        (folder / "policy.py").write_text(policy.source)
        (folder / "manifest.json").write_text(json.dumps(dict(
            sha256=policy.sha256, parent=policy.parent, round=round_index,
            rationale=policy.rationale, api_version=policy.api_version), indent=2))
        if policy.parent in self.policies:
            old = self.policies[policy.parent].source
            (folder / "revision.diff").write_text("".join(difflib.unified_diff(
                old.splitlines(True), policy.source.splitlines(True), fromfile=policy.parent, tofile=policy.sha256)))
        self.policies[policy.sha256] = policy

    def freeze(self, policy, destination, evidence):
        destination = Path(destination)
        destination.mkdir(exist_ok=False)
        # Copy exactly the bytes evaluated by the simulator; no sim/real branch.
        (destination / "policy.py").write_bytes(policy.source.encode())
        (destination / "manifest.json").write_text(json.dumps(dict(
            sha256=policy.sha256, api_version=policy.api_version, entrypoint="run(robot)",
            status="task_sufficient", evidence=evidence), indent=2))


def execute(policy, dispatch):
    from physrsr.coding.runtime import execute_source
    if policy.api_version == IDENTIFICATION_API_VERSION:
        # The shared physical budget is 360 s; allow controller/transport
        # overhead on hardware without imposing a shorter hidden wall deadline.
        return execute_source(policy.source, dispatch, timeout=480, capture_return=True, max_calls=192)
    return execute_source(policy.source, dispatch, capture_return=True)
