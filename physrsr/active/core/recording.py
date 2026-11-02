"""Session persistence, separate from reasoning and environment execution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class Recorder(Protocol):
    root: Path

    def scope(self, name: str, *, exclusive: bool = False) -> Recorder: ...
    def write_json(self, name: str, value: Any) -> None: ...
    def write_text(self, name: str, value: str) -> None: ...
    def read_json(self, name: str) -> Any: ...
    def read_text(self, name: str) -> str: ...
    def event(self, kind: str, **value: Any) -> None: ...


class SessionRecorder:
    def __init__(self, root: str | Path, *, exclusive: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=not exclusive)

    def scope(self, name: str, *, exclusive: bool = False) -> SessionRecorder:
        return SessionRecorder(self.root / name, exclusive=exclusive)

    def write_json(self, name: str, value: Any) -> None:
        self.write_text(name, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))

    def write_text(self, name: str, value: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def read_json(self, name: str) -> Any:
        return json.loads(self.read_text(name))

    def read_text(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def event(self, kind: str, **value: Any) -> None:
        line = json.dumps(dict(event=kind, **value), ensure_ascii=False, allow_nan=False)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
