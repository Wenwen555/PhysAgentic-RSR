"""Source and dependency roots; importing this module has no runtime effects."""
from pathlib import Path
import argparse

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
CAPX_ROOT = WORKSPACE_ROOT / "src" / "cap-x"
OUTPUT_ROOT = Path("/data/wjt/PhyAgent-RSR/output")


def output_path(value: str | Path) -> Path:
    """Resolve CLI write destinations under the project's shared output root.

    Importing or parsing paths never creates directories or starts a run.
    Programmatic workflow callers can still inject temporary paths for tests.
    """
    root = OUTPUT_ROOT.resolve()
    path = Path(value).expanduser()
    destination = (path if path.is_absolute() else root / path).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise argparse.ArgumentTypeError(f"Output must be a child of {root}; use a run name or a path inside it")
    return destination
