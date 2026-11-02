"""Source and dependency roots; importing this module has no runtime effects."""
from pathlib import Path
import argparse
import importlib.util
import os

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
def capx_root():
    configured=os.environ.get("PHYSRSR_CAPX_ROOT")
    if configured:
        root=Path(configured).expanduser().resolve()
        if not (root/"capx").is_dir():raise ValueError("PHYSRSR_CAPX_ROOT must contain the capx package")
        return root
    spec=importlib.util.find_spec("capx")
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve().parent
    local=WORKSPACE_ROOT/"src"/"cap-x"
    if (local/"capx").is_dir():return local
    # Offline help/config/export can work without allocating or importing capx.
    return local


# Resolve installed package resources; no requirement for an adjacent checkout.
CAPX_ROOT = capx_root()
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
