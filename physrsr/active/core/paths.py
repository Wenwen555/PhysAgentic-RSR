"""Stable package locations, independent of workflow module nesting."""
from pathlib import Path
from ...paths import CAPX_ROOT, WORKSPACE_ROOT

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT
PROMPTS_ROOT = PACKAGE_ROOT / "prompts"
