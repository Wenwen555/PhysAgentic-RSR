"""Task interface consumed by the generic predictive service."""
from typing import Protocol

from ..core.contracts import Hypothesis
from ..core.policy import Policy


class TaskAdapter(Protocol):
    task_id: str
    mechanisms: tuple[str, ...]
    parameter_contracts: dict

    def validate_task(self, task: dict) -> None:
        """Validate success, control and geometry contracts without interacting."""
        ...

    def validate(self, task: dict, model: Hypothesis) -> None:
        """Check task success/control constraints and supported physical parameters."""
        ...

    def execute(self, task: dict, model: Hypothesis, policy: Policy, seed: int,
                belief: dict, purpose: str, *, execution_budget: dict | None = None) -> dict:
        """Return task scores or probe observations through the shared v2 API."""
        ...
