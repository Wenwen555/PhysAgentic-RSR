"""Task interface consumed by the generic predictive service."""
from typing import Protocol

from ..core.contracts import Hypothesis
from ..core.policy import Policy


class TaskAdapter(Protocol):
    task_id: str
    mechanisms: tuple[str, ...]
    parameter_contracts: dict

    def validate(self, task: dict, model: Hypothesis) -> None:
        """Check task success/control constraints and supported physical parameters."""
        ...

    def execute(self, task: dict, model: Hypothesis, policy: Policy, seed: int,
                belief: dict, purpose: str) -> dict:
        """Return task scores or probe observations through the shared v2 API."""
        ...
