"""Coding Agent: generate and revise CaP through one measured v2 contract."""
import json

from .base import AgentService
from ..core.agent_models import PolicySynthesis
from ..api.client import APIError
from ..core.paths import PROMPTS_ROOT
from ..core.programs import task_candidate


SYNTHESIS_SCHEMA = PolicySynthesis.model_json_schema()


def synthesis_prompt(context):
    return (PROMPTS_ROOT/"identified_policy.txt").read_text().format(
        context=json.dumps(context,ensure_ascii=False,allow_nan=False,separators=(",",":")))


def validate_policy(answer):
    candidate=PolicySynthesis.model_validate(answer)
    return task_candidate(candidate.code)


class CodingService(AgentService):
    def synthesize_policy(self, context):
        return self.respond("identified",synthesis_prompt(context),schema=SYNTHESIS_SCHEMA,
            validate=validate_policy,
            label="Task-source authoring",context=context)

    def call(self, method, params):
        if method=="agent.describe" and not params:return self.descriptor()
        if method != "agent.synthesize_policy":
            raise APIError(f"Unknown coding operation: {method}")
        return self.synthesize_policy(params)
