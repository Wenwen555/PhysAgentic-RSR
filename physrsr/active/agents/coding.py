"""Coding Agent: generate and revise CaP through one measured v2 contract."""
import json

from .base import AgentService, STRING, STRINGS, obj, structured_response
from ..api.client import APIError
from ..core.paths import PROMPTS_ROOT
from ..core.programs import task_candidate


SYNTHESIS_SCHEMA = obj(dict(code=STRING, rationale=STRING,
    changes=STRINGS, embodiment_checks=STRINGS, remaining_uncertainties=STRINGS))


def synthesis_prompt(context):
    return (PROMPTS_ROOT/"identified_policy.txt").read_text().format(
        context=json.dumps(context,ensure_ascii=False,allow_nan=False,separators=(",",":")))



class CodingService(AgentService):
    def synthesize_policy(self, context):
        return structured_response(self.agent,synthesis_prompt(context),self.destination("identified"),
            schema=SYNTHESIS_SCHEMA,validate=lambda answer:task_candidate(answer["code"]),label="Task-source authoring")

    def call(self, method, params):
        if method != "agent.synthesize_policy":
            raise APIError(f"Unknown coding operation: {method}")
        return self.synthesize_policy(params)
