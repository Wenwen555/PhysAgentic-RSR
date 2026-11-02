"""Shared inference transport, JSON-schema validation and bounded source review.

This module does not choose experiments, update beliefs or execute programs.
"""
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from ..api.client import APIError


STRING = dict(type="string")
STRINGS = dict(type="array", minItems=1, items=STRING)


def obj(properties):
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)


def validate_schema(answer, schema):
    json.dumps(answer,allow_nan=False)
    try:
        Draft202012Validator(schema).validate(answer)
    except ValidationError as exc:
        location=".".join(map(str,exc.absolute_path)) or "response"
        raise ValueError(f"{location}: {exc.message}") from None


def structured_response(agent, prompt, destination, *, schema, validate, label):
    """A failed source/contract review adds no physical interaction or evidence."""
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"prompt.txt").write_text(prompt)
    (destination/"schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2))
    messages=[dict(role="user",content=prompt)]
    for attempt in range(1,4):
        answer=agent.generate(messages,destination/f"turn_{attempt:04d}",schema=schema)
        try:
            validate_schema(answer,schema)
            validate(answer)
        except (ValueError,KeyError,TypeError,SyntaxError) as exc:
            messages += [dict(role="assistant",content=json.dumps(answer,ensure_ascii=False)),
                dict(role="user",content=f"{label} contract/source review only: {exc}. "
                    "Return the corrected complete JSON. No execution has occurred; no probe was executed "
                    "and no new physical or simulation evidence is available.")]
            continue
        return answer
    raise APIError(f"{label} failed after three attempts")


class AgentService:
    def __init__(self, output, agent=None):
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.turn=0
        if agent is None:
            from physrsr.coding.agent import CodexAgent
            agent=CodexAgent()
        self.agent=agent

    def destination(self, prefix):
        self.turn+=1
        return self.output/f"{prefix}_{self.turn:04d}"
