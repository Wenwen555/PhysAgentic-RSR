"""Shared inference transport, JSON-schema validation and bounded source review.

This module does not choose experiments, update beliefs or execute programs.
"""
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from ..api.client import APIError
from ..core.recording import SessionRecorder


def validate_schema(answer, schema):
    json.dumps(answer,allow_nan=False)
    try:
        Draft202012Validator(schema).validate(answer)
    except ValidationError as exc:
        location=".".join(map(str,exc.absolute_path)) or "response"
        raise ValueError(f"{location}: {exc.message}") from None


def structured_response(agent, prompt, destination, *, schema, validate, label, recorder=None, context=None):
    """A failed source/contract review adds no physical interaction or evidence."""
    destination=Path(destination)
    recorder = recorder if recorder is not None else SessionRecorder(destination, exclusive=False)
    recorder.write_text("prompt.txt", prompt)
    recorder.write_json("schema.json", schema)
    if context is not None:
        recorder.write_json("effective_context.json", context)
    messages=[dict(role="user",content=prompt)]
    errors=[]
    for attempt in range(1,4):
        answer=None
        generation_error=None
        try:
            answer=agent.generate(messages,destination/f"turn_{attempt:04d}",schema=schema)
        except json.JSONDecodeError as exc:
            # Only malformed model JSON is repaired at the generation boundary.
            generation_error=ValueError(f"Malformed generated JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}")
        try:
            if generation_error is not None:raise generation_error
            validate_schema(answer,schema)
            validate(answer)
        except (ValueError,KeyError,TypeError,SyntaxError) as exc:
            errors.append(dict(attempt=attempt,error_type=type(exc).__name__,error=str(exc)))
            recorder.write_json("validation_errors.json", errors)
            if answer is not None:
                messages.append(dict(role="assistant",content=json.dumps(answer,ensure_ascii=False)))
            messages.append(dict(role="user",content=f"{label} contract/source review only: {exc}. "
                    "Return the corrected complete JSON. No execution has occurred; no probe was executed "
                    "and no new physical or simulation evidence is available."))
            if attempt==3:
                raise APIError(f"{label} failed after three attempts; last validation error: {exc}") from exc
            continue
        recorder.write_json("response.json", answer)
        return answer


class AgentService:
    def __init__(self, output, agent=None, *, recorder=None):
        self.output=Path(output)
        self.recorder=recorder if recorder is not None else SessionRecorder(self.output)
        if self.recorder.root != self.output:
            raise ValueError("Agent recorder root must match output")
        self.turn=0
        if agent is None:
            from physrsr.coding.agent import CodexAgent
            agent=CodexAgent()
        self.agent=agent

    def destination(self, prefix):
        self.turn+=1
        return self.output/f"{prefix}_{self.turn:04d}"

    def respond(self, prefix, prompt, *, schema, validate, label, context):
        destination=self.destination(prefix)
        return structured_response(self.agent,prompt,destination,schema=schema,validate=validate,label=label,
            recorder=self.recorder.scope(destination.name),context=context)
