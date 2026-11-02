"""Physic Agent: one measured Probe contract and physical reasoning only.

Assessment also handles reassessment and stop requests. The workflow owns
execution, belief commits, budgets, candidate evaluation and certification.
"""
import json

from .base import AgentService
from ..core.agent_models import ProbeDesign, TaskDecision, InstanceAssessment, Requirement, physical_revision_model
from ..api.client import APIError
from ..core.exploration import CONTRACT
from ..core.identification import ROBOT_API
from ..core.paths import PROMPTS_ROOT
from ..core.programs import source_policy, resumable_probe_source


DESIGN_SCHEMA = ProbeDesign.model_json_schema()
REQUIREMENT_SCHEMA = Requirement.model_json_schema()
INSTANCE_ASSESSMENT_SCHEMA = InstanceAssessment.model_json_schema()
DECISION_SCHEMA = TaskDecision.model_json_schema()


def probe_prompt(context):
    return (PROMPTS_ROOT/"physical_identification.txt").read_text().format(
        context=json.dumps(context,ensure_ascii=False,allow_nan=False,separators=(",",":")))


def validate_probe_design(answer, context):
    ProbeDesign.model_validate(answer, context=context)
    return (source_policy(answer["code"]) if context.get("purpose","fit")=="validation"
        else resumable_probe_source(answer["code"]))


class PhysicAgent(AgentService):
    def assess_task(self, context):
        if (context.get("decision_contract")!=CONTRACT or not context.get("predictive_validation")
                or [item["episode_id"] for item in context.get("instances",[])]!=["current"]):
            raise APIError("Physic assessment requires the single-World predictive decision contract")
        prompt=(PROMPTS_ROOT/"world_decision.txt").read_text().format(
            context=json.dumps(context,ensure_ascii=False,allow_nan=False,separators=(",",":")))
        return self.respond("assessment",prompt,schema=DECISION_SCHEMA,
            validate=lambda answer:TaskDecision.model_validate(answer,context=context),
            label="Physical task assessment",context=context)

    def design_probe(self, context):
        context=dict(context,robot_api=ROBOT_API)
        return self.respond("design",probe_prompt(context),schema=DESIGN_SCHEMA,
            validate=lambda answer:validate_probe_design(answer,context),
            label="Identification source authoring",context=context)

    def revise_physical_model(self, context):
        # Parameter names and bounds belong to the trusted Simulator adapter.
        context={key:context[key] for key in ("goal","models","parameter_contracts","belief",
            "fit_evidence","world","feature_schema") if key in context}
        model=physical_revision_model(context["parameter_contracts"])
        schema=model.model_json_schema()
        prompt=(PROMPTS_ROOT/"physical_model.txt").read_text().format(
            context=json.dumps(context,ensure_ascii=False,allow_nan=False))
        return self.respond("model",prompt,schema=schema,
            validate=lambda answer:model.model_validate(answer,context=context),
            label="Physical model revision",context=context)

    def call(self, method, params):
        operations={"physic.assess_task":self.assess_task,"physic.design_probe":self.design_probe,
            "physic.revise_physical_model":self.revise_physical_model}
        if method not in operations:
            raise APIError(f"Unknown Physic Agent operation: {method}")
        return operations[method](params)


# The HTTP server name is an alias of the same implementation and contract.
PhysicService = PhysicAgent
