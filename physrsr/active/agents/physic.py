"""Physic Agent: one measured Probe contract and physical reasoning only.

Assessment also handles reassessment and stop requests. The workflow owns
execution, belief commits, budgets, candidate evaluation and certification.
"""
import json

from .base import AgentService, STRING, STRINGS, obj, structured_response, validate_schema
from ..api.client import APIError
from ..core.contracts import Hypothesis, normalize
from ..core.evidence import available_evidence_ids, require_evidence
from ..core.exploration import CONTRACT, ACTIONS, validate_plan
from ..core.identification import ROBOT_API
from ..core.paths import PROMPTS_ROOT
from ..core.programs import source_policy, resumable_probe_source


DESIGN_SCHEMA = obj(dict(
    observations=dict(type="array", minItems=1, items=obj(dict(fact=STRING, evidence_ids=STRINGS))),
    task_relevant_factors=dict(type="array", minItems=1, items=obj(dict(
        factor=STRING, decision_affected=STRING, competing_explanations=STRINGS,
        identifiable_components=STRING, required_precision=STRING))),
    selected_factors=STRINGS,
    alternatives=dict(type="array", minItems=2, maxItems=4, items=obj(dict(
        interaction=STRING, information_gained=STRING, limitations=STRING, selection_tradeoff=STRING))),
    experimental_design=obj(dict(manipulated_variables=STRINGS, controlled_variables=STRINGS,
        minimum_distinct_settings=dict(type="integer", minimum=2, maximum=9),
        adaptive_next_setting=STRING, discriminating_response=STRING,
        sufficient_evidence_to_stop=STRING)),
    measurement_model=obj(dict(equations=STRINGS, assumptions=STRINGS, confounders=STRINGS,
        identifiable_subspace=STRING, uncertainty_calculation=STRING, falsification_checks=STRINGS)),
    embodiment_analysis=obj(dict(grasp_wrench_constraints=STRING, arm_torque_constraints=STRING,
        unknown_limits=STRINGS, effect_on_experiment_and_final_action=STRING)),
    estimator_outputs=dict(type="array", minItems=1, items=obj(dict(
        name=STRING, unit=STRING, frame=STRING, interpretation=STRING))),
    code=STRING, recovery=STRINGS, limitations=STRINGS))

REQUIREMENT_SCHEMA=obj(dict(quantity_or_condition=STRING,operation_decision=STRING,
    met=dict(type="boolean"),evidence_ids=dict(type="array",items=STRING)))
INSTANCE_ASSESSMENT_SCHEMA=obj(dict(episode_id=STRING,sufficient=dict(type="boolean"),reason=STRING,
    evidence_ids=dict(type="array",items=STRING),
    requirements=dict(type="array",minItems=1,items=REQUIREMENT_SCHEMA)))
DECISION_SCHEMA = obj(dict(
    action=dict(type="string",enum=sorted(ACTIONS|{"validate"})), rationale=STRING,
    instance_assessments=dict(type="array",minItems=1,items=INSTANCE_ASSESSMENT_SCHEMA),
    target_instance_ids=dict(type="array",items=STRING),
    interactions=dict(type="array",items=obj(dict(episode_id=STRING,uncertainty=STRING,
        why_existing_evidence_insufficient=STRING,evidence_ids=STRINGS,
        outcomes=dict(type="array",minItems=2,items=obj(dict(condition=STRING,operation_decision=STRING))),
        estimated_cost_s=dict(type="number",exclusiveMinimum=0)))),
    expected_decision_benefit=dict(type="number",minimum=0),
    estimated_probe_cost_s=dict(type="number",minimum=0)))


def probe_prompt(context):
    return (PROMPTS_ROOT/"physical_identification.txt").read_text().format(
        context=json.dumps(context,ensure_ascii=False,allow_nan=False,separators=(",",":")))


def validate_probe_design(answer, context):
    validate_schema(answer,DESIGN_SCHEMA)
    available=available_evidence_ids(context)
    for fact in answer["observations"]:
        require_evidence(fact["evidence_ids"],available)
    declared={item["factor"] for item in answer["task_relevant_factors"]}
    if not set(answer["selected_factors"])<=declared:
        raise ValueError("Selected factors must be declared in the physical analysis")
    return (source_policy(answer["code"]) if context.get("purpose","fit")=="validation"
        else resumable_probe_source(answer["code"]))


class PhysicAgent(AgentService):
    def assess_task(self, context):
        if (context.get("decision_contract")!=CONTRACT or not context.get("predictive_validation")
                or [item["episode_id"] for item in context.get("instances",[])]!=["current"]):
            raise APIError("Physic assessment requires the single-World predictive decision contract")
        prompt=(PROMPTS_ROOT/"world_decision.txt").read_text().format(
            context=json.dumps(context,ensure_ascii=False,allow_nan=False,separators=(",",":")))
        return structured_response(self.agent,prompt,self.destination("assessment"),schema=DECISION_SCHEMA,
            validate=lambda answer:validate_plan(answer,context),label="Physical task assessment")

    def design_probe(self, context):
        context=dict(context,robot_api=ROBOT_API)
        destination=self.destination("design")
        destination.mkdir(parents=True)
        (destination/"effective_context.json").write_text(json.dumps(context,ensure_ascii=False,indent=2))
        return structured_response(self.agent,probe_prompt(context),destination,schema=DESIGN_SCHEMA,
            validate=lambda answer:validate_probe_design(answer,context),label="Identification source authoring")

    def revise_physical_model(self, context):
        # Parameter names and bounds belong to the trusted Simulator adapter.
        context={key:context[key] for key in ("goal","models","parameter_contracts","belief",
            "fit_evidence","world","feature_schema") if key in context}
        choices=[]
        for mechanism,parameters in context["parameter_contracts"].items():
            fields={name:dict(type="number",**{key:value for key,value in limits.items()
                if key in ("minimum","maximum","exclusiveMinimum","exclusiveMaximum")})
                for name,limits in parameters.items()}
            choices.append(obj(dict(id=STRING,mechanism=dict(type="string",enum=[mechanism]),
                weight=dict(type="number",minimum=0),parameters=obj(fields))))
        schema=obj(dict(rationale=STRING,evidence_ids=STRINGS,
            models=dict(type="array",minItems=1,maxItems=128,items=dict(anyOf=choices))))
        fit_ids=[row["evidence_id"] for row in context["fit_evidence"]]
        def validate(answer):
            require_evidence(answer["evidence_ids"],fit_ids,exact=True)
            normalize([Hypothesis(**model) for model in answer["models"]])
        prompt=(PROMPTS_ROOT/"physical_model.txt").read_text().format(
            context=json.dumps(context,ensure_ascii=False,allow_nan=False))
        return structured_response(self.agent,prompt,self.destination("model"),schema=schema,
            validate=validate,label="Physical model revision")

    def call(self, method, params):
        operations={"physic.assess_task":self.assess_task,"physic.design_probe":self.design_probe,
            "physic.revise_physical_model":self.revise_physical_model}
        if method not in operations:
            raise APIError(f"Unknown Physic Agent operation: {method}")
        return operations[method](params)


# The HTTP server name is an alias of the same implementation and contract.
PhysicService = PhysicAgent
