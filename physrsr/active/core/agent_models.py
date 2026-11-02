"""Typed Agent contracts; adapters supply physical parameter names and bounds.

The exported JSON schemas and Python validation derive from the same models.
Context-dependent validators only inspect supplied public evidence.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, create_model, model_validator

from .contracts import Hypothesis, normalize
from .evidence import available_evidence_ids, require_evidence
from .exploration import validate_plan


Strings = Annotated[list[str], Field(min_length=1)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ObservedFact(ContractModel):
    fact: str
    evidence_ids: Strings


class PhysicalFactor(ContractModel):
    factor: str
    decision_affected: str
    competing_explanations: Strings
    identifiable_components: str
    required_precision: str


class ProbeAlternative(ContractModel):
    interaction: str
    information_gained: str
    limitations: str
    selection_tradeoff: str


class ExperimentalDesign(ContractModel):
    manipulated_variables: Strings
    controlled_variables: Strings
    minimum_distinct_settings: Annotated[int, Field(ge=2, le=9)]
    adaptive_next_setting: str
    discriminating_response: str
    sufficient_evidence_to_stop: str


class MeasurementModel(ContractModel):
    equations: Strings
    assumptions: Strings
    confounders: Strings
    identifiable_subspace: str
    uncertainty_calculation: str
    falsification_checks: Strings


class EmbodimentAnalysis(ContractModel):
    grasp_wrench_constraints: str
    arm_torque_constraints: str
    unknown_limits: Strings
    effect_on_experiment_and_final_action: str


class EstimatorOutput(ContractModel):
    name: str
    unit: str
    frame: str
    interpretation: str


class ProbeDesign(ContractModel):
    observations: Annotated[list[ObservedFact], Field(min_length=1)]
    task_relevant_factors: Annotated[list[PhysicalFactor], Field(min_length=1)]
    selected_factors: Strings
    alternatives: Annotated[list[ProbeAlternative], Field(min_length=2, max_length=4)]
    experimental_design: ExperimentalDesign
    measurement_model: MeasurementModel
    embodiment_analysis: EmbodimentAnalysis
    estimator_outputs: Annotated[list[EstimatorOutput], Field(min_length=1)]
    code: str
    recovery: Strings
    limitations: Strings

    @model_validator(mode="after")
    def check_analysis(self, info: ValidationInfo) -> ProbeDesign:
        if not set(self.selected_factors) <= {item.factor for item in self.task_relevant_factors}:
            raise ValueError("Selected factors must be declared in the physical analysis")
        if info.context is not None:
            available = available_evidence_ids(info.context)
            for fact in self.observations:
                require_evidence(fact.evidence_ids, available)
        return self


class Requirement(ContractModel):
    quantity_or_condition: str
    operation_decision: str
    met: bool
    evidence_ids: list[str]


class InstanceAssessment(ContractModel):
    episode_id: str
    sufficient: bool
    reason: str
    evidence_ids: list[str]
    requirements: Annotated[list[Requirement], Field(min_length=1)]


class InteractionOutcome(ContractModel):
    condition: str
    operation_decision: str


class Interaction(ContractModel):
    episode_id: str
    uncertainty: str
    why_existing_evidence_insufficient: str
    evidence_ids: Strings
    outcomes: Annotated[list[InteractionOutcome], Field(min_length=2)]
    estimated_cost_s: Positive


class TaskDecision(ContractModel):
    action: Literal["probe", "execute_probe", "resume_probe", "repair_probe", "revise_policy", "stop", "validate"]
    rationale: str
    instance_assessments: Annotated[list[InstanceAssessment], Field(min_length=1)]
    target_instance_ids: list[str]
    interactions: list[Interaction]
    expected_decision_benefit: Nonnegative
    estimated_probe_cost_s: Nonnegative

    @model_validator(mode="after")
    def check_decision(self, info: ValidationInfo) -> TaskDecision:
        if info.context is not None:
            validate_plan(self.model_dump(), info.context)
        return self


class PolicySynthesis(ContractModel):
    code: str
    rationale: str
    changes: Strings
    embodiment_checks: Strings
    remaining_uncertainties: Strings


class PhysicalModelRevision(ContractModel):
    rationale: str
    evidence_ids: Strings

    @model_validator(mode="after")
    def check_fit(self, info: ValidationInfo) -> PhysicalModelRevision:
        if info.context is not None:
            require_evidence(self.evidence_ids,
                [row["evidence_id"] for row in info.context["fit_evidence"]], exact=True)
        normalize([Hypothesis(**model) for model in self.model_dump()["models"]])
        return self


def physical_revision_model(contracts: dict[str, Any]) -> type[BaseModel]:
    """Generate strict parameter models from trusted adapter contracts, not LLM text."""
    choices: list[Any] = []
    bounds = {"minimum": "ge", "maximum": "le", "exclusiveMinimum": "gt", "exclusiveMaximum": "lt"}
    for index, (mechanism, parameters) in enumerate(contracts.items()):
        fields: dict[str, Any] = {name: (float, Field(..., **{bounds[key]: value for key, value in limits.items()
            if key in bounds})) for name, limits in parameters.items()}
        parameter_model = create_model(f"Parameters{index}", __base__=ContractModel, **fields)
        choices.append(create_model(f"Hypothesis{index}", __base__=ContractModel,
            id=(str, ...), mechanism=(Literal[mechanism], ...),
            weight=(Nonnegative, ...), parameters=(parameter_model, ...)))
    if not choices:
        raise ValueError("Physical model revision requires adapter parameter contracts")
    hypothesis_type = choices[0] if len(choices) == 1 else Union[tuple(choices)]
    # This annotation is dynamic because the trusted task adapter supplies it.
    sequence: Any = list
    return create_model("ModelRevision", __base__=PhysicalModelRevision,
        models=(sequence[hypothesis_type], Field(..., min_length=1, max_length=128)))
