"""Task-independent physical experimental design; no benchmark dispatch/templates."""
import ast
import json
from copy import deepcopy
from pathlib import Path

from .coding import STRING, obj
from ..core.contracts import IDENTIFICATION_API_VERSION
from ..core.identification import ROBOT_API
from ..core.paths import PROMPTS_ROOT
from ..core.policy import Policy
from ..core.exploration import CONTRACT, ACTIONS, validate_plan


STRINGS = dict(type="array", minItems=1, items=STRING)
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

SYNTHESIS_SCHEMA = obj(dict(code=STRING, rationale=STRING,
    changes=STRINGS, embodiment_checks=STRINGS, remaining_uncertainties=STRINGS))

DECISION_SCHEMA = obj(dict(action=dict(type="string", enum=["probe","revise_policy","validate","stop"]),
    rationale=STRING, uncertainties=dict(type="array",items=obj(dict(factor=STRING,
        decision_affected=STRING,evidence_ids=STRINGS))),
    estimates_sufficient=dict(type="boolean"), expected_decision_benefit=dict(type="number",minimum=0),
    estimated_probe_cost_s=dict(type="number",minimum=0)))

REQUIREMENT_SCHEMA=obj(dict(quantity_or_condition=STRING,operation_decision=STRING,
    met=dict(type="boolean"),evidence_ids=dict(type="array",items=STRING)))
INSTANCE_ASSESSMENT_SCHEMA=obj(dict(episode_id=STRING,sufficient=dict(type="boolean"),reason=STRING,
    evidence_ids=dict(type="array",items=STRING),
    requirements=dict(type="array",minItems=1,items=REQUIREMENT_SCHEMA)))
SCOPED_DECISION_SCHEMA = obj(dict(
    action=dict(type="string",enum=sorted(ACTIONS)), rationale=STRING,
    instance_assessments=dict(type="array",minItems=1,items=INSTANCE_ASSESSMENT_SCHEMA),
    target_instance_ids=dict(type="array",items=STRING),
    interactions=dict(type="array",items=obj(dict(episode_id=STRING,uncertainty=STRING,
        why_existing_evidence_insufficient=STRING,evidence_ids=STRINGS,
        outcomes=dict(type="array",minItems=2,items=obj(dict(condition=STRING,operation_decision=STRING))),
        estimated_cost_s=dict(type="number",exclusiveMinimum=0)))),
    expected_decision_benefit=dict(type="number",minimum=0),
    estimated_probe_cost_s=dict(type="number",minimum=0)))


def assess_task(agent, context, destination):
    scoped=context.get("decision_contract")==CONTRACT
    prompt = (PROMPTS_ROOT / ("scoped_decision.txt" if scoped else "loop_decision.txt")).read_text().format(
        context=json.dumps(context,ensure_ascii=False,allow_nan=False,separators=(",",":")))
    messages=[dict(role="user",content=prompt)]
    schema=deepcopy(SCOPED_DECISION_SCHEMA) if scoped else DECISION_SCHEMA
    if scoped and context.get("predictive_validation"):
        schema["properties"]["action"]["enum"].append("validate")
    for attempt in range(1,4):
        answer=agent.generate(messages,Path(destination)/f"turn_{attempt:04d}" if scoped else destination,
            schema=schema)
        try:
            return validate_plan(answer,context) if scoped else answer
        except (ValueError,KeyError,TypeError) as exc:
            messages += [dict(role="assistant",content=json.dumps(answer,ensure_ascii=False)),
                dict(role="user",content=f"Decision-contract review only; no robot interaction occurred: {exc}. Correct the decision.")]
    raise ValueError("No valid scoped exploration decision")


def revise_predictive_model(agent, context, destination):
    """Parameter names come from the simulator adapter, never a task branch."""
    choices = []
    for mechanism, parameters in context["parameter_contracts"].items():
        fields = {name:dict(type="number",**{k:v for k,v in limits.items()
            if k in ("minimum","maximum","exclusiveMinimum","exclusiveMaximum")})
            for name,limits in parameters.items()}
        choices.append(obj(dict(id=STRING,mechanism=dict(type="string",enum=[mechanism]),
            weight=dict(type="number",minimum=0),parameters=obj(fields))))
    schema = obj(dict(rationale=STRING,evidence_ids=STRINGS,
        models=dict(type="array",minItems=1,maxItems=128,items=dict(anyOf=choices))))
    instruction = """Revise an executable predictive physical model ensemble from the supplied
measured identification evidence. Use only supported parameter names, mechanisms,
units and frames from parameter_contracts. Explain changes in Chinese. Preserve
plausible uncertainty and unidentifiable parameters. Bind numerical estimates to
their stated units/frames; do not treat a recommended action setting as a physical
parameter. Evidence IDs must identify the supplied fit interactions. Do not use
validation data to fit, invent mechanism implementations, or claim prediction
success. The host will replay every fit interaction before committing the model.
Return complete model hypotheses, not a policy or simulator source code.
Context:\n"""
    return agent.generate([dict(role="user",content=instruction+json.dumps(context,
        ensure_ascii=False,allow_nan=False))],destination,schema=schema)


def design_prompt(context):
    return (PROMPTS_ROOT / "physical_identification.txt").read_text().format(
        context=json.dumps(context, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def source_policy(source):
    policy = Policy(source, api_version=IDENTIFICATION_API_VERSION)
    methods = {"describe", "observe", "hold", "checkpoint", "goto_pose", "open_gripper", "close_gripper"}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "robot" and node.attr not in methods):
            raise ValueError("Source uses an unavailable measured API method")
    return policy


def resumable_probe_source(source):
    policy=source_policy(source)
    tree=ast.parse(source)
    methods={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)
        and isinstance(n.value,ast.Name) and n.value.id=="robot"}
    literals={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    if not {"checkpoint","describe"}<=methods or not {"probe_execution","resume_state"}<=literals:
        raise ValueError("Probe must read the same-source resume_state and publish a safe budget checkpoint")
    return policy


def generate_design(agent, context, destination):
    destination = Path(destination)
    context=dict(context,robot_api=ROBOT_API)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"effective_context.json").write_text(json.dumps(context,ensure_ascii=False,indent=2))
    prompt = design_prompt(context)
    messages = [dict(role="user", content=prompt)]
    for attempt in range(1, 4):
        answer = agent.generate(messages, destination / f"turn_{attempt:04d}", schema=DESIGN_SCHEMA)
        try:
            if context.get("purpose","fit")=="validation":source_policy(answer["code"])
            else:resumable_probe_source(answer["code"])
            if answer["experimental_design"]["minimum_distinct_settings"] < 2:
                raise ValueError("Numerical identification requires a discriminating multi-setting experiment")
        except (ValueError, SyntaxError, KeyError) as exc:
            messages += [dict(role="assistant", content=json.dumps(answer, ensure_ascii=False)),
                dict(role="user", content=f"Source/contract review only, no execution: {exc}. Return the corrected complete JSON.")]
            continue
        return answer
    raise ValueError("Identification source authoring failed")


def synthesize(agent, context, destination):
    prompt = (PROMPTS_ROOT / "identified_policy.txt").read_text().format(
        context=json.dumps(context, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    messages = [dict(role="user", content=prompt)]
    for attempt in range(1, 4):
        answer = agent.generate(messages, Path(destination) / f"turn_{attempt:04d}", schema=SYNTHESIS_SCHEMA)
        try:
            task_candidate(answer["code"])
        except (ValueError, SyntaxError, KeyError) as exc:
            messages += [dict(role="assistant", content=json.dumps(answer, ensure_ascii=False)),
                dict(role="user", content=f"Task-source contract review only; no task execution occurred: {exc}. Return a complete executable candidate, retaining physical uncertainty in the analysis.")]
            continue
        return answer
    raise ValueError("No executable task candidate after source review")


def task_candidate(source):
    policy = source_policy(source)
    # Bound public methods can be passed to a local RPC-counting helper. This
    # minimal source check must not reject that legitimate call pattern. Actual
    # reachability and task completion are established by execution, not AST.
    actions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name) and n.value.id == "robot"
        and n.attr in {"goto_pose", "open_gripper", "close_gripper"}]
    if not actions:
        raise ValueError("A task candidate must include a state-changing public robot action; a no-op certifier is not synthesis")
    return policy
