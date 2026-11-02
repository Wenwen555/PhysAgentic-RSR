"""Complete structured-output fixtures, never claimed physical reasoning."""
from physrsr.active.core.evidence import available_evidence_ids


def design_answer(context, code):
    validation=None
    if context.get('purpose')=='validation':
        validation=dict(candidate_sha256=context.get('candidate_sha256','fixture'),
            object_ids=list(context.get('validation_requirements',{'object':{}})),
            hypothesis_ids=[m['id'] for m in context.get('models',[]) if m['weight']>0],
            alternative_explanation='Unmodeled measured response',task_operation='Fixture contact and hold',
            max_prediction_error=context.get('max_prediction_error',3.),settings=[dict(object_id=key,
                tcp_offset_body_m=[0.,0.,0.],setting_tolerance_m=.003,**{'min_object_translation_m':0.,'min_rotation_rad':0.,**requirements})
                for key,requirements in context.get('validation_requirements',{}).items()])
    return dict(validation=validation,
        observations=[dict(fact="Supplied numerical fixture",evidence_ids=sorted(available_evidence_ids(context)))],
        task_relevant_factors=[dict(factor="response",decision_affected="Choose operation",
            competing_explanations=["fixture contrast"],identifiable_components="fixture response",
            required_precision="fixture tolerance")],
        selected_factors=["response"],
        alternatives=[dict(interaction=name,information_gained="Fixture response",limitations="Not physics",
            selection_tradeoff="Fixture contrast") for name in ("contact setting one","contact setting two")],
        experimental_design=dict(manipulated_variables=["contact configuration"],controlled_variables=["fixture state"],
            minimum_distinct_settings=len(validation["settings"]) if validation else 2,adaptive_next_setting="next fixture setting",
            discriminating_response="signed response",sufficient_evidence_to_stop="both fixture settings observed"),
        measurement_model=dict(equations=["response = fixture sensor"],assumptions=["numerical fixture"],
            confounders=["not a physical experiment"],identifiable_subspace="fixture response",
            uncertainty_calculation="fixture tolerance",falsification_checks=["fixture mismatch"]),
        embodiment_analysis=dict(grasp_wrench_constraints="fixture limits",arm_torque_constraints="fixture limits",
            unknown_limits=["fixture only"],effect_on_experiment_and_final_action="fixture bounds"),
        estimator_outputs=[dict(name="response",unit="N",frame="robot_base_x",interpretation="fixture only")],
        code=code,recovery=["supported fixture state"],limitations=["not physical identification"])
