"""Probe-first v2 loop: task/vision, scientist, measurement, model and policy.

All environment/model/agent interactions cross the supplied APIs. The trusted
simulator owns task adapters; no task name or physical factor is dispatched here.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import uuid

from ..core.recording import SessionRecorder
from ..core.bundle import execution_identity, content_hash
from ..core.provenance import snapshot, verify_snapshot
from ..core.validation import task_coverage, check_executed, intervention_trace
from ..api.client import RecordedCalls
from ..core.evidence import evidence_samples, prediction_feedback
from ..core.belief import merge_identification
from ..core.contracts import IDENTIFICATION_API_VERSION, Hypothesis, normalize, finite
from ..core.identification import ROBOT_API, DEFAULT_BUDGET, public_goal
from ..core.exploration import CONTRACT,PHYSICAL_ACTIONS,PAUSED_STATUSES,resumable,validate_plan
from ..core.measurements import (FEATURE_SCHEMA, SCENE_FEATURE_SCHEMA, FEATURE_NAMES, FEATURE_SCALES,
    response_features, response_schema, scene_scales)
from ..core.scene import object_ids, check_objects, visual_observation, supported
from ..core.policy import Policy, PolicyStore
from ..core.reasoning import Probe, assess, posterior, predictive_error, feature_error, probe_value, validate_predictions, joint_fit
from ..core.programs import source_policy, task_candidate
from ..agents.physic import DESIGN_SCHEMA, probe_prompt, validate_probe_design
from ..agents.coding import SYNTHESIS_SCHEMA, synthesis_prompt
from ..runtime.instrumented import observation


class ActiveLoop:
    def __init__(self, task, config, models, *, agent, physic, simulator, target, output, recorder=None):
        if len(config.feature_scales)!=len(FEATURE_NAMES):raise ValueError("Expected 16 explicit base feature scales")
        scales = config.feature_scales
        self.task, self.cfg, self.models = task, replace(config,feature_scales=scales), normalize(models) if models else ()
        self.agent, self.physic, self.simulator, self.target = agent, physic, simulator, target
        self.output = Path(output)
        self.recorder = recorder if recorder is not None else SessionRecorder(self.output)
        if self.recorder.root != self.output:
            raise ValueError("Loop recorder root must match output")
        self.rpc = RecordedCalls(self.recorder)
        self.run_manifest=snapshot(self.recorder)
        self.store = PolicyStore(self.output / "policies",recorder=self.recorder.scope("policies",exclusive=True))
        self.session_id = uuid.uuid4().hex
        self.model_version = self.probe_count = self.revision_count = self.model_revisions = 0
        self.spent = 0.
        self.heuristic_spent=0.
        self.heuristic_count=0.
        self.probe_predictions={}
        self.prediction_budgets={}
        self.cost_complete = True
        self.probe_attempt_count = 0
        self.fit_data, self.validation_data, self.history = [], [], []
        self.probe_data=[]
        self.current_probe=None
        self.execution_attempts={}
        self.probe_design_count=self.probe_repair_count=0
        self.probe_experiment_id=None
        self.validation_checkpoints={}
        self.validation_segments={}
        self.validation_traces={}
        self.belief = dict(estimates=[],instance_id="current",evidence_source="no_interaction_yet")
        self.evidence_ids = set()
        self.assessment, self.rows = None, []
        self.needs_identification = False
        self.model_diagnostics = None
        self.validation_failures = []
        self.sim_configured = False
        self.scene_object_ids = None
        self.feature_schema = FEATURE_SCHEMA
        self.write("config.json",dict(task=task.public(IDENTIFICATION_API_VERSION),loop=asdict(self.cfg),
            models=self.model_json(),session_id=self.session_id))

    def write(self, name, value):
        self.recorder.write_json(name,value)

    def event(self, kind, **value):
        self.recorder.event(kind,**value)

    def call(self, endpoint, method, params):
        return self.rpc.call(endpoint,method,params)

    def model_json(self):
        return [asdict(m) for m in self.models]

    def sim_params(self):
        return dict(session_id=self.session_id,model_version=self.model_version,seeds=list(self.cfg.model_seeds))

    def checked_version(self, response, method):
        return self.rpc.checked_version(response,session_id=self.session_id,
            method=method,expected=self.model_version)

    def evaluate(self, destination):
        candidates = list(self.store.policies.values())[-self.cfg.max_candidates:]
        if self.assessment and all(p.sha256 != self.assessment["selected"] for p in candidates):
            candidates = [self.store.policies[self.assessment["selected"]],*candidates[:self.cfg.max_candidates-1]]
        response = self.checked_version(self.call(self.simulator,"sim.evaluate",dict(**self.sim_params(),
            policies=[p.wire() for p in candidates])),"sim.evaluate")
        self.rows = response["rows"]
        self.assessment = assess(self.models,candidates,self.rows,self.cfg)
        self.write(destination,dict(model_version=self.model_version,assessment=self.assessment,rows=self.rows))
        return candidates

    evidence_samples = staticmethod(evidence_samples)

    def prediction_feedback(self):
        return prediction_feedback(self.rows)

    def context(self):
        """Scientist gets a semantic goal and measured/predictive evidence, no task ID."""
        self.capabilities=self.call(self.target,"target.describe",{})
        current=observation(self.call(self.target,"target.observe",{}))
        self.check_scene(current)
        if not self.probe_data:
            current=visual_observation(current)
        evidence={"task_goal","robot_capabilities","initial_visual"}
        evidence.update(d["result"]["evidence_id"] for d in self.probe_data)
        evidence.update(v["evidence_id"] for v in self.validation_data)
        if self.rows:evidence.add(f"prediction:v{self.model_version}")
        context=dict(goal=public_goal(self.task.public(IDENTIFICATION_API_VERSION)),robot_api=ROBOT_API,
            decision_contract=CONTRACT,predictive_validation=True,
            world=dict(id="current",object_ids=self.scene_object_ids,one_robot=True,
                shared_scene=True,independent_instances=False),
            instances=[dict(episode_id="current",retained_identification=self.belief,
                available_evidence_ids=sorted(evidence),
                latest_execution=(dict(status=self.probe_data[-1]["result"]["status"],
                    error=self.probe_data[-1]["result"].get("error"),
                    program_return=self.probe_data[-1]["result"].get("program_return"),
                    observations=self.evidence_samples(self.probe_data[-1]["result"]["observations"],limit=18))
                    if self.probe_data else None))],
            capabilities=self.capabilities,current_observation=current,
            phase="model_repair" if self.needs_identification else "grounded_revision" if self.model_version>0 else "bootstrap_identification",
            initial_evidence=([dict(evidence_id=f"prediction:v{self.model_version}",source="predictive_model",
                **self.prediction_feedback())] if self.rows else [dict(evidence_id="initial_visual",
                source="initial_visual_observation",observation=self.initial_visual)]),
            assessment=self.assessment,model_version=self.model_version,models=self.model_json(),
            model_diagnostics=self.model_diagnostics,unresolved_validation_failures=self.unresolved_failures(),
            prior_identification_evidence=[dict(evidence_id=d["result"]["evidence_id"],
                program_return=d["result"].get("program_return"),
                observations=self.evidence_samples(d["result"]["observations"]))
                for d in self.probe_data],
            validation_reports=self.validation_data,identification=self.belief,
            pending_validation_segments=self.validation_segments,
            prediction_mismatch_requires_new_fit=self.needs_identification,
            heuristic_budget=dict(remaining_interactions=self.cfg.max_heuristic_probes-self.heuristic_count,
                remaining_cost_s=self.cfg.max_heuristic_cost-self.heuristic_spent,
                meaning="Uncalibrated bootstrap/new-program exploration, separate from computed candidate decision value"),
            experiment_budget=dict(DEFAULT_BUDGET,remaining_interactions=self.cfg.max_probes-self.probe_count,
                remaining_cost_s=max(0.,self.cfg.max_probe_cost-self.spent)))
        if self.current_probe:
            context["current_probe"]=dict(source=self.current_probe.policy.source,sha256=self.current_probe.policy.sha256,
                purpose=self.current_probe.purpose,
                resumable=resumable(self.probe_memory(self.current_probe),self.current_probe.policy.sha256))
        if self.assessment:
            selected=self.assessment["selected"]
            context.update(current_task_policy=self.store.policies[selected].source,candidate_sha256=selected,
                validation_requirements=task_coverage(self.rows,selected,self.scene_object_ids),
                max_prediction_error=self.cfg.max_prediction_error)
        return context

    def probe_memory(self,probe):
        if probe.purpose!="validation":return self.belief
        checkpoint=self.validation_checkpoints.get(probe.policy.sha256)
        if checkpoint is None:return self.belief
        progress=dict(self.belief.get("probe_progress",{}));progress[probe.policy.sha256]=checkpoint
        return dict(self.belief,probe_progress=progress)

    def revise_policy(self, context):
        self.revision_count += 1
        parent = self.store.policies[self.assessment["selected"]] if self.assessment else None
        stage = f"revised_cap_{self.revision_count}" if parent else "candidate_cap_1"
        self.recorder.scope(stage,exclusive=True)
        payload = dict(context,previous_task_candidate=parent.source if parent else None,
            prediction_feedback=self.prediction_feedback(),execution_protocol=dict(one_shared_policy=True,
                one_world=True,object_estimates_from_api=True,target_task_executed=False))
        self.write(stage+"/context.json",payload)
        self.write(stage+"/schema.json",SYNTHESIS_SCHEMA)
        self.recorder.write_text(stage+"/prompt.txt",synthesis_prompt(payload))
        answer = self.call(self.agent,"agent.synthesize_policy",payload)
        self.write(stage+"/response.json",answer)
        checked = task_candidate(answer["code"])
        policy = Policy(checked.source,answer["rationale"],parent.sha256 if parent else None,IDENTIFICATION_API_VERSION)
        self.store.add(policy,self.revision_count)
        self.recorder.write_text(stage+"/policy.py",policy.source)
        self.write(stage+"/manifest.json",dict(sha256=policy.sha256,
            parent=parent.sha256 if parent else None,api_version=policy.api_version,model_version=self.model_version))
        self.evaluate(stage+"/simulation.json")

    def predict(self, probe, **prospective):
        if "execution_budget" not in prospective:
            capabilities=getattr(self,"capabilities",{})
            published=dict(DEFAULT_BUDGET,**capabilities.get("experiment_budget",{}))
            duration=min(published["max_duration_s"],max(.001,self.cfg.max_probe_cost-self.spent))
            prospective["execution_budget"]=dict(max_rpc_calls=min(192,published["max_rpc_calls"]),
                max_distinct_settings=published["max_distinct_settings"],
                max_duration_s=max(.001,duration-(capabilities.get("last_setup_cost_s") or 0.)))
        self.prediction_budgets[probe.policy.sha256]=prospective["execution_budget"]
        response = self.checked_version(self.call(self.simulator,"sim.predict_probe",dict(
            **self.sim_params(),probe=probe.wire(),**prospective)),"sim.predict_probe")
        if response.get("feature_schema") != self.feature_schema or response.get("policy_sha256") != probe.policy.sha256:
            raise ValueError("Measured probe prediction contract mismatch")
        models = normalize([Hypothesis(**m) for m in prospective["models"]]) if "models" in prospective else self.models
        validate_predictions(response["predictions"],models,self.cfg.feature_scales)
        for item in response["predictions"].values():
            if finite(item["duration_s"],"predicted physical cost")<0:raise ValueError("Negative predicted physical cost")
        return response["predictions"]

    def update_model(self):
        """Replay measured fit data, then commit physics and learned API memory."""
        if not self.sim_configured:
            self.simulator_info=self.call(self.simulator,"sim.describe",{})
            if self.simulator_info.get("api_version")!=IDENTIFICATION_API_VERSION or self.simulator_info.get("feature_schema")!=FEATURE_SCHEMA:
                raise ValueError("Predictive simulator must implement measured API v2")
            if self.feature_schema not in self.simulator_info.get("supported_feature_schemas",[FEATURE_SCHEMA]):
                raise ValueError("Predictive simulator does not support this World observation contract")
            self.adapter_info=self.simulator_info["adapters"][self.task.id]
        fit_ids = [d["result"]["evidence_id"] for d in self.fit_data]
        for failure in self.unresolved_failures():
            if not set(fit_ids)-set(failure["fit_evidence_ids"]):
                self.model_diagnostics=dict(reason="New fit evidence is required to resolve a falsified model",failure=failure)
                return False
        proposed = self.models
        if self.belief["estimates"] or self.needs_identification or not self.models:
            if self.model_revisions >= self.cfg.max_model_revisions:
                return False
            context = dict(goal=public_goal(self.task.public()),models=self.model_json(),
                world=dict(id="current",object_ids=self.scene_object_ids),
                feature_schema=self.feature_schema,
                parameter_contracts=self.adapter_info["parameter_contracts"],belief=self.belief,
                fit_evidence=[dict(evidence_id=d["result"]["evidence_id"],
                    probe_sha256=d["probe"].policy.sha256,source=d["probe"].policy.source,
                    execution_budget=d["result"].get("execution_budget"),
                    program_return=d["result"].get("program_return"),features=d["result"]["features"])
                    for d in self.fit_data])
            answer = self.call(self.physic,"physic.revise_physical_model",context)
            if not answer.get("evidence_ids") or set(answer["evidence_ids"]) != set(fit_ids):
                raise ValueError("Predictive model revision cites unavailable fit evidence")
            proposed = normalize([Hypothesis(**m) for m in answer["models"]])
            self.model_revisions += 1
            self.write(f"models/revision_{self.model_revisions:03d}.json",dict(context=context,response=answer))
        if not self.sim_configured:
            task=self.task.public(IDENTIFICATION_API_VERSION)
            scene = {}
            if self.feature_schema==SCENE_FEATURE_SCHEMA:
                scene=dict(object_ids=self.scene_object_ids)
                task=dict(task,scene=self.initial_visual)
            self.checked_version(self.call(self.simulator,"sim.configure",dict(session_id=self.session_id,
                task=task,models=[asdict(m) for m in proposed],**scene)),"sim.configure")
            self.sim_configured=True
        # Recompute from a host-owned prior. Every surviving hypothesis must
        # explain the entire fit dataset; different models cannot cover its parts.
        replay=[]
        for datum in self.fit_data:
            predictions=self.predict(datum["probe"],models=[asdict(m) for m in proposed],belief=datum["belief_before"],execution_budget=datum["result"].get("execution_budget"))
            replay.append(dict(evidence_id=datum["result"]["evidence_id"],actual=datum["result"]["features"],predictions=predictions))
        proposed,report=joint_fit(proposed,replay,self.cfg.feature_scales,self.cfg.max_prediction_error)
        self.write(f"models/fit_review_{self.model_revisions:03d}.json",report)
        if proposed is None:
            self.model_diagnostics=dict(reason="No joint physical explanation of all fit evidence",**report)
            self.needs_identification=True
            self.event("model_update_rejected",**self.model_diagnostics)
            return False
        # Retired held-out failures are never fitted or re-used as a new
        # certificate. A new model must nevertheless resolve their contradiction.
        resolving=self.unresolved_failures()
        for failure in resolving:
            datum=next(d for d in self.probe_data if d["result"]["evidence_id"]==failure["evidence_id"])
            predictions=self.predict(datum["probe"],models=[asdict(m) for m in proposed],belief=datum["belief_before"],execution_budget=datum["result"].get("execution_budget"))
            error=predictive_error(proposed,datum["result"]["features"],predictions,self.cfg)
            if error>self.cfg.max_prediction_error:
                self.model_diagnostics=dict(reason="New fit model still contradicts retired validation",evidence_id=failure["evidence_id"],prediction_error=error)
                return False
        response = self.call(self.simulator,"sim.update_model",dict(session_id=self.session_id,
            model_version=self.model_version,models=[asdict(m) for m in proposed],belief=self.belief,
            fit_evidence_ids=fit_ids))
        self.rpc.checked_version(response,session_id=self.session_id,
            method="sim.update_model",expected=self.model_version+1)
        acknowledged = normalize([Hypothesis(**m) for m in response["models"]])
        if [asdict(m) for m in acknowledged] != [asdict(m) for m in proposed]:
            raise ValueError("Simulator applied different model parameters")
        target = self.call(self.target,"target.update_identification",dict(belief=self.belief))
        if target.get("updated") is not True:
            raise ValueError("Target did not acknowledge identification memory")
        self.models,self.model_version = acknowledged,response["model_version"]
        for failure in resolving:failure["resolved_by_model_version"]=self.model_version
        self.needs_identification = False
        self.model_diagnostics = None
        self.write(f"models/version_{self.model_version:03d}.json",response)
        return True

    def author_probe(self,purpose,context,decision,*,repair=False):
        if repair and self.current_probe is not None:purpose=self.current_probe.purpose
        self.probe_attempt_count += 1
        number = self.probe_attempt_count
        name = f"active_probe_{number}"
        self.recorder.scope(name,exclusive=True)
        context = dict(context,purpose=purpose,design_review=decision,
            authoring_mode="repair_probe" if repair else "new_experiment",
            used_action_sources=[d["probe"].policy.source for d in self.fit_data],
            validation_requirement="New action/setting, distinct from fitted and previous validation sources; return supported rest")
        if repair:
            if self.current_probe is None:raise ValueError("There is no Probe source to repair")
            context["previous_probe_source"]=self.current_probe.policy.source
        self.write(name+"/context.json",context)
        self.write(name+"/schema.json",DESIGN_SCHEMA)
        self.recorder.write_text(name+"/prompt.txt",probe_prompt(context))
        answer = self.call(self.physic,"physic.design_probe",context)
        policy = validate_probe_design(answer,context)
        cost = max(finite(decision["estimated_probe_cost_s"]),.001)
        probe = Probe(name,policy,cost,tuple(answer["selected_factors"]),
            ";".join(answer["experimental_design"]["manipulated_variables"]),purpose,answer.get("validation"),tuple(answer["estimator_outputs"]))
        self.write(name+"/analysis.json",{k:v for k,v in answer.items() if k!="code"})
        self.write(name+"/response.json",answer)
        self.recorder.write_text(name+"/probe.py",policy.source)
        if repair:self.probe_repair_count+=1
        else:
            self.probe_design_count+=1
            self.probe_experiment_id=uuid.uuid4().hex
        self.write(name+"/manifest.json",dict(status="authored",sha256=policy.sha256,
            api_version=policy.api_version,purpose=purpose,model_version=self.model_version,
            change_kind="repair" if repair else "new_design",physical_executions=0,
            experiment_id=self.probe_experiment_id))
        self.current_probe=probe
        if purpose=="validation":
            self.validation_checkpoints.pop(policy.sha256,None)
            self.validation_segments.pop(policy.sha256,None)
            self.validation_traces.pop(policy.sha256,None)
        return probe

    def computed_probe_value(self,probe,predictions,candidates):
        cost=max(predictions[m.id]["duration_s"] for m in self.models if m.weight>0)
        result=probe_value(replace(probe,cost=max(.001,cost)),predictions,self.models,candidates,self.rows,self.cfg)
        return dict(result,cost_basis="predicted_measured_setup_and_execution",predicted_cost_s=cost)

    def select_probe(self,purpose,context,decision,candidates):
        count=self.cfg.probe_candidates if purpose=="fit" and self.models and candidates else 1
        proposals=[];seen=set()
        for index in range(count):
            supplied=dict(context,executable_candidate_index=index,
                competing_probe_sources=[p.policy.source for p,_,_ in proposals],
                comparison_requirement="Propose a distinct executable discriminating intervention, not a textual alternative")
            probe=self.author_probe(purpose,supplied,decision)
            if probe.policy.behavior_sha256 in seen:continue
            seen.add(probe.policy.behavior_sha256)
            predictions=self.predict(probe,belief=self.probe_memory(probe)) if self.sim_configured and self.models else None
            if predictions is not None:self.probe_predictions[probe.policy.sha256]=predictions
            value=(self.computed_probe_value(probe,predictions,candidates)
                if predictions is not None and candidates and self.rows else None)
            proposals.append((probe,predictions,value))
        eligible=[item for item in proposals if item[1] is None or
            max(p["risk"] for key,p in item[1].items() if next(m for m in self.models if m.id==key).weight>0)<=self.cfg.max_risk]
        chosen=max(eligible or proposals,key=lambda item:item[2]["net_value"] if item[2] else -float("inf"))[0]
        self.write(f"probe_comparisons/design_{self.probe_attempt_count:03d}.json",dict(model_version=self.model_version,
            selected_sha256=chosen.policy.sha256,candidates=[dict(sha256=p.policy.sha256,predictions=pred,value=value)
                for p,pred,value in proposals],heuristic_claim=decision.get("value_basis")))
        self.probe_experiment_id=self.recorder.read_json(chosen.id+"/manifest.json")["experiment_id"]
        return chosen

    def probe(self, purpose, context, decision, candidates, *, mode="probe"):
        if mode=="repair_probe":
            self.author_probe(purpose,context,decision,repair=True)
            return "repaired"
        if mode in ("execute_probe","resume_probe"):
            if self.current_probe is None:raise ValueError("There is no frozen Probe to execute")
            probe=replace(self.current_probe,cost=finite(decision["estimated_probe_cost_s"]))
            if mode=="resume_probe" and not resumable(self.probe_memory(probe),probe.policy.sha256):
                raise ValueError("Resume requires a safe checkpoint for the same frozen source")
            if mode=="resume_probe" and probe.purpose=="validation" and self.validation_checkpoints[probe.policy.sha256]["model_version"]!=self.model_version:
                raise ValueError("Validation continuation is stale after a model update")
            if mode=="execute_probe" and resumable(self.probe_memory(probe),probe.policy.sha256):
                raise ValueError("Use resume_probe for unfinished same-source progress")
            purpose=probe.purpose
        else:probe=self.select_probe(purpose,context,decision,candidates)
        self.current_probe=probe
        policy=probe.policy;name=probe.id;cost=probe.cost
        if source_policy(self.recorder.read_text(name+"/probe.py")).sha256!=policy.sha256:
            raise ValueError("Probe source changed after authoring")
        attempt=self.execution_attempts.get(name,0)+1
        destination="world_execution" if attempt==1 else f"world_execution/attempt_{attempt:03d}"
        self.write(name+"/"+destination+"/authorization.json",decision)
        used = {d["probe"].policy.behavior_sha256 for d in self.fit_data} | {
            v.get("behavior_sha256") for v in self.validation_data}
        if purpose == "validation" and policy.behavior_sha256 in used:
            self.write(name+"/rejection.json",dict(reason="Validation must use a fresh action source"))
            return "rejected"
        if purpose=="validation" and probe.validation["candidate_sha256"]!=self.assessment["selected"]:
            return "rejected"
        execution_belief=self.probe_memory(probe)
        predictions=self.probe_predictions.pop(policy.sha256,None)
        if predictions is None and self.sim_configured and self.models:
            predictions=self.predict(probe,belief=execution_belief)
        value=(self.computed_probe_value(probe,predictions,candidates)
            if predictions is not None and candidates and self.rows else None)
        # A new policy structure lies outside the finite candidate VOI estimate.
        # Keep the scientist's estimate explicitly separate from computed VOI.
        estimated_net = finite(decision["expected_decision_benefit"])-self.cfg.probe_cost_weight*cost
        prediction_file="prediction.json" if attempt==1 else f"prediction_{attempt:03d}.json"
        self.write(name+"/"+prediction_file,dict(predictions=predictions,computed_value=value,
            scientist_estimated_net_value=estimated_net,
            prediction_deferred_until_physical_evidence=predictions is None))
        risk=(value["predicted_risk"] if value is not None else
            max(predictions[m.id]["risk"] for m in self.models if m.weight>0) if predictions is not None else None)
        if risk is not None and risk > self.cfg.max_risk:
            return "rejected"
        if predictions is not None:cost=max(cost,max(p["duration_s"] for p in predictions.values()))
        if self.spent+cost > self.cfg.max_probe_cost:
            return "cost_budget_exhausted"
        heuristic=False
        if purpose=="fit" and (value is None or value["net_value"]<=self.cfg.min_net_value):
            basis=decision.get("value_basis")
            allowed=(basis=="bootstrap" and not self.models) or basis=="new_structure"
            if not allowed:return "low_value"
            if self.heuristic_count>=self.cfg.max_heuristic_probes or self.heuristic_spent+cost>self.cfg.max_heuristic_cost:
                return "heuristic_budget_exhausted"
            heuristic=True
        self.write(name+f"/value_authorization_{attempt:03d}.json",dict(heuristic=heuristic,
            computed_net_value=value["net_value"] if value else None,agent_estimate_used_as_computed_value=False))
        self.call(self.target,"target.update_identification",dict(belief=execution_belief))
        remaining=self.cfg.max_probe_cost-self.spent
        if heuristic:remaining=min(remaining,self.cfg.max_heuristic_cost-self.heuristic_spent)
        result = self.call(self.target,"target.execute_probe",dict(policy=policy.wire(),
            budget=dict(max_duration_s=min(DEFAULT_BUDGET["max_duration_s"],remaining))))
        self.execution_attempts[name]=attempt
        self.write(name+"/"+destination+"/result.json",result)
        self.probe_count += 1
        if result.get("feature_schema") != self.feature_schema or result.get("policy_sha256") != policy.sha256:
            raise ValueError("Measured target response contract mismatch")
        for sample in result["observations"]:
            self.check_scene(sample["observation"])
        evidence = result.get("evidence_id")
        if not evidence or evidence in self.evidence_ids:
            raise ValueError("Repeated/missing measured evidence ID")
        self.evidence_ids.add(evidence)
        self.probe_data.append(dict(probe=probe,result=result,belief_before=self.belief))
        manifest=self.recorder.read_json(name+"/manifest.json")
        self.write(name+"/manifest.json",dict(manifest,status="executed_in_target",execution_status=result["status"],
            sha256=policy.sha256,api_version=policy.api_version,purpose=purpose,
            model_version=self.model_version,evidence_id=evidence,physical_executions=attempt))
        setup_cost = result.get("setup_cost_s")
        cost_actual = finite(result["cost_s"])+(finite(setup_cost) if setup_cost is not None else 0.)
        if cost_actual < 0:
            raise ValueError("Negative measured probe cost")
        self.spent += cost_actual
        if heuristic:
            self.heuristic_count+=1
            self.heuristic_spent+=cost_actual
        actual = response_features(result["observations"])
        if actual != result["features"]:
            raise ValueError("Measured features differ from archived sensor trajectory")
        measured_budget=result.get("execution_budget")
        if purpose=="validation" and measured_budget is not None and measured_budget!=self.prediction_budgets.get(policy.sha256):
            self.write(name+f"/prediction_input_alignment_{attempt:03d}.json",dict(
                reason="Condition on actual pre-program budget inputs; models/weights unchanged",
                prospective_predictions=predictions,execution_budget=measured_budget))
            predictions=self.predict(probe,belief=execution_belief,execution_budget=measured_budget)
        returned = result.get("program_return")
        if purpose=="validation":
            checkpoint=result.get("experiment_checkpoint")
            if checkpoint is None:self.validation_checkpoints.pop(policy.sha256,None)
            else:self.validation_checkpoints[policy.sha256]=dict(checkpoint,model_version=self.model_version)
        belief_before=self.belief
        pending_belief = None
        if purpose=="fit":
            structured=returned if isinstance(returned,dict) and isinstance(returned.get("estimates"),list) else dict(
                status="program_error",reason=result.get("error"),estimates=[])
            try:
                pending_belief=merge_identification(self.belief,structured,source_probe=name,evidence_id=evidence,
                    checkpoint=result.get("experiment_checkpoint"),program_sha256=policy.sha256,contracts=probe.estimator_outputs)
            except ValueError as exc:
                pending_belief=self.belief
                result.update(status="program_error",error=f"Invalid identification output: {exc}")
                self.write(name+"/"+destination+"/result.json",result)
            # Save valid numerical evidence even when subsequent recovery/cost
            # checks prevent continuing or committing a predictive model.
            self.write(name+"/"+("belief.json" if attempt==1 else f"belief_{attempt:03d}.json"),pending_belief)
            self.belief=pending_belief
        if setup_cost is None:
            self.cost_complete=False
            return "cost_unknown"
        if result["status"]=="budget_exhausted":return "heuristic_budget_exhausted" if heuristic else "cost_budget_exhausted"
        if result["status"]=="program_error":return "program_error"
        if (result["status"] != "completed" or isinstance(returned,dict) and returned.get("status")=="recovery_failed"
                or not supported(result["observations"][-1]["observation"])):
            return "recovery_failed"
        if self.spent > self.cfg.max_probe_cost:
            return "cost_budget_exhausted"
        if purpose == "validation":
            previous_traces=self.validation_traces.setdefault(policy.sha256,[])
            coverage=check_executed(probe.validation,result["observations"],earlier=previous_traces)
            previous_traces.append(intervention_trace(result["observations"]))
            self.write(name+f"/coverage_{attempt:03d}.json",coverage)
            error = predictive_error(self.models,actual,predictions,self.cfg)
            segments=self.validation_segments.setdefault(policy.sha256,[])
            segments.append(dict(evidence_id=evidence,model_version=self.model_version,prediction_error=error))
            if error>min(self.cfg.max_prediction_error,probe.validation["max_prediction_error"]):
                self.validation_failures.append(dict(evidence_id=evidence,model_version=self.model_version,
                    prediction_error=error,fit_evidence_ids=[d["result"]["evidence_id"] for d in self.fit_data]))
                self.needs_identification=True
            if isinstance(returned,dict) and returned.get("status") in PAUSED_STATUSES:
                self.write(name+f"/validation_segment_{attempt:03d}.json",segments[-1])
                if error>self.cfg.max_prediction_error:self.needs_identification=True
                return "paused" if resumable(self.probe_memory(probe),policy.sha256) else "program_error"
            if not coverage["passed"]:
                self.validation_data.append(dict(evidence_id=evidence,model_version=self.model_version,
                    candidate_sha256=probe.validation["candidate_sha256"],passed=False,reason="intervention_coverage_failed",coverage=coverage))
                return "invalid_validation"
            fingerprint=coverage["trace"]["action_fingerprint"]
            if any(intervention_trace(d["result"]["observations"])["action_fingerprint"]==fingerprint for d in self.fit_data):
                return "invalid_validation"
            passed=all(s["model_version"]==self.model_version and s["prediction_error"]<=self.cfg.max_prediction_error for s in segments)
            self.validation_data.append(dict(evidence_id=evidence,behavior_sha256=policy.behavior_sha256,
                model_version=self.model_version,prediction_error=max(s["prediction_error"] for s in segments),passed=passed,
                candidate_sha256=probe.validation["candidate_sha256"],coverage=coverage,
                segment_evidence_ids=[s["evidence_id"] for s in segments],
                observed_features=actual,predictions=predictions))
            self.write(name+"/validation.json",self.validation_data[-1])
            self.needs_identification = self.needs_identification or not passed
            return "validated" if not self.needs_identification else "prediction_mismatch"
        if pending_belief is None:
            raise ValueError("Identification must return structured physical estimates")
        if not isinstance(returned,dict) or not isinstance(returned.get("estimates"),list):return "program_error"
        self.fit_data.append(dict(probe=probe,result=result,belief_before=belief_before))
        self.belief=pending_belief
        if returned.get("status") in PAUSED_STATUSES:
            if not resumable(self.belief,policy.sha256):return "probe_execution_error"
            self.call(self.target,"target.update_identification",dict(belief=self.belief))
            return "paused"
        if not self.belief["estimates"] and not self.models:
            self.needs_identification=True
            return "needs_more_evidence"
        return "updated" if self.update_model() else "model_inadequate"

    def finish(self,status,reason):
        result = dict(status=status,reason=reason,model_version=self.model_version,models=self.model_json(),
            world_count=1,object_ids=self.scene_object_ids,feature_schema=self.feature_schema,
            probe_count=self.probe_count,probe_cost=self.spent,probe_cost_complete=self.cost_complete,assessment=self.assessment,
            probe_program_versions=self.probe_attempt_count,probe_designs=self.probe_design_count,
            probe_repairs=self.probe_repair_count,
            final_policy=None,target_task_executed=False)
        if status == "task_sufficient":
            verify_snapshot(self.run_manifest)
            current_agents=dict(coding=self.call(self.agent,"agent.describe",{}),physic=self.call(self.physic,"physic.describe",{}))
            if current_agents!=self.run_manifest["agents"]:raise ValueError("Agent service or inference configuration changed during the run")
            policy = self.store.policies[self.assessment["selected"]]
            self.store.freeze(policy,self.output/"final",dict(model_version=self.model_version,
                assessment=self.assessment,validation=[v for v in self.validation_data if v["model_version"]==self.model_version]),
                recorder=self.recorder.scope("final",exclusive=True),execution=dict(
                    identification=self.belief,task=self.task.public(IDENTIFICATION_API_VERSION),
                    models=self.model_json(),model_version=self.model_version,world_identity=self.world_identity,
                    config=asdict(self.cfg),run_manifest=self.run_manifest,runtime_contract=self.capabilities["runtime_contract"]))
            result["final_policy"] = dict(path=str(self.output/"final/policy.py"),sha256=policy.sha256)
        self.run_manifest["artifacts"]=dict(policies=list(self.store.policies),
            probes=[dict(sha256=d["probe"].policy.sha256,evidence_id=d["result"]["evidence_id"],
                response_sha256=content_hash(d["result"])) for d in self.probe_data],
            model_version=self.model_version,identification_sha256=content_hash(self.belief))
        self.write("run_manifest.json",self.run_manifest)
        self.write("result.json",result)
        self.write("progress.json",self.history)
        self.event("loop_stopped",**result)
        return result

    def run(self):
        try:
            return self._run()
        except Exception as exc:
            self.finish("infrastructure_error",f"{type(exc).__name__}: {exc}")
            raise

    def _run(self):
        self.run_manifest["agents"]=dict(coding=self.call(self.agent,"agent.describe",{}),physic=self.call(self.physic,"physic.describe",{}))
        self.run_manifest["config_sha256"]=content_hash(dict(task=self.task.public(IDENTIFICATION_API_VERSION),loop=asdict(self.cfg)))
        self.write("run_manifest.json",self.run_manifest)
        self.capabilities = self.call(self.target,"target.describe",{})
        for descriptor in (self.capabilities,):
            if descriptor.get("api_version") != IDENTIFICATION_API_VERSION or descriptor.get("feature_schema") != FEATURE_SCHEMA:
                raise ValueError("Unified loop requires measured robot/predictive API v2")
        self.simulator_info=self.call(self.simulator,"sim.describe",{})
        self.call(self.simulator,"sim.validate_task",dict(task=self.task.public(IDENTIFICATION_API_VERSION),models=self.model_json()))
        initial=observation(self.call(self.target,"target.observe",{}))
        self.world_identity=execution_identity(self.capabilities,initial)
        if len(self.capabilities["arm"]["joint_torque_limits_nm"])!=7 or any(finite(x)<=0 for x in self.capabilities["arm"]["joint_torque_limits_nm"]) or finite(self.capabilities["gripper"]["normal_force_limit_per_finger_n"])<=0:
            raise ValueError("Invalid robot capabilities before first interaction")
        expected_force=self.task.constraints.get("grip_force_per_finger_n")
        if expected_force is not None and self.capabilities["gripper"]["normal_force_limit_per_finger_n"]!=expected_force:
            raise ValueError("Target gripper force cap differs from the task contract")
        self.call(self.target,"target.update_identification",dict(belief=self.belief))
        if self.cfg.max_probes==0:return self.finish("probe_budget_exhausted","No physical interaction budget")
        if self.cfg.max_probe_cost==0:return self.finish("cost_budget_exhausted","No physical time budget")
        prepared=self.call(self.target,"target.prepare",dict(budget=dict(max_duration_s=min(DEFAULT_BUDGET["max_duration_s"],self.cfg.max_probe_cost))))
        observed=observation(prepared)
        self.scene_object_ids=object_ids(observed)
        if self.scene_object_ids!=self.world_identity["object_ids"]:raise ValueError("Scene changed during preparation")
        setup=self.call(self.target,"target.describe",{}).get("last_setup_cost_s")
        if setup is None:
            self.cost_complete=False
            return self.finish("cost_unknown","Initial World preparation did not report actual cost")
        self.spent+=finite(setup,"initial setup cost")
        if prepared.get("preparation",{}).get("status")=="budget_exhausted":return self.finish("cost_budget_exhausted","Initial preparation could not finish within its hard budget")
        if self.spent>self.cfg.max_probe_cost:return self.finish("cost_budget_exhausted","Initial preparation exhausted the budget")
        self.feature_schema=response_schema(observed)
        if self.feature_schema==SCENE_FEATURE_SCHEMA:
            scales=scene_scales(self.cfg.feature_scales,len(self.scene_object_ids))
            self.cfg=replace(self.cfg,feature_scales=scales)
            self.belief.update(object_ids=self.scene_object_ids,
                objects={key:dict(estimates=[]) for key in self.scene_object_ids})
        self.initial_visual=visual_observation(observed)
        self.write("task_context/world.json",dict(world_count=1,object_ids=self.scene_object_ids,
            feature_schema=self.feature_schema,feature_scales=list(self.cfg.feature_scales)))
        self.write("task_context/initial_visual.json",self.initial_visual)
        for index in range(self.cfg.max_rounds):
            context = self.context()
            decision = self.call(self.physic,"physic.assess_task",context)
            validate_plan(decision,context)
            self.write(f"rounds/{index:03d}/context.json",context)
            self.write(f"rounds/{index:03d}/decision.json",decision)
            action = decision["action"]
            entry = dict(round=index,action=action,model_version=self.model_version,probe_cost=self.spent,
                assessment=self.assessment)
            if action!="stop":self.history.append(entry)
            handlers={"stop":self.handle_stop,"revise_policy":self.handle_policy,
                **{key:self.handle_probe for key in ("probe","validate","repair_probe","execute_probe","resume_probe")}}
            if action not in handlers:raise ValueError("Unknown physical loop action")
            candidates=([self.store.policies[row["policy_sha256"]] for row in self.assessment["scores"]]
                if self.assessment else [])
            result=handlers[action](context,decision,candidates,entry)
            if result is not None:return result
        return self.finish("round_budget_exhausted","Automatic revision budget reached before sufficiency")

    def unresolved_failures(self):
        return [failure for failure in self.validation_failures if not failure.get("resolved_by_model_version")]

    def handle_stop(self, context, decision, candidates, entry):
        valid = [v for v in self.validation_data if v["model_version"]==self.model_version and v["passed"]
            and self.assessment and v.get("candidate_sha256")==self.assessment["selected"]]
        robust=bool(self.assessment and self.assessment["robust"])
        net=self.decision_value(decision)
        if robust and len(valid)>=self.cfg.min_validation_probes and not self.needs_identification and not self.unresolved_failures() and net<=self.cfg.min_net_value and self.assessment["decision_regret"]<=self.cfg.min_net_value:
            return self.finish("task_sufficient","Robust validated operation with low regret over the evaluated finite candidate set; no global exploration optimality claim")
        return self.finish("exploration_stopped","Stop cannot certify a candidate without current independent evidence")

    def handle_policy(self, context, decision, candidates, entry):
        if not decision["instance_assessments"][0]["sufficient"]:
            return self.finish("insufficient_decision_evidence","Policy revision lacks sufficient instance evidence")
        if not self.models or self.model_version==0 or self.needs_identification:
            if not self.fit_data:return self.finish("needs_physical_evidence","No grounded model evidence is available")
            if not self.update_model():
                entry["outcome"]="model_inadequate"
                if self.model_revisions>=self.cfg.max_model_revisions:
                    return self.finish("model_inadequate","Fit evidence does not support model commit within revision budget")
                return None
        self.revise_policy(context)
        return None

    def decision_value(self, decision):
        return finite(decision["expected_decision_benefit"])-self.cfg.probe_cost_weight*finite(decision["estimated_probe_cost_s"])

    def handle_probe(self, context, decision, candidates, entry):
        action=decision["action"]
        if action=="validate" and not (self.assessment and self.assessment["robust"]):
            return self.finish("invalid_decision","Independent predictive validation needs a robust candidate")
        if action!="repair_probe" and self.probe_count>=self.cfg.max_probes:
            return self.finish("probe_budget_exhausted","Physical interaction budget reached before sufficiency")
        outcome=self.probe("validation" if action=="validate" else "fit",context,decision,candidates,mode=action)
        entry["outcome"]=outcome
        if outcome=="model_inadequate" and self.model_revisions>=self.cfg.max_model_revisions:
            return self.finish("model_inadequate","Physical model revision budget exhausted; see joint fit diagnostics")
        if outcome in ("cost_budget_exhausted","probe_execution_error","cost_unknown","heuristic_budget_exhausted"):
            return self.finish(outcome,"See the frozen probe, predictions and measured evidence")
        if outcome=="updated" and self.store.policies:
            # First re-predict the same candidate under the updated physics.
            self.evaluate(f"models/version_{self.model_version:03d}_prediction.json")
        return None

    def check_scene(self, observed):
        check_objects(observed,self.scene_object_ids)
        if response_schema(observed)!=self.feature_schema:
            raise ValueError("World observation contract changed during the loop")
