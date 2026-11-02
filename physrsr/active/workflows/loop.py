"""Probe-first v2 loop: task/vision, scientist, measurement, model and policy.

All environment/model/agent interactions cross the supplied APIs. The trusted
simulator owns task adapters; no task name or physical factor is dispatched here.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import uuid
import json

from ..core.artifacts import save_json
from ..core.contracts import IDENTIFICATION_API_VERSION, Hypothesis, normalize, finite
from ..core.identification import ROBOT_API, DEFAULT_BUDGET, public_goal
from ..core.exploration import CONTRACT,PHYSICAL_ACTIONS,PAUSED_STATUSES,resumable,validate_plan
from ..core.measurements import FEATURE_SCHEMA, FEATURE_NAMES, FEATURE_SCALES, response_features
from ..core.policy import Policy, PolicyStore
from ..core.reasoning import Probe, assess, posterior, predictive_error, feature_error, probe_value, validate_predictions
from ..agents.identification import source_policy, task_candidate, design_prompt
from ..runtime.instrumented import observation


class ActiveLoop:
    def __init__(self, task, config, models, *, agent, physic, simulator, target, output):
        scales = config.feature_scales if len(config.feature_scales)==len(FEATURE_NAMES) else FEATURE_SCALES
        self.task, self.cfg, self.models = task, replace(config,feature_scales=scales), normalize(models) if models else ()
        self.agent, self.physic, self.simulator, self.target = agent, physic, simulator, target
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.store = PolicyStore(self.output / "policies")
        self.session_id = uuid.uuid4().hex
        self.model_version = self.probe_count = self.revision_count = self.model_revisions = 0
        self.spent = 0.
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
        self.belief = dict(estimates=[],instance_id="current",evidence_source="no_interaction_yet")
        self.evidence_ids = set()
        self.assessment, self.rows = None, []
        self.needs_identification = False
        self.sim_configured = False
        self.write("config.json",dict(task=task.public(IDENTIFICATION_API_VERSION),loop=asdict(self.cfg),
            models=self.model_json(),session_id=self.session_id))

    def write(self, name, value):
        save_json(self.output / name,value)

    def event(self, kind, **value):
        import json
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event=kind,**value),ensure_ascii=False,allow_nan=False)+"\n")

    def call(self, endpoint, method, params):
        self.event("api_request",method=method,params=params)
        result = endpoint.call(method,params)
        self.event("api_response",method=method,result=result)
        return result

    def model_json(self):
        return [asdict(m) for m in self.models]

    def sim_params(self):
        return dict(session_id=self.session_id,model_version=self.model_version,seeds=list(self.cfg.model_seeds))

    def checked_version(self, response):
        if response.get("model_version") != self.model_version:
            raise ValueError("Simulator returned a stale model version")
        return response

    def evaluate(self, destination):
        candidates = list(self.store.policies.values())[-self.cfg.max_candidates:]
        if self.assessment and all(p.sha256 != self.assessment["selected"] for p in candidates):
            candidates = [self.store.policies[self.assessment["selected"]],*candidates[:self.cfg.max_candidates-1]]
        response = self.checked_version(self.call(self.simulator,"sim.evaluate",dict(**self.sim_params(),
            policies=[p.wire() for p in candidates])))
        self.rows = response["rows"]
        self.assessment = assess(self.models,candidates,self.rows,self.cfg)
        self.write(destination,dict(model_version=self.model_version,assessment=self.assessment,rows=self.rows))
        return candidates

    @staticmethod
    def evidence_samples(samples,limit=10):
        """Keep endpoints, elapsed phases and load extrema; archive stays complete."""
        if len(samples)<=limit:
            return samples
        indices={0,len(samples)-1,*[i*(len(samples)-1)//(limit-3) for i in range(limit-2)]}
        for channel in ("force_n","torque_nm"):
            indices.add(max(range(len(samples)),key=lambda i:sum(
                x*x for x in samples[i]["observation"]["wrench"][channel])))
        return [samples[i] for i in sorted(indices)]

    def prediction_feedback(self):
        scores=[{k:v for k,v in row.items() if k!="feedback"} for row in self.rows]
        trajectories=[]
        for row in sorted(self.rows,key=lambda r:(r["success_rate"],r["margin"]))[:4]:
            for outcome in row.get("feedback",[])[:2]:
                trajectories.append(dict(model_id=row["model_id"],policy_sha256=row["policy_sha256"],
                    status=outcome.get("status"),reason=outcome.get("reason"),
                    observations=self.evidence_samples(outcome.get("observations",[]))))
        return dict(model_policy_scores=scores,failure_trajectories=trajectories,
            sampling="Selected API boundaries and load extrema; complete responses remain in simulation artifacts")

    def context(self):
        """Scientist gets a semantic goal and measured/predictive evidence, no task ID."""
        self.capabilities=self.call(self.target,"target.describe",{})
        current=observation(self.call(self.target,"target.observe",{}))
        if not self.probe_data:
            current={key:current[key] for key in ("time_s","coordinate_frame","object","tcp")}
        evidence={"task_goal","robot_capabilities","initial_visual"}
        evidence.update(d["result"]["evidence_id"] for d in self.probe_data)
        evidence.update(v["evidence_id"] for v in self.validation_data)
        if self.rows:evidence.add(f"prediction:v{self.model_version}")
        context=dict(goal=public_goal(self.task.public(IDENTIFICATION_API_VERSION)),robot_api=ROBOT_API,
            decision_contract=CONTRACT,predictive_validation=True,
            instances=[dict(episode_id="current",retained_identification=self.belief,
                available_evidence_ids=sorted(evidence),
                latest_execution=(dict(status=self.probe_data[-1]["result"]["status"],
                    error=self.probe_data[-1]["result"].get("error"),
                    program_return=self.probe_data[-1]["result"].get("program_return"),
                    observations=self.evidence_samples(self.probe_data[-1]["result"]["observations"],limit=18))
                    if self.probe_data else None))],
            capabilities=self.capabilities,current_observation=current,
            phase="grounded_revision" if self.sim_configured else "bootstrap_identification",
            initial_evidence=([dict(evidence_id=f"prediction:v{self.model_version}",source="predictive_model",
                **self.prediction_feedback())] if self.rows else [dict(evidence_id="initial_visual",
                source="initial_visual_observation",observation=self.initial_visual)]),
            assessment=self.assessment,model_version=self.model_version,
            prior_identification_evidence=[dict(evidence_id=d["result"]["evidence_id"],
                program_return=d["result"].get("program_return"),
                observations=self.evidence_samples(d["result"]["observations"]))
                for d in self.probe_data],
            validation_reports=self.validation_data,identification=self.belief,
            pending_validation_segments=self.validation_segments,
            prediction_mismatch_requires_new_fit=self.needs_identification,
            experiment_budget=dict(DEFAULT_BUDGET,remaining_interactions=self.cfg.max_probes-self.probe_count,
                remaining_cost_s=max(0.,self.cfg.max_probe_cost-self.spent)))
        if self.current_probe:
            context["current_probe"]=dict(source=self.current_probe.policy.source,sha256=self.current_probe.policy.sha256,
                purpose=self.current_probe.purpose,
                resumable=resumable(self.probe_memory(self.current_probe),self.current_probe.policy.sha256))
        if self.assessment:context["current_task_policy"]=self.store.policies[self.assessment["selected"]].source
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
        folder = self.output / stage
        folder.mkdir()
        payload = dict(context,previous_task_candidate=parent.source if parent else None,
            prediction_feedback=self.prediction_feedback(),execution_protocol=dict(one_shared_policy=True,
                current_instance_estimates_from_api=True,target_task_executed=False))
        self.write(stage+"/context.json",payload)
        from ..core.paths import PROMPTS_ROOT
        (folder/"prompt.txt").write_text((PROMPTS_ROOT/"identified_policy.txt").read_text().format(
            context=json.dumps(payload,ensure_ascii=False,allow_nan=False)))
        answer = self.call(self.agent,"agent.synthesize_identified_policy",payload)
        self.write(stage+"/response.json",answer)
        checked = task_candidate(answer["code"])
        policy = Policy(checked.source,answer["rationale"],parent.sha256 if parent else None,IDENTIFICATION_API_VERSION)
        self.store.add(policy,self.revision_count)
        (folder/"policy.py").write_bytes(policy.source.encode())
        self.write(stage+"/manifest.json",dict(sha256=policy.sha256,
            parent=parent.sha256 if parent else None,api_version=policy.api_version,model_version=self.model_version))
        self.evaluate(stage+"/simulation.json")

    def predict(self, probe, **prospective):
        response = self.checked_version(self.call(self.simulator,"sim.predict_probe",dict(
            **self.sim_params(),probe=probe.wire(),**prospective)))
        if response.get("feature_schema") != FEATURE_SCHEMA or response.get("policy_sha256") != probe.policy.sha256:
            raise ValueError("Measured probe prediction contract mismatch")
        models = normalize([Hypothesis(**m) for m in prospective["models"]]) if "models" in prospective else self.models
        validate_predictions(response["predictions"],models,self.cfg.feature_scales)
        return response["predictions"]

    def update_model(self):
        """Replay measured fit data, then commit physics and learned API memory."""
        if not self.sim_configured:
            self.simulator_info=self.call(self.simulator,"sim.describe",{})
            if self.simulator_info.get("api_version")!=IDENTIFICATION_API_VERSION or self.simulator_info.get("feature_schema")!=FEATURE_SCHEMA:
                raise ValueError("Predictive simulator must implement measured API v2")
            self.adapter_info=self.simulator_info["adapters"][self.task.id]
        fit_ids = [d["result"]["evidence_id"] for d in self.fit_data]
        proposed = self.models
        if self.belief["estimates"] or self.needs_identification or not self.models:
            if self.model_revisions >= self.cfg.max_model_revisions:
                return False
            context = dict(goal=public_goal(self.task.public()),models=self.model_json(),
                parameter_contracts=self.adapter_info["parameter_contracts"],belief=self.belief,
                fit_evidence=[dict(evidence_id=d["result"]["evidence_id"],
                    program_return=d["result"].get("program_return"),features=d["result"]["features"])
                    for d in self.fit_data])
            answer = self.call(self.physic,"physic.revise_predictive_model",context)
            if not answer.get("evidence_ids") or set(answer["evidence_ids"]) != set(fit_ids):
                raise ValueError("Predictive model revision cites unavailable fit evidence")
            proposed = normalize([Hypothesis(**m) for m in answer["models"]])
            self.model_revisions += 1
            self.write(f"models/revision_{self.model_revisions:03d}.json",dict(context=context,response=answer))
        if not self.sim_configured:
            self.checked_version(self.call(self.simulator,"sim.configure",dict(session_id=self.session_id,
                task=self.task.public(IDENTIFICATION_API_VERSION),models=[asdict(m) for m in proposed])))
            self.sim_configured=True
        # Validation measurements are deliberately absent from this fitting path.
        for datum in self.fit_data:
            predictions = self.predict(datum["probe"],models=[asdict(m) for m in proposed],belief=datum["belief_before"])
            actual = datum["result"]["features"]
            error = min(feature_error(actual,p["features"],self.cfg.feature_scales) for p in predictions.values())**.5
            if error > self.cfg.max_prediction_error:
                self.event("model_update_rejected",prediction_error=error,evidence_id=datum["result"]["evidence_id"])
                return False
            proposed = posterior(proposed,actual,predictions,self.cfg.feature_scales)
        response = self.call(self.simulator,"sim.update_model",dict(session_id=self.session_id,
            model_version=self.model_version,models=[asdict(m) for m in proposed],belief=self.belief,
            fit_evidence_ids=fit_ids))
        if response["model_version"] != self.model_version+1:
            raise ValueError("Predictive model commit did not advance its version")
        acknowledged = normalize([Hypothesis(**m) for m in response["models"]])
        if [asdict(m) for m in acknowledged] != [asdict(m) for m in proposed]:
            raise ValueError("Simulator applied different model parameters")
        target = self.call(self.target,"target.update_identification",dict(belief=self.belief))
        if target.get("updated") is not True:
            raise ValueError("Target did not acknowledge identification memory")
        self.models,self.model_version = acknowledged,response["model_version"]
        self.needs_identification = False
        self.write(f"models/version_{self.model_version:03d}.json",response)
        return True

    def author_probe(self,purpose,context,decision,*,repair=False):
        if repair and self.current_probe is not None:purpose=self.current_probe.purpose
        self.probe_attempt_count += 1
        number = self.probe_attempt_count
        name = f"active_probe_{number}"
        folder = self.output/name
        folder.mkdir()
        context = dict(context,purpose=purpose,design_review=decision,
            authoring_mode="repair_probe" if repair else "new_experiment",
            used_action_sources=[d["probe"].policy.source for d in self.fit_data],
            validation_requirement="New action/setting, distinct from fitted and previous validation sources; return supported rest")
        if repair:
            if self.current_probe is None:raise ValueError("There is no Probe source to repair")
            context["previous_probe_source"]=self.current_probe.policy.source
        self.write(name+"/context.json",context)
        (folder/"prompt.txt").write_text(design_prompt(context))
        answer = self.call(self.physic,"physic.design_identification",context)
        policy = source_policy(answer["code"])
        cost = max(finite(decision["estimated_probe_cost_s"]),.001)
        probe = Probe(name,policy,cost,tuple(answer["selected_factors"]),
            ";".join(answer["experimental_design"]["manipulated_variables"]),purpose)
        self.write(name+"/analysis.json",{k:v for k,v in answer.items() if k!="code"})
        self.write(name+"/response.json",answer)
        (folder/"probe.py").write_bytes(policy.source.encode())
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
        return probe

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
        else:probe=self.author_probe(purpose,context,decision)
        self.current_probe=probe
        policy=probe.policy;name=probe.id;folder=self.output/name;cost=probe.cost
        if source_policy((folder/"probe.py").read_text()).sha256!=policy.sha256:
            raise ValueError("Probe source changed after authoring")
        attempt=self.execution_attempts.get(name,0)+1
        destination="world_execution" if attempt==1 else f"world_execution/attempt_{attempt:03d}"
        self.write(name+"/"+destination+"/authorization.json",decision)
        used = {d["probe"].policy.behavior_sha256 for d in self.fit_data} | {
            v["behavior_sha256"] for v in self.validation_data}
        if purpose == "validation" and policy.behavior_sha256 in used:
            self.write(name+"/rejection.json",dict(reason="Validation must use a fresh action source"))
            return "rejected"
        execution_belief=self.probe_memory(probe)
        predictions=self.predict(probe,belief=execution_belief) if self.sim_configured else None
        value=(probe_value(probe,predictions,self.models,candidates,self.rows,self.cfg)
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
        if self.spent+cost > self.cfg.max_probe_cost:
            return "cost_budget_exhausted"
        if (purpose=="fit" and value is not None and max(value["net_value"],estimated_net)<=self.cfg.min_net_value
                or purpose=="fit" and value is None and estimated_net<=self.cfg.min_net_value):
            return "low_value"
        self.call(self.target,"target.update_identification",dict(belief=execution_belief))
        result = self.call(self.target,"target.execute_probe",dict(policy=policy.wire()))
        self.execution_attempts[name]=attempt
        self.write(name+"/"+destination+"/result.json",result)
        self.probe_count += 1
        if result.get("feature_schema") != FEATURE_SCHEMA or result.get("policy_sha256") != policy.sha256:
            raise ValueError("Measured target response contract mismatch")
        evidence = result.get("evidence_id")
        if not evidence or evidence in self.evidence_ids:
            raise ValueError("Repeated/missing measured evidence ID")
        self.evidence_ids.add(evidence)
        self.probe_data.append(dict(probe=probe,result=result,belief_before=self.belief))
        manifest=json.loads((folder/"manifest.json").read_text())
        self.write(name+"/manifest.json",dict(manifest,status="executed_in_target",execution_status=result["status"],
            sha256=policy.sha256,api_version=policy.api_version,purpose=purpose,
            model_version=self.model_version,evidence_id=evidence,physical_executions=attempt))
        setup_cost = result.get("setup_cost_s")
        cost_actual = finite(result["cost_s"])+(finite(setup_cost) if setup_cost is not None else 0.)
        if cost_actual < 0:
            raise ValueError("Negative measured probe cost")
        self.spent += cost_actual
        actual = response_features(result["observations"])
        if actual != result["features"]:
            raise ValueError("Measured features differ from archived sensor trajectory")
        returned = result.get("program_return")
        if purpose=="validation":
            checkpoint=result.get("experiment_checkpoint")
            if checkpoint is None:self.validation_checkpoints.pop(policy.sha256,None)
            else:self.validation_checkpoints[policy.sha256]=dict(checkpoint,model_version=self.model_version)
        belief_before=self.belief
        pending_belief = None
        if purpose=="fit":
            from .identification import merge_identification
            structured=returned if isinstance(returned,dict) and isinstance(returned.get("estimates"),list) else dict(
                status="program_error",reason=result.get("error"),estimates=[])
            pending_belief=merge_identification(self.belief,structured,source_probe=name,evidence_id=evidence,
                checkpoint=result.get("experiment_checkpoint"),program_sha256=policy.sha256)
            # Save valid numerical evidence even when subsequent recovery/cost
            # checks prevent continuing or committing a predictive model.
            self.write(name+"/"+("belief.json" if attempt==1 else f"belief_{attempt:03d}.json"),pending_belief)
            self.belief=pending_belief
        if setup_cost is None:
            self.cost_complete=False
            return "cost_unknown"
        if result["status"]=="program_error":return "program_error"
        if (result["status"] != "completed" or isinstance(returned,dict) and returned.get("status")=="recovery_failed"
                or result["observations"][-1]["observation"]["object"]["support_clearance_m"] > .003):
            return "recovery_failed"
        if self.spent > self.cfg.max_probe_cost:
            return "cost_budget_exhausted"
        if purpose == "validation":
            error = predictive_error(self.models,actual,predictions,self.cfg)
            segments=self.validation_segments.setdefault(policy.sha256,[])
            segments.append(dict(evidence_id=evidence,model_version=self.model_version,prediction_error=error))
            if isinstance(returned,dict) and returned.get("status") in PAUSED_STATUSES:
                self.write(name+f"/validation_segment_{attempt:03d}.json",segments[-1])
                if error>self.cfg.max_prediction_error:self.needs_identification=True
                return "paused" if resumable(self.probe_memory(probe),policy.sha256) else "program_error"
            passed=all(s["model_version"]==self.model_version and s["prediction_error"]<=self.cfg.max_prediction_error for s in segments)
            self.validation_data.append(dict(evidence_id=evidence,behavior_sha256=policy.behavior_sha256,
                model_version=self.model_version,prediction_error=max(s["prediction_error"] for s in segments),passed=passed,
                segment_evidence_ids=[s["evidence_id"] for s in segments],
                observed_features=actual,predictions=predictions))
            self.write(name+"/validation.json",self.validation_data[-1])
            self.needs_identification = not passed
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
            probe_count=self.probe_count,probe_cost=self.spent,probe_cost_complete=self.cost_complete,assessment=self.assessment,
            probe_program_versions=self.probe_attempt_count,probe_designs=self.probe_design_count,
            probe_repairs=self.probe_repair_count,
            final_policy=None,target_task_executed=False)
        if status == "task_sufficient":
            policy = self.store.policies[self.assessment["selected"]]
            self.store.freeze(policy,self.output/"final",dict(model_version=self.model_version,
                assessment=self.assessment,validation=[v for v in self.validation_data if v["model_version"]==self.model_version]))
            result["final_policy"] = dict(path=str(self.output/"final/policy.py"),sha256=policy.sha256)
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
        self.capabilities = self.call(self.target,"target.describe",{})
        for descriptor in (self.capabilities,):
            if descriptor.get("api_version") != IDENTIFICATION_API_VERSION or descriptor.get("feature_schema") != FEATURE_SCHEMA:
                raise ValueError("Unified loop requires measured robot/predictive API v2")
        self.call(self.target,"target.update_identification",dict(belief=self.belief))
        observed=observation(self.call(self.target,"target.prepare",{}))
        self.initial_visual={key:observed[key] for key in ("time_s","coordinate_frame","object","tcp")}
        self.write("task_context/initial_visual.json",self.initial_visual)
        candidates=[]
        for index in range(self.cfg.max_rounds):
            context = self.context()
            decision = self.call(self.physic,"physic.assess_task",context)
            validate_plan(decision,context)
            self.write(f"rounds/{index:03d}/context.json",context)
            self.write(f"rounds/{index:03d}/decision.json",decision)
            valid = [v for v in self.validation_data if v["model_version"]==self.model_version and v["passed"]]
            action = decision["action"]
            robust=bool(self.assessment and self.assessment["robust"])
            net = finite(decision["expected_decision_benefit"])-self.cfg.probe_cost_weight*finite(decision["estimated_probe_cost_s"])
            if action=="stop" and robust and len(valid)>=self.cfg.min_validation_probes and not self.needs_identification and net<=self.cfg.min_net_value:
                return self.finish("task_sufficient","Robust candidate, fresh independent predictions and low remaining task decision value")
            if action=="stop":
                return self.finish("exploration_stopped","Stop cannot certify a candidate without current independent evidence")
            if action=="validate" and not robust:
                return self.finish("invalid_decision","Independent predictive validation needs a robust candidate")
            entry = dict(round=index,action=action,model_version=self.model_version,probe_cost=self.spent,
                assessment=self.assessment)
            self.history.append(entry)
            if action=="revise_policy":
                if not decision["instance_assessments"][0]["sufficient"]:
                    return self.finish("insufficient_decision_evidence","Policy revision lacks sufficient instance evidence")
                if not self.sim_configured:
                    if not self.fit_data:return self.finish("needs_physical_evidence","No grounded model evidence is available")
                    if not self.update_model():return self.finish("model_inadequate","Fit evidence does not support model commit")
                self.revise_policy(context)
                candidates = [self.store.policies[s["policy_sha256"]] for s in self.assessment["scores"]]
                continue
            if action not in ("probe","validate","repair_probe","execute_probe","resume_probe"):
                raise ValueError("Unknown physical loop action")
            if action!="repair_probe" and net<=self.cfg.min_net_value:
                return self.finish("low_decision_value","No positive predicted decision benefit remains after interaction cost")
            if action!="repair_probe" and self.probe_count>=self.cfg.max_probes:
                return self.finish("probe_budget_exhausted","Physical interaction budget reached before sufficiency")
            outcome = self.probe("validation" if action=="validate" else "fit",context,decision,candidates,mode=action)
            entry["outcome"]=outcome
            if outcome in ("model_inadequate","cost_budget_exhausted","probe_execution_error","cost_unknown"):
                return self.finish(outcome,"See the frozen probe, predictions and measured evidence")
            if outcome=="updated":
                # The same candidate is first re-predicted under the updated physics.
                if self.store.policies:
                    candidates = self.evaluate(f"models/version_{self.model_version:03d}_prediction.json")
        return self.finish("round_budget_exhausted","Automatic revision budget reached before sufficiency")
