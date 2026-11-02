"""Trusted Rod Lift simulator service, isolated from target service parameters."""
from __future__ import annotations

from dataclasses import asdict
import math

from ..api.client import APIError
from ..core.contracts import API_VERSION, Hypothesis, finite
from ..core.policy import Policy, execute
from .robot import RobotContractError, RobotDispatcher, RobotService, public_observation, response_features

from ..tasks.rod_lift import MECHANISM, FIXED_MASS_MECHANISM, ROD_SUCCESS


from .observations import observation_provider

class APIPolicyProgram:
    """RolloutWorker protocol, with policy execution exclusively via public RPC."""
    def __init__(self, policy):
        self.id, self.policy = policy.sha256, policy
        self.trace, self.dispatcher = [], None
        self.error_kind = None
        self.runtime = {}

    def run(self, env, api, physics=None):
        service = RobotService(api, lambda: observation_provider(env), lambda: None)
        self.dispatcher = RobotDispatcher(service, observation_provider(env))
        from physrsr.execution import ProgramError
        try:
            self.runtime = execute(self.policy, self.dispatcher)
        except (ProgramError, RobotContractError):
            self.error_kind = "program"
            raise
        except Exception:
            self.error_kind = "infrastructure"
            raise

    def specification(self):
        return dict(id=self.id, source_sha256=self.policy.sha256)


class RodSimulatorService:
    def __init__(self, worker_factory=None):
        self.worker_factory = worker_factory

    def worker(self, model):
        from physrsr.physics_spec import PhysicsSpec
        if self.worker_factory is None:
            from physrsr.worker import RolloutWorker
            factory = RolloutWorker
        else:
            factory = self.worker_factory
        extra = dict(inertial_mode="fixed_mass_com") if model.mechanism == FIXED_MASS_MECHANISM else {}
        return factory(PhysicsSpec(**model.parameters, grip_force=2., **extra))

    def call(self, method, params):
        if method == "sim.describe" and not params:
            return dict(api_version=API_VERSION, mechanisms=[MECHANISM, FIXED_MASS_MECHANISM], tasks=["rod_lift"],
                        feature_schema="rod_response_v1")
        if method not in ("sim.evaluate", "sim.predict_probe"):
            raise APIError(f"Unknown simulator operation: {method}")
        task = params["task"]
        if task["id"] != "rod_lift" or task["success"] != ROD_SUCCESS:
            raise APIError("Rod adapter requires the unchanged Rod Lift success predicate")
        if task["constraints"].get("grip_force_per_finger_n") != 2.0:
            raise APIError("Rod adapter requires the shared 2 N per-finger control limit")
        models = [Hypothesis(**m) for m in params["models"]]
        seeds = params["seeds"]
        if not seeds:
            raise APIError("Simulation requires explicit development seeds")
        for model in models:
            expected = {"com_offset", "friction", "mass_kg"} if model.mechanism == FIXED_MASS_MECHANISM else {"com_offset", "friction"}
            if model.mechanism not in (MECHANISM, FIXED_MASS_MECHANISM) or set(model.parameters) != expected:
                raise APIError("Unsupported physical model; configure a simulator implementing this mechanism")
            if abs(model.parameters["com_offset"]) > .12 or model.parameters["friction"] <= 0:
                raise APIError("Invalid rod physical parameters")
            if model.mechanism == FIXED_MASS_MECHANISM and model.parameters["mass_kg"] <= 0:
                raise APIError("Fixed rod mass must be positive")
        policies = ([Policy.from_wire(params["probe"]["policy"])] if method == "sim.predict_probe"
                    else [Policy.from_wire(p) for p in params["policies"]])
        rows, predictions = [], {}
        for model in models:
            worker = self.worker(model)
            try:
                env, api = worker._ensure_env()
                env.robosuite_env.ignore_done = True
                for policy in policies:
                    outcomes, features, risks, margins, times = [], [], [], [], []
                    for seed in seeds:
                        program = APIPolicyProgram(policy)
                        if method == "sim.evaluate":
                            record = asdict(worker.rollout(program, int(seed), role="model"))
                            if record["status"] == "error" and program.error_kind != "program":
                                raise APIError("Simulator infrastructure failure: " + record["reason"])
                        else:
                            # Probe predictions do not append the task's closed-grip hold.
                            api.cfg = None
                            env.robosuite_env.seed_episode(int(seed))
                            env.reset(seed=int(seed))
                            start = float(env.robosuite_env.sim.data.time)
                            from physrsr.execution import ProgramError
                            try:
                                program.run(env, api)
                                record = dict(status="completed", reason="")
                            except (ProgramError, RobotContractError) as exc:
                                record = dict(status="error", reason=str(exc))
                            record["sim_duration_s"] = float(env.robosuite_env.sim.data.time)-start
                        dispatcher = program.dispatcher
                        if dispatcher is None:
                            raise APIError("Simulator did not start policy execution")
                        dispatcher.samples.append(dict(event="terminal", observation=observation_provider(env)))
                        ff = response_features(dispatcher.samples)
                        risk = float(dispatcher.unsafe_release or max(abs(ff[0]), abs(ff[1])) > .75)
                        if method == "sim.predict_probe" and (abs(ff[2]) > .015 or record["status"] == "error"):
                            risk = 1.0
                        rod = env.robosuite_env
                        clearance = float(rod.sim.data.body_xpos[rod.cube_body_id][2]) - float(rod.model.mujoco_arena.table_offset[2]) - .04
                        margin = min(clearance/.1, .25-abs(ff[0]), .25-abs(ff[1]))
                        outcome = dict(status=record["status"], reason=record["reason"], seed=int(seed),
                            observations=dispatcher.samples, error_kind=program.error_kind,
                            program_return=program.runtime.get("return_value"))
                        if method == "sim.evaluate":
                            outcome["task_evaluation"] = {key: record[key] for key in (
                                "status", "reason", "failure_step", "failure_phase", "grasp_established",
                                "duration_s", "sim_duration_s")}
                            if model.mechanism == FIXED_MASS_MECHANISM:
                                compiled = rod.sim.model
                                body = rod.cube_body_id
                                inertia_check = dict(mass_kg=float(compiled.body_mass[body]),
                                    com_body_m=compiled.body_ipos[body].tolist(),
                                    principal_inertia_kg_m2=compiled.body_inertia[body].tolist(),
                                    inertia_quaternion_wxyz=compiled.body_iquat[body].tolist())
                                if (not math.isclose(inertia_check["mass_kg"], model.parameters["mass_kg"], abs_tol=1e-10)
                                        or not math.isclose(inertia_check["com_body_m"][0], model.parameters["com_offset"], abs_tol=1e-10)):
                                    raise APIError("Compiled rod mass/CoM differs from requested intervention")
                                outcome["compiled_inertial_check"] = inertia_check
                        outcomes.append(outcome)
                        features.append(ff); risks.append(risk); margins.append(margin)
                        elapsed = record.get("sim_duration_s")
                        if elapsed is None:
                            # Worker preserves program errors without duration;
                            # they must reach code revision as failures, not be
                            # mistaken for a broken simulator service.
                            elapsed = float(env.robosuite_env.sim.data.time) - dispatcher.initial["time_s"]
                        times.append(finite(elapsed))
                    if method == "sim.evaluate":
                        rows.append(dict(model_id=model.id, policy_sha256=policy.sha256,
                            success_rate=sum(o["status"] == "success" for o in outcomes)/len(outcomes),
                            risk=max(risks), margin=min(margins), duration_s=sum(times)/len(times), feedback=outcomes))
                    else:
                        predictions[model.id] = dict(features=[sum(x)/len(x) for x in zip(*features)], risk=max(risks))
            finally:
                worker.close()
        return dict(rows=rows) if method == "sim.evaluate" else dict(predictions=predictions,
            policy_sha256=policies[0].sha256, feature_schema="rod_response_v1")
