"""Private simulated World: public probes and independent task evaluation."""
from dataclasses import asdict

from .backends import SimulatedRobotService
from .robot import TargetService
from .simulator import APIPolicyProgram, observation_provider
from ..core.policy import Policy


class WorldService(TargetService):
    """The request never supplies target physics or an evaluator seed."""
    def __init__(self, private_config):
        super().__init__(SimulatedRobotService(private_config))
        self.evaluation_seed = int(private_config.get("evaluation_seed", private_config["seeds"][0]))

    def call(self, method, params):
        if method != "world.evaluate_policy":
            return super().call(method, params)
        if set(params) != {"policy"}:
            raise ValueError("World evaluation accepts only the frozen policy")
        policy = Policy.from_wire(params["policy"])
        env, _ = self.robot.ensure()
        program = APIPolicyProgram(policy)
        record = asdict(self.robot.worker.rollout(program, self.evaluation_seed, role="world"))
        samples = [] if program.dispatcher is None else program.dispatcher.samples
        samples.append(dict(event="terminal", observation=observation_provider(env)))
        rod, compiled = env.robosuite_env, env.robosuite_env.sim.model
        body = rod.cube_body_id
        return dict(policy_sha256=policy.sha256, record=record, observations=samples,
            program_return=program.runtime.get("return_value"),
            error_kind=(program.error_kind or "infrastructure") if record["status"] == "error" else None,
            evaluator_only_compiled_parameters=dict(mass_kg=float(compiled.body_mass[body]),
                com_body_m=compiled.body_ipos[body].tolist(),
                principal_inertia_kg_m2=compiled.body_inertia[body].tolist(),
                inertia_quaternion_wxyz=compiled.body_iquat[body].tolist()))
