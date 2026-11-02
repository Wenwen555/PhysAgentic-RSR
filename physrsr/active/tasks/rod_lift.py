"""Rod Lift task definition, physical parameter contract and v2 adapter.

Only this task module owns its prediction margin and benchmark-specific rules.
The original task evaluator and shared robot execution API remain unchanged.
"""
from ..api.client import APIError
from ..core.contracts import finite

MECHANISM = "rod_counterweight_soft_contact_v1"
FIXED_MASS_MECHANISM = "rod_fixed_mass_com_v1"
ROD_SUCCESS = dict(lift_above_table_m=.04, max_abs_axis_z=.25, hold_steps=10)


class RodMeasuredAdapter:
    task_id = "rod_lift"
    mechanisms = (FIXED_MASS_MECHANISM, MECHANISM)
    parameter_contracts = {
        FIXED_MASS_MECHANISM: dict(com_offset=dict(unit="m", frame="object_body_x", minimum=-.12, maximum=.12),
            friction=dict(unit="1", frame="contact", exclusiveMinimum=0),
            mass_kg=dict(unit="kg", frame="scalar", exclusiveMinimum=0)),
        MECHANISM: dict(com_offset=dict(unit="m", frame="object_body_x", minimum=-.12, maximum=.12),
            friction=dict(unit="1", frame="contact", exclusiveMinimum=0))}

    def validate_task(self, task):
        if task["success"] != ROD_SUCCESS or task["constraints"].get("grip_force_per_finger_n") != 2.:
            raise APIError("Task adapter requires the original success predicate and force cap")

    def validate(self, task, model):
        self.validate_task(task)
        contract = self.parameter_contracts.get(model.mechanism)
        if contract is None or set(model.parameters) != set(contract):
            raise APIError("Unsupported model mechanism/parameters")
        for name, limits in contract.items():
            value = finite(model.parameters[name], name)
            if (value < limits.get("minimum", -float("inf")) or value > limits.get("maximum", float("inf"))
                    or value <= limits.get("exclusiveMinimum", -float("inf"))):
                raise APIError("Physical model parameter outside adapter bounds")

    def execute(self, task, model, policy, seed, belief, purpose, *, execution_budget=None):
        from .rod_lift_world import InstrumentedWorld
        physics = dict(model.parameters, grip_force=2.)
        if model.mechanism == FIXED_MASS_MECHANISM:
            physics["inertial_mode"] = "fixed_mass_com"
        world = InstrumentedWorld(dict(physics=physics, seeds=[seed]))
        world.replay_budget=execution_budget
        try:
            world.call("target.update_identification", dict(belief=belief))
            if purpose == "policy":
                result = world.call("world.evaluate_policy", dict(policy=policy.wire()))
                record = result["record"]
                if record["status"] == "error" and result.get("error_kind") != "program":
                    raise APIError("Predictive simulator execution error: "+record["reason"])
                terminal = result["observations"][-1]["observation"]
                vertical = abs(terminal["object"]["axes"][0][2])
                obj = terminal["object"]
                centre_above_support = obj["support_clearance_m"] + sum(
                    abs(axis[2])*dimension/2 for axis,dimension in zip(obj["axes"],obj["dimensions_m"]))
                margin = min(centre_above_support-ROD_SUCCESS["lift_above_table_m"],
                    ROD_SUCCESS["max_abs_axis_z"]-vertical)
                duration = record.get("sim_duration_s")
                if duration is None:
                    duration = result["observations"][-1]["observation"]["time_s"]-result["observations"][0]["observation"]["time_s"]
                return dict(status=record["status"], reason=record["reason"],error_kind=result.get("error_kind"),
                    success=record["status"] == "success", risk=float(record["status"] == "error"),
                    margin=margin, duration_s=finite(duration,"rollout physical duration"), **{
                        key: result[key] for key in ("observations", "program_return")})
            result = world.call("target.execute_probe", dict(policy=policy.wire()))
            terminal = result["observations"][-1]["observation"]
            return dict(result, risk=float(result["status"] != "completed" or
                terminal["object"]["support_clearance_m"] > .003))
        finally:
            world.close()
