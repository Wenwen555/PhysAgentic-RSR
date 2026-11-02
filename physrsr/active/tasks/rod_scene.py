"""Three coexisting rods: public task, physical contracts and temporal scoring.

One arm lifts, holds and returns each rod. Simultaneous three-rod grasping is
neither required nor counted as three independent environment rollouts.
"""
from copy import deepcopy
import math

from ..api.client import APIError
from ..core.contracts import finite
from ..core.scene import supported

OBJECT_IDS = ("rod_a", "rod_b", "rod_c")
SCENE_MECHANISM = "three_rod_fixed_mass_com_v1"
SCENE_SUCCESS = dict(lift_above_table_m=.04, max_abs_axis_z=.25,
    hold_duration_s=.5, return_stable_duration_s=.25, return_xy_tolerance_m=.04,
    support_tolerance_m=.003, max_return_speed_m_s=.01, max_return_angular_speed_rad_s=.05,
    require_all_objects=True)
DEFAULT_HALF_SIZE = (.15, .015, .015)


class SequentialLiftScore:
    """Advance only on distinct physical control ticks, never observe() calls."""
    def __init__(self, start_time):
        self.last_time = finite(start_time)
        self.progress = {key: dict(hold_s=0., rest_s=0., held=False, returned=False,
            max_hold_s=0.,streak_margin=None, hold_margin=None) for key in OBJECT_IDS}

    def update(self, time_s, states):
        now = finite(time_s)
        if now < self.last_time:
            raise ValueError("Reset the scene score when physical time restarts")
        dt = now-self.last_time
        if dt == 0:
            return
        if set(states) != set(OBJECT_IDS):
            raise ValueError("Score all three rods in the same control tick")
        self.last_time = now
        for key, state in states.items():
            row = self.progress[key]
            if state["lifted"] and state["grasped"] and state["level"]:
                row["hold_s"] += dt
                row["max_hold_s"] = max(row["max_hold_s"],row["hold_s"])
                margin = finite(state["lift_margin"])
                row["streak_margin"] = margin if row["streak_margin"] is None else min(row["streak_margin"],margin)
                if row["hold_s"] >= SCENE_SUCCESS["hold_duration_s"]-1e-9:
                    row["held"] = True
                    row["hold_margin"] = max(row["hold_margin"] or 0., row["streak_margin"])
            else:
                row["hold_s"], row["streak_margin"] = 0., None
            row["rest_s"] = row["rest_s"]+dt if row["held"] and state["at_rest"] else 0.
            row["returned"] = row["held"] and row["rest_s"] >= SCENE_SUCCESS["return_stable_duration_s"]-1e-9

    def report(self):
        rows = deepcopy(self.progress)
        ok = all(row["returned"] for row in rows.values())
        missing = [f"{key}:"+("not_held" if not row["held"] else "not_returned_stably")
            for key,row in rows.items() if not row["returned"]]
        return dict(success=ok, reason="" if ok else ", ".join(missing), objects=rows,
            margin=min(row["hold_margin"] for row in rows.values()) if ok else -.04)


def private_scene(config):
    """Validate evaluator-only physics before allocating any environment."""
    rods = config["objects"]
    if set(rods) != set(OBJECT_IDS):
        raise ValueError("A rod scene requires exactly rod_a, rod_b and rod_c")
    force = finite(config.get("grip_force",2.))
    if force != 2.:
        raise ValueError("The scene shares the unchanged 2 N per-finger cap")
    result = {}
    for index,key in enumerate(OBJECT_IDS):
        rod = deepcopy(rods[key])
        physics = rod["physics"]
        if set(physics) != {"mass_kg","com_offset","friction"}:
            raise ValueError("Each rod requires mass_kg, com_offset and friction")
        half = rod.get("half_size_m",DEFAULT_HALF_SIZE)
        if len(half)!=3 or any(finite(x)<=0 for x in half) or not half[0]>max(half[1:]):
            raise ValueError("Rod half sizes must describe a positive long x-axis box")
        if (not 0<finite(physics["mass_kg"])<=.5 or not 0<finite(physics["friction"])<=3.
                or abs(finite(physics["com_offset"]))>=half[0]):
            raise ValueError("Invalid scene rod physics")
        xy = rod.get("table_xy_m",[0.,(index-1)*.18])
        if len(xy)!=2 or any(abs(finite(x))>.3 for x in xy):
            raise ValueError("Rod placement is outside the table")
        yaw = finite(rod.get("yaw_rad",0.))
        # Restrict the initial geometry to the table, including rotated corners.
        extent = [abs(math.cos(yaw))*half[0]+abs(math.sin(yaw))*half[1],
            abs(math.sin(yaw))*half[0]+abs(math.cos(yaw))*half[1]]
        if any(abs(x)+e>.39 for x,e in zip(xy,extent)):
            raise ValueError("The full rod must start inside the table")
        result[key] = dict(physics=physics,half_size_m=list(half),table_xy_m=list(xy),yaw_rad=yaw)
    seeds=config["seeds"]
    if not isinstance(seeds,list) or not seeds or any(isinstance(s,bool) or not isinstance(s,int) or s<0 for s in seeds):
        raise ValueError("Explicit nonnegative scene reset seeds are required")
    return dict(objects=result,seeds=list(seeds),grip_force=force,initial_scene=deepcopy(config.get("initial_scene")))


class ThreeRodAdapter:
    task_id = "three_rod_lift_return"
    mechanisms = (SCENE_MECHANISM,)
    parameter_contracts = {SCENE_MECHANISM: {f"{key}__{name}":dict(limits,
        object_id=key,quantity=name) for key in OBJECT_IDS for name,limits in dict(
            mass_kg=dict(unit="kg",frame="scalar",exclusiveMinimum=0,maximum=.5),
            com_offset=dict(unit="m",frame="object_body_x",minimum=-.12,maximum=.12),
            friction=dict(unit="1",frame="contact",exclusiveMinimum=0,maximum=3.)).items()}}

    def validate(self, task, model):
        if task["success"] != SCENE_SUCCESS or task["constraints"].get("grip_force_per_finger_n")!=2.:
            raise APIError("Three-rod adapter requires the complete lift-hold-return predicate and 2 N cap")
        contract = self.parameter_contracts.get(model.mechanism)
        if contract is None or set(model.parameters)!=set(contract):
            raise APIError("A scene hypothesis must model all three rods")
        for name,limits in contract.items():
            value=finite(model.parameters[name])
            if (value<limits.get("minimum",-float("inf")) or value>limits.get("maximum",float("inf"))
                    or value<=limits.get("exclusiveMinimum",-float("inf"))):
                raise APIError("Scene physical parameter outside adapter bounds")

    def execute(self, task, model, policy, seed, belief, purpose):
        from .rod_scene_world import ThreeRodWorld
        self.validate(task,model)
        if purpose not in ("probe","policy"):
            raise APIError("Unknown scene prediction purpose")
        scene=task.get("scene")
        if not scene or set(scene.get("objects",{}))!=set(OBJECT_IDS):
            raise APIError("Scene prediction requires the full public initial scene")
        config=dict(seeds=[seed],initial_scene=scene,objects={key:dict(
            physics={name:model.parameters[f"{key}__{name}"] for name in ("mass_kg","com_offset","friction")},
            half_size_m=[v/2 for v in scene["objects"][key]["dimensions_m"]]) for key in OBJECT_IDS})
        world=ThreeRodWorld(config)
        try:
            world.call("target.update_identification",dict(belief=belief))
            result=world.call("world.evaluate_policy" if purpose=="policy" else "target.execute_probe",dict(policy=policy.wire()))
            if purpose=="probe":
                return dict(result,risk=float(result["status"]!="completed" or
                    not supported(result["observations"][-1]["observation"])))
            record=result["record"]
            return dict(status=record["status"],reason=record["reason"],success=record["status"]=="success",
                risk=float(record["status"]=="error"),margin=result["task_result"]["margin"],
                duration_s=result["cost_s"],observations=result["observations"],program_return=result["program_return"],
                task_result=result["task_result"])
        finally:
            world.close()
