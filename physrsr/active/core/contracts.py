"""JSON contracts shared by the controller, simulator and real robot services."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

API_VERSION = "physrsr.robot.v1"
IDENTIFICATION_API_VERSION = "physrsr.robot.v2"
ROBOT_CONTRACT = """Define run(robot). Only the following robot API is available:
robot.observe() -> rod_position [x,y,z], rod_axis [x,y,z], grasp_quat [w,x,y,z],
tcp_position [x,y,z], coordinate_frame='robot_base', time_s (sensor timestamp).
Positions are robot_base metres; time_s is seconds. Repeated observations can
have the same timestamp, and observe() does not advance simulation time.
There is no wait/step API; do not busy-poll observations to simulate a timed hold.
robot.open_gripper(); robot.close_gripper();
robot.goto_pose(position, quaternion_wxyz, z_approach=0.0, speed=None).
speed is None or 0.1..1; approach is 0..0.12 m. Target xy is within 0.25 m of
the initial rod centre; z is initial centre z -0.02 .. +0.20 m. At most 96 RPCs.
Use observations for control. Bounded retries and Python math are allowed.
No environment, simulator, hidden physics, filesystem or network access.
The identical Python source runs in the simulator and on the real arm.
Do not change force limits or task success criteria.
"""
FEATURE_SCALES = (.005, .005, .005, .05, .005, .05, .005, .005, .05, .005, .05, .05, .05, .005, .005, .005)


def finite(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class Hypothesis:
    id: str
    parameters: dict
    weight: float
    mechanism: str = "rod_counterweight_soft_contact_v1"

    def __post_init__(self):
        if not self.id or not self.mechanism or not isinstance(self.parameters, dict):
            raise ValueError("Invalid model hypothesis")
        if finite(self.weight, "model weight") < 0:
            raise ValueError("Negative model weight")
        for value in self.parameters.values():
            finite(value, "physical parameter")


def normalize(models):
    if not models or len({m.id for m in models}) != len(models):
        raise ValueError("A nonempty, uniquely named model ensemble is required")
    total = sum(m.weight for m in models)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Model weight sum must be finite and positive")
    return tuple(replace(m, weight=m.weight / total) for m in models)


@dataclass(frozen=True)
class LoopConfig:
    max_rounds: int = 8
    max_probes: int = 6
    max_probe_cost: float = 120.0
    max_model_revisions: int = 2
    max_candidates: int = 12
    min_validation_probes: int = 1
    reliability: float = 0.95
    max_risk: float = 0.05
    min_margin: float = 0.0
    support_mass: float = 0.99
    time_weight: float = 0.001
    risk_weight: float = 1.0
    probe_cost_weight: float = 0.001
    min_net_value: float = 0.0001
    max_prediction_error: float = 3.0
    feature_scales: tuple = FEATURE_SCALES
    model_seeds: tuple = (7201, 7202)
    probe_candidates: int = 2
    max_heuristic_probes: int = 2
    max_heuristic_cost: float = 30.

    def __post_init__(self):
        for key in ("max_rounds", "max_probes", "max_model_revisions", "max_candidates", "min_validation_probes","max_heuristic_probes","probe_candidates"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid {key}")
        if not self.max_rounds or self.max_candidates < 2 or not self.min_validation_probes:
            raise ValueError("Positive rounds/validation and at least two candidates are required")
        if not 1<=self.probe_candidates<=4:raise ValueError("Compare between one and four executable Probe candidates")
        for key in ("reliability", "support_mass"):
            if not 0 < finite(getattr(self, key), key) <= 1:
                raise ValueError(f"Invalid {key}")
        if not 0 <= finite(self.max_risk) <= 1:
            raise ValueError("Invalid max_risk")
        for key in ("max_probe_cost", "time_weight", "risk_weight", "probe_cost_weight", "min_net_value", "max_prediction_error","max_heuristic_cost"):
            if finite(getattr(self, key), key) < 0:
                raise ValueError(f"Invalid {key}")
        finite(self.min_margin, "min_margin")
        if not self.feature_scales or any(finite(x) <= 0 for x in self.feature_scales):
            raise ValueError("Prediction feature scales must be positive")
        if not self.model_seeds or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("Unique development seeds are required")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.model_seeds):
            raise ValueError("Development seeds must be nonnegative integers")


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    success: dict
    constraints: dict

    def public(self, api_version=API_VERSION):
        if api_version == IDENTIFICATION_API_VERSION:
            from .identification import ROBOT_API
            return dict(**asdict(self), api_version=api_version, robot_api=ROBOT_API)
        return dict(**asdict(self), api_version=API_VERSION, robot_api=ROBOT_CONTRACT)


def read_config(payload):
    task = Task(**payload["task"])
    if not task.id or not task.instruction.strip() or not task.success or not task.constraints:
        raise ValueError("Task ID, semantic instruction, success and constraints are required")
    config = LoopConfig(**payload.get("loop", {}))
    if len(config.feature_scales)!=16:
        raise ValueError("Configure exactly 16 base feature scales; scene layout is derived explicitly")
    proposed=[Hypothesis(**m) for m in payload.get("models",[])]
    models = normalize(proposed) if proposed else ()
    if len(models) > 128:
        raise ValueError("At most 128 model hypotheses")
    from ..tasks.registry import BUILTIN_ADAPTERS, create_adapters
    specification=payload.get("adapter",task.id)
    if specification not in BUILTIN_ADAPTERS and ":" not in specification:
        raise ValueError("Unknown task ID; an external task requires an explicit adapter factory")
    adapter=create_adapters([specification])[0]
    if adapter.task_id!=task.id:raise ValueError("Task does not match adapter")
    adapter.validate_task(task.public(IDENTIFICATION_API_VERSION))
    for model in models:adapter.validate(task.public(IDENTIFICATION_API_VERSION),model)
    return task, config, models
