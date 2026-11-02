"""Physics and task specifications for the physics-CaP experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhysicsSpec:
    """Oracle physical parameters of the rod.

    In M1 (this stage) every agent — Direct-CaP and Sim-CaP alike — receives
    these exact values. M2 removes them and adds active probing.
    """

    rod_half_size: tuple[float, float, float] = (0.15, 0.015, 0.015)
    com_offset: float = 0.05  # CoM offset along the rod long axis (m)
    density: float = 1000.0  # kg / m^3
    friction: float = 1.0  # sliding friction coefficient
    #: Contact-difficulty condition: cap on each finger's force (N). ``None``
    #: keeps the robot's own limit (20 N), under which the grip is so much
    #: stronger than friction that ``friction`` is unobservable. A value near
    #: the friction limit (e.g. 2 N) makes the grasp slip for low mu, which is
    #: what turns the friction axis into a real physical effect. This is a
    #: gripper condition rather than a rod property, but it is part of the
    #: physical configuration the agent must cope with.
    grip_force: float | None = None
    inertial_mode: str = "counterweight"
    mass_kg: float | None = None

    @property
    def rod_half_length(self) -> float:
        return self.rod_half_size[0]

    @property
    def case_id(self) -> str:
        """Stable, filesystem-safe identifier for this physics configuration."""
        sign = "p" if self.com_offset >= 0 else "m"
        base = f"com_{sign}{abs(self.com_offset) * 1000:03.0f}mm_mu{self.friction:.2f}"
        if self.grip_force is not None:
            base += f"_gf{self.grip_force:.1f}N"
        if self.inertial_mode == "fixed_mass_com":
            base += f"_fixedmass{self.mass_kg * 1000:.0f}g"
        return base

    def as_env_kwargs(self) -> dict:
        """Map to :class:`FrankaRobosuiteRodLiftLowLevel` constructor kwargs."""
        kwargs = {
            "rod_half_size": list(self.rod_half_size),
            "com_offset": self.com_offset,
            "density": self.density,
            "friction": self.friction,
            "grip_force": self.grip_force,
        }
        if self.inertial_mode != "counterweight" or self.mass_kg is not None:
            kwargs.update(inertial_mode=self.inertial_mode, mass_kg=self.mass_kg)
        return kwargs


@dataclass(frozen=True)
class TestCase:
    """One physics configuration + candidate pool, evaluated in a paired design.

    Direct-CaP and Sim-CaP both see the same physics and the same candidate
    pool; only Sim-CaP also gets simulator-rollout feedback.
    """

    case_id: str
    physics: PhysicsSpec
    candidate_ids: tuple[str, ...]


@dataclass
class RolloutRecord:
    """Outcome of a single (program, seed) rollout.

    ``status`` separates a genuine physical failure (``"fail"``) from an
    execution exception (``"error"``). Active adapters distinguish generated program
    errors from infrastructure faults via ``error_kind``. Treating an
    exception as a failure — as the first harness version did — silently turns
    a broken pipeline into a plausible-looking zero success rate.
    """

    case_id: str
    program_id: str
    seed: int
    role: str  # "sim" | "gt"
    status: str  # "success" | "fail" | "error"
    reason: str = ""
    failure_step: int | None = None
    #: Phase in which the rollout broke down (``pre_grasp`` / ``contact`` /
    #: ``lift`` / ``transport``); ``None`` on success.
    failure_phase: str | None = None
    #: Whether *both* finger pads ended up touching the object when the gripper
    #: closed (a real two-finger grasp, as opposed to pinching the top edge).
    grasp_established: bool = False
    #: Per-phase state samples; see :mod:`physrsr.phase_tracker`.
    phases: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    #: MuJoCo elapsed time after reset, including partial execution on error.
    #: None only when no reset/start clock was established; excludes host/IK latency.
    sim_duration_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass
class CaseResult:
    """Per-case outcome of the paired comparison.

    All candidates are scored on the ground-truth worker (not just the two
    chosen ones), so the best achievable program (oracle) and the regret of
    each method are well defined.
    """

    case_id: str
    com_offset: float
    friction: float
    density: float
    direct_choice: str
    sim_choice: str
    oracle_choice: str
    direct_gt_rate: float
    sim_gt_rate: float
    oracle_gt_rate: float
    # Per-GT-seed outcomes, ordered by seed and aligned between direct/sim so a
    # paired (within-seed) comparison is possible.
    direct_gt_outcomes: list[int]
    sim_gt_outcomes: list[int]
    gt_seeds: list[int]
    sim_feedback: dict[str, float]
    gt_rates: dict[str, float]
    #: Grip-force condition (N); ``None`` = the robot's own limit.
    grip_force: float | None = None
    # Per-candidate per-seed outcomes (seed-ordered), so later analysis can do
    # split-half oracle estimates and paired per-candidate comparisons.
    sim_outcomes: dict[str, list[int]] = field(default_factory=dict)
    gt_outcomes: dict[str, list[int]] = field(default_factory=dict)
    n_sim_errors: int = 0
    n_gt_errors: int = 0
    error_messages: dict[str, int] = field(default_factory=dict)
