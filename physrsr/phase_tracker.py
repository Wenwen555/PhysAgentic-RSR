"""Per-phase instrumentation for CaP programs.

A rollout used to report only the first violated success condition
(``not_lifted`` / ``not_grasped`` / ``not_level``). That says *what* failed but
not *where*: the same ``not_lifted`` can come from a bad approach (the gripper
knocked the object over), a bad contact (the fingers closed on air) or a bad
lift (the object slipped out). This module segments a program into the phases
of the classic manipulation pipeline and samples the physical state at each
boundary:

============  ==========================================================
phase         what it should establish
============  ==========================================================
pre_grasp     the arm reaches the pre-grasp pose without disturbing the
              object (checks IK/controller accuracy and approach collisions)
contact       an *effective* grasp: both finger pads touch the object
lift          the object leaves the table (CoM / grip force / friction)
transport     the object stays grasped, lifted and level while the arm moves
placement     the object comes to rest stably (not used by the rod-lift task,
              whose success condition is a stable held lift)
============  ==========================================================

Phase segmentation is inferred from the control-API calls only (``goto_pose`` /
``open_gripper`` / ``close_gripper``), so the exact same code runs on the real
arm. The state sampling is injected via ``state_fn``: here it reads privileged
simulator state, while on hardware it would read the same quantities from the
robot's observables (gripper width, FT/contact sensing, object tracking).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

PHASES = ("pre_grasp", "contact", "lift", "transport", "placement")

#: A vertical command move at least this large (m) after the grasp counts as a lift.
LIFT_DZ = 0.02
#: Object displacement (m) above which the approach is considered disturbing.
DISTURB_TOL = 0.005


@dataclass
class PhaseSample:
    """Physical state at one phase boundary."""

    phase: str
    #: commanded TCP target (robot base frame)
    commanded: list[float] | None = None
    #: dimensionless speed fraction passed to ``goto_pose`` (None = controller max)
    commanded_speed: float | None = None
    #: measured TCP pose (robot base frame)
    achieved: list[float] | None = None
    #: |commanded - achieved| in metres; surfaces IK / controller limits
    tcp_error: float | None = None
    rod_position: list[float] | None = None
    rod_z: float | None = None
    #: |z| component of the rod long axis: 0 = level
    rod_tilt: float | None = None
    lifted: bool = False
    grasped: bool = False
    level: bool = False
    #: both finger pads touch the object (a real two-finger grasp)
    grasp_established: bool = False
    finger_width: float | None = None
    #: object displacement since the episode started (m)
    rod_displacement: float | None = None
    duration_s: float | None = None
    #: filled for samples taken during the post-program hold
    step_in_hold: int | None = None
    hold_ok: bool | None = None


def _obj_geom_ids(model, body_id: int) -> set[int]:
    start = int(model.body_geomadr[body_id])
    return set(range(start, start + int(model.body_geomnum[body_id])))


def rod_lift_state(env) -> dict:
    """Privileged state of the rod-lift environment (simulator implementation)."""
    rod = env.robosuite_env
    sim = rod.sim
    model, data = sim.model, sim.data
    bid = rod.cube_body_id

    pos = np.asarray(data.body_xpos[bid], dtype=np.float64)
    xmat = np.asarray(data.body_xmat[bid], dtype=np.float64).reshape(3, 3)

    finger_ids = np.asarray(rod.robots[0]._ref_gripper_joint_pos_indexes["right"])
    finger_q = np.asarray(data.qpos[finger_ids], dtype=np.float64)

    obj_geoms = _obj_geom_ids(model, bid)
    pad_contact = {"finger1": False, "finger2": False}
    for ci in range(int(data.ncon)):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in obj_geoms:
            other = model.geom_id2name(g2) or ""
        elif g2 in obj_geoms:
            other = model.geom_id2name(g1) or ""
        else:
            continue
        for finger in pad_contact:
            if finger in other:
                pad_contact[finger] = True

    return {
        "rod_position": pos.tolist(),
        "rod_z": float(pos[2]),
        "rod_tilt": float(abs(xmat[2, 0])),
        "lifted": bool(rod._is_lifted()),
        "grasped": bool(rod._is_grasped()),
        "level": bool(rod._is_level()),
        "grasp_established": bool(pad_contact["finger1"] and pad_contact["finger2"]),
        "finger_width": float(np.abs(finger_q).sum()),
    }


class PhaseTracker:
    """Drop-in proxy around the control API that records per-phase state.

    Programs call the same methods as before; every call is forwarded to the
    wrapped API and followed by a state sample. ``failure_phase`` is derived
    from the samples (see :meth:`attribute_failure`).
    """

    def __init__(
        self,
        env,
        api,
        state_fn: Callable[[Any], dict] | None = None,
    ):
        self.env = env
        self.api = api
        self.state_fn = state_fn or rod_lift_state
        self.samples: list[PhaseSample] = []
        self.closed = False
        self.grasp_established = False
        self.stage = "pre_grasp"
        self._last_command: np.ndarray | None = None
        self._t0 = time.perf_counter()
        self._initial_rod: np.ndarray | None = None

    # ------------------------------------------------------------- API proxy
    def goto_pose(
        self,
        position,
        quaternion_wxyz,
        z_approach: float = 0.0,
        speed: float | None = None,
    ) -> None:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        self.api.goto_pose(position, quaternion_wxyz, z_approach=z_approach, speed=speed)
        phase = self._classify_move(position)
        self._record(phase, commanded=position, speed=speed)

    def open_gripper(self) -> None:
        self.api.open_gripper()

    def close_gripper(self) -> None:
        # The approach phase ends where the gripper starts closing.
        self._record("pre_grasp")
        self.api.close_gripper()
        self.closed = True
        sample = self._record("contact")
        if sample is not None:
            self.grasp_established = sample.grasp_established
        self.stage = "lift"

    def __getattr__(self, name: str) -> Any:
        # Anything not proxied (e.g. ``sample_grasp_pose``) falls through.
        return getattr(self.api, name)

    # -------------------------------------------------------------- sampling
    def _classify_move(self, target: np.ndarray) -> str:
        if not self.closed:
            return "pre_grasp"
        prev = self._last_command
        if prev is None:
            return "lift"
        dz = float(target[2] - prev[2])
        lateral = float(np.linalg.norm((target - prev)[:2]))
        if dz >= LIFT_DZ and lateral < 0.01:
            return "lift"
        return "transport"

    def _record(
        self,
        phase: str,
        commanded: np.ndarray | None = None,
        speed: float | None = None,
    ) -> PhaseSample | None:
        try:
            state = self.state_fn(self.env)
        except Exception:  # noqa: BLE001 - instrumentation must never break a rollout
            return None
        measured = None
        tcp_error = None
        getter = getattr(self.env, "get_eef_pose", None)
        if getter is not None:
            try:
                _, measured_pos = getter()
                measured = np.asarray(measured_pos, dtype=np.float64).tolist()
                if commanded is not None:
                    tcp_error = float(
                        np.linalg.norm(np.asarray(measured) - np.asarray(commanded))
                    )
            except Exception:  # noqa: BLE001
                pass

        rod_pos = np.asarray(state["rod_position"], dtype=np.float64)
        if self._initial_rod is None:
            self._initial_rod = rod_pos.copy()
        displacement = float(np.linalg.norm(rod_pos - self._initial_rod))

        sample = PhaseSample(
            phase=phase,
            commanded=None if commanded is None else np.asarray(commanded).tolist(),
            commanded_speed=None if speed is None else float(speed),
            achieved=measured,
            tcp_error=tcp_error,
            rod_position=state["rod_position"],
            rod_z=state["rod_z"],
            rod_tilt=state["rod_tilt"],
            lifted=state["lifted"],
            grasped=state["grasped"],
            level=state["level"],
            grasp_established=state["grasp_established"],
            finger_width=state["finger_width"],
            rod_displacement=displacement,
            duration_s=time.perf_counter() - self._t0,
        )
        self.samples.append(sample)
        if commanded is not None:
            self._last_command = np.asarray(commanded, dtype=np.float64)
        return sample

    def record_initial(self) -> None:
        """Baseline sample right after reset (reference for object displacement)."""
        self._record("pre_grasp")

    def record_hold(self, step: int, ok: bool) -> None:
        """Sample the transport phase while the grasp is held after the program."""
        sample = self._record("transport")
        if sample is not None:
            sample.step_in_hold = int(step)
            sample.hold_ok = bool(ok)

    # ------------------------------------------------------------ attribution
    def _phase_samples(self, phase: str) -> list[PhaseSample]:
        return [s for s in self.samples if s.phase == phase]

    def approach_max_tcp_error(self) -> float | None:
        errs = [s.tcp_error for s in self._phase_samples("pre_grasp") if s.tcp_error is not None]
        return max(errs) if errs else None

    def approach_max_displacement(self) -> float | None:
        disp = [
            s.rod_displacement
            for s in self._phase_samples("pre_grasp")
            if s.rod_displacement is not None
        ]
        return max(disp) if disp else None

    def attribute_failure(self) -> str | None:
        """Return the phase in which the rollout broke down.

        Rules, in order:
          1. the object was disturbed before contact -> ``pre_grasp``
          2. no effective two-finger grasp was made -> ``contact``
          3. the object never left the table, or was already tilted when the
             lift move finished -> ``lift`` (CoM / grip force / friction)
          4. otherwise the grasp was lost / tilted while carried -> ``transport``
        """
        displacement = self.approach_max_displacement()
        if (
            not self.grasp_established
            and displacement is not None
            and displacement > DISTURB_TOL
        ):
            return "pre_grasp"
        if not self.grasp_established:
            return "contact"
        lift = self._phase_samples("lift")
        if not lift or not lift[-1].lifted:
            return "lift"
        if not lift[-1].level:
            # The rod was already tilted when the vertical lift finished: the
            # torque about the grasp point did it, not the later transport.
            return "lift"
        transport = self._phase_samples("transport")
        if transport and not transport[-1].lifted:
            return "transport"
        if transport and not transport[-1].level:
            return "transport"
        # Grasped and lifted at the end of the program but the hold still failed
        # (slipped / rotated after the last sample): attribute to transport.
        return "transport"

    def as_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.samples]
