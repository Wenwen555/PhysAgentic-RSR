"""Robosuite "Rod Lift" environment (long object) with controllable physics.

Step 3 of the physics-CaP experiment: extend Cube Lift into a long rod object
with a *physically self-consistent* mass distribution, and replace the
instantaneous lift check with a stable-hold success condition.

Mass / CoM / inertia model
--------------------------
The rod is a uniform box with an optional hidden counterweight at one end. The
counterweight is a non-visual, non-colliding sphere geom placed inside the same
body, so MuJoCo sums the two geoms and computes the combined mass, centre of
mass and inertia tensor itself (parallel-axis theorem included). This guarantees
the resulting inertial parameters always correspond to a realisable "uniform rod
+ point weight" distribution, rather than an arbitrary hand-edited
``body_ipos`` / ``body_inertia``.

An opt-in ``fixed_mass_com`` mode instead uses an explicit MJCF inertial element
to hold total mass and CoM-frame principal inertia constant while moving only
the CoM. This mode is a controlled inertial intervention; historical experiments
retain their original counterweight mode by default.

Real2Sim2Real note
------------------
The control surface is inherited unchanged from the shared low-level interface,
so programs written against this environment hot-swap to ``FrankaRealLowLevel``.
Physics knobs (rod size, CoM offset, density, friction) are simulator-side only;
they are not exposed through ``info`` / observations.

The analytic model is exposed as :func:`rod_mass_properties` so tests can verify
that MuJoCo's compiled inertial parameters match the parallel-axis prediction.
"""

from __future__ import annotations

from typing import Any
from xml.etree.ElementTree import Element

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, new_geom
from robosuite.utils.placement_samplers import UniformRandomSampler

from capx.envs.simulators.robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel

#: Radius of the hidden counterweight sphere (metres). Kept small so its own
#: rotational inertia is negligible relative to the rod's.
_COUNTERWEIGHT_RADIUS = 0.003
#: Distance from the rod end to the counterweight centre (metres).
_COUNTERWEIGHT_INSET = 0.006


def _counterweight_spec(
    half_size: tuple[float, float, float],
    density: float,
    com_offset: float,
) -> tuple[float, float]:
    """Return (counterweight x-position, counterweight mass) for a target CoM.

    The counterweight sits on the rod's long (x) axis, just inside the end the
    CoM is biased towards, and its mass is solved so that the combined CoM lands
    exactly at ``com_offset``.
    """
    a = float(half_size[0])
    rho = float(density)
    x_c = float(com_offset)

    if abs(x_c) <= 1e-9:
        return 0.0, 0.0

    end = 1.0 if x_c > 0 else -1.0
    x_w = end * (a - _COUNTERWEIGHT_INSET)

    m_b = rho * 8.0 * float(half_size[0]) * float(half_size[1]) * float(half_size[2])
    # x_c = (m_b * 0 + m_w * x_w) / (m_b + m_w)  =>  solve for m_w
    m_w = m_b * x_c / (x_w - x_c)
    if m_w <= 0:
        raise ValueError(f"com_offset={x_c} is not reachable with a counterweight inside the rod")
    return x_w, m_w


def rod_mass_properties(
    half_size: tuple[float, float, float],
    density: float,
    com_offset: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Analytic mass properties of the "uniform rod + point weight" model.

    Args:
        half_size: rod half-extents (a, b, c) along the (x, y, z) body axes.
        density: rod material density (kg/m^3).
        com_offset: desired CoM offset along the rod's long (x) axis (metres).
            Positive biases towards the +x end. 0 means a uniform, centred rod.

    Returns:
        (mass, com, inertia_diag): total mass (kg), CoM in the body frame (m),
        and the diagonal of the inertia tensor about the CoM (kg m^2).
    """
    a, b, c = (float(x) for x in half_size)
    rho = float(density)
    L, W, H = 2.0 * a, 2.0 * b, 2.0 * c

    m_b = rho * L * W * H  # uniform rod mass
    x_w, m_w = _counterweight_spec(half_size, rho, com_offset)
    x_c = float(com_offset)

    mass = m_b + m_w
    com = np.array([x_c, 0.0, 0.0])

    # Uniform box inertia about its own centre (principal axes = body axes).
    ixx_b = m_b / 12.0 * (W * W + H * H)
    iyy_b = m_b / 12.0 * (L * L + H * H)
    izz_b = m_b / 12.0 * (L * L + W * W)

    # Parallel-axis transfer to the CoM (counterweight treated as a point mass).
    ixx = ixx_b  # point on the x-axis contributes nothing about x
    iyy = iyy_b + m_b * x_c * x_c + m_w * (x_w - x_c) ** 2
    izz = izz_b + m_b * x_c * x_c + m_w * (x_w - x_c) ** 2
    inertia_diag = np.array([ixx, iyy, izz])

    return mass, com, inertia_diag


def fixed_mass_properties(half_size, mass_kg, com_offset):
    """Controlled CoM intervention: constant mass and inertia about the CoM.

    This explicit inertial model isolates CoM from total mass and principal
    inertia. It is separate from the historical uniform-rod/end-weight model.
    """
    half = np.asarray(half_size, dtype=float)
    mass, offset = float(mass_kg), float(com_offset)
    if (half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0)
            or not np.isfinite(mass) or mass <= 0 or not np.isfinite(offset) or abs(offset) >= half[0]):
        raise ValueError("Invalid fixed-mass rod inertial parameters")
    a, b, c = half
    inertia = mass / 3. * np.array([b*b+c*c, a*a+c*c, a*a+b*b])
    return mass, np.array([offset, 0., 0.]), inertia


class RodObject(BoxObject):
    """A long box ("rod") with an optional hidden counterweight at one end.

    The counterweight is injected as a second geom in the same body, so MuJoCo
    derives the combined inertial properties itself (see module docstring).
    """

    def __init__(self, name: str, size, com_offset: float = 0.0, density=None,
                 inertial_mode="counterweight", mass_kg=None, **kwargs) -> None:
        self._com_offset = float(com_offset)
        if inertial_mode not in ("counterweight", "fixed_mass_com"):
            raise ValueError("Unknown rod inertial mode")
        if inertial_mode == "fixed_mass_com":
            fixed_mass_properties(size, mass_kg, com_offset)
        elif mass_kg is not None:
            raise ValueError("mass_kg requires fixed_mass_com mode")
        self._inertial_mode, self._mass_kg = inertial_mode, mass_kg
        super().__init__(name=name, size=size, density=density, **kwargs)

    def _get_object_subtree(self):
        obj = super()._get_object_subtree()

        # MuJoCo mixes the contact parameters of two geoms by taking the
        # element-wise *maximum* when their priorities are equal. Robosuite's
        # finger pads declare friction 2.0, so with the default priority the
        # rod's friction coefficient never reaches a single contact and the
        # whole friction axis is inert (verified: mu in {0.15, 0.35, 1.0} gave
        # bit-identical contact friction). Giving the rod's surface a higher
        # priority makes the object's material govern its own contacts, which is
        # the physically right model for a smooth rod on rubber pads.
        for geom in obj.findall("geom"):
            if (geom.get("name") or "").endswith("_vis"):
                continue
            geom.set("priority", "1")

        if self._inertial_mode == "fixed_mass_com":
            mass, com, inertia = fixed_mass_properties(self.size, self._mass_kg, self._com_offset)
            obj.append(Element("inertial", {
                "mass": str(mass), "pos": " ".join(map(str, com)),
                "diaginertia": " ".join(map(str, inertia)),
            }))
            return obj

        x_w, m_w = _counterweight_spec(self.size, self.density, self._com_offset)
        if m_w > 0:
            obj.append(
                new_geom(
                    name="counterweight",
                    type="sphere",
                    size=[_COUNTERWEIGHT_RADIUS],
                    pos=[x_w, 0.0, 0.0],
                    # MuJoCo only counts geoms in the collision group (bit 0) towards
                    # the body's mass/inertia, so group=0 is required for the weight to
                    # have any effect. group=0 is not rendered (no visual bit), and
                    # conaffinity=0 guarantees it never generates a contact.
                    group=0,
                    contype=1,
                    conaffinity=0,
                    mass=float(m_w),
                )
            )
        return obj


class RodLift(Lift):
    """Robosuite Lift environment with a long rod object and controllable physics.

    Extra args (in addition to all :class:`Lift` args):

    Args:
        rod_half_size: rod half-extents (a, b, c) along (x, y, z). Default
            (0.15, 0.015, 0.015) -> 0.30 m x 0.03 m x 0.03 m rod.
        com_offset: CoM offset along the rod's long axis (metres).
        density: rod material density (kg/m^3). Default None -> MuJoCo 1000.
        friction: sliding friction coefficient for the rod contact geom.
        lift_threshold: required height of the rod above the table top (metres).
        hold_steps: number of consecutive successful checks before the task is
            considered stably held (replaces the instantaneous lift check).
        level_tolerance: max |z| component of the rod's world-frame long axis to
            count as "level" (None disables the orientation check).
        grip_force: optional cap (N) on each finger actuator's force, i.e. the
            contact-difficulty knob. The Panda's compiled limit is 20 N, far
            above what a 0.27 kg rod needs, which makes the object's friction
            coefficient unobservable; a lower cap puts the grip near the
            friction limit so mu actually decides whether the rod is held.
            ``None`` keeps the robot's own limit.
    """

    def __init__(
        self,
        robots,
        rod_half_size=(0.15, 0.015, 0.015),
        com_offset: float = 0.0,
        density=None,
        friction: float = 1.0,
        lift_threshold: float = 0.04,
        hold_steps: int = 5,
        level_tolerance=None,
        grip_force: float | None = None,
        inertial_mode: str = "counterweight",
        mass_kg: float | None = None,
        **kwargs,
    ):
        self.rod_half_size = [float(x) for x in rod_half_size]
        self.com_offset = float(com_offset)
        self.rod_density = density
        self.rod_friction = float(friction)
        self.lift_threshold = float(lift_threshold)
        self.hold_steps = int(hold_steps)
        self.level_tolerance = None if level_tolerance is None else float(level_tolerance)
        self.grip_force = None if grip_force is None else float(grip_force)
        self.inertial_mode, self.mass_kg = inertial_mode, mass_kg
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        super()._load_model()

        # Adjust base pose accordingly (same as Lift)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Rod object with controllable mass distribution. The name "cube" is
        # kept on purpose so Lift's _setup_references / observables / grasp
        # detection continue to work unchanged.
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = RodObject(
            name="cube",
            size=self.rod_half_size,
            com_offset=self.com_offset,
            density=self.rod_density,
            inertial_mode=self.inertial_mode,
            mass_kg=self.mass_kg,
            friction=self.rod_friction,
            rgba=[1, 0, 0, 1],
            material=redwood,
            rng=self.rng,
        )

        # Create placement initializer (same as Lift)
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def seed_episode(self, seed: int) -> None:
        """Seed every RNG that drives episode initialisation.

        robosuite's placement sampler is created with the *constructor's* rng
        and reused across hard resets, so seeding ``self.rng`` alone leaves
        object placement nondeterministic (verified: the same seed placed the
        rod differently in different processes). Seeding both makes a
        ``(program, seed)`` rollout reproducible, which the paired Direct-vs-Sim
        comparison depends on.

        Must be called *before* ``reset()``.
        """
        self.seed = int(seed)
        self.rng = np.random.default_rng(int(seed))
        if self.placement_initializer is not None:
            self.placement_initializer.rng = np.random.default_rng(int(seed))

    def _reset_internal(self):
        super()._reset_internal()
        # Stable-hold state resets every episode.
        self._lift_hold_count = 0
        self._last_hold_check_time = None
        # `hard_reset=True` recompiles the model on every reset, so the grip
        # limit has to be reapplied here rather than once in __init__.
        if self.grip_force is not None:
            self._apply_grip_force_limit()

    def _apply_grip_force_limit(self) -> None:
        """Cap each finger actuator's force at ``self.grip_force`` newtons."""
        model = self.sim.model
        for i in range(model.nu):
            name = model.actuator_id2name(i) or ""
            if "finger_joint" in name:
                model.actuator_forcerange[i] = [-self.grip_force, self.grip_force]

    def _is_lifted(self) -> bool:
        rod_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return rod_height > table_height + self.lift_threshold

    def _is_grasped(self) -> bool:
        return self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)

    def _is_level(self) -> bool:
        if self.level_tolerance is None:
            return True
        xmat = self.sim.data.body_xmat[self.cube_body_id].reshape(3, 3)
        # world-frame z-component of the rod's long (body x) axis.
        return abs(xmat[2, 0]) < self.level_tolerance

    def _is_successful_state(self) -> bool:
        """Pure (side-effect-free) instantaneous success check.

        Exposed so external workers can evaluate a rollout's final state without
        driving the stable-hold counter in :meth:`_check_success`.
        """
        return self._is_lifted() and self._is_grasped() and self._is_level()

    def _update_hold_state(self) -> None:
        """Advance the stable-hold counter once per (sim) time step.

        :meth:`_check_success` is called from both ``reward()`` and the wrapper's
        ``task_completed()``, so the counter update is guarded by the current sim
        time to avoid double-counting within a single high-level step.
        """
        now = float(self.sim.data.time)
        if now == self._last_hold_check_time:
            return
        self._last_hold_check_time = now

        if self._is_successful_state():
            self._lift_hold_count += 1
        else:
            self._lift_hold_count = 0

    def _check_success(self):
        """Rod is stably held: lifted, grasped and level for ``hold_steps`` checks."""
        self._update_hold_state()
        return self._lift_hold_count >= self.hold_steps


class FrankaRobosuiteRodLiftLowLevel(FrankaRobosuiteCubeLiftLowLevel):
    """Low-level Rod Lift wrapper with FrankaPickPlaceLowLevel-compatible interface."""

    _robosuite_env_cls = RodLift

    def __init__(
        self,
        rod_half_size=(0.15, 0.015, 0.015),
        com_offset: float = 0.0,
        density=None,
        friction: float = 1.0,
        lift_threshold: float = 0.04,
        hold_steps: int = 5,
        level_tolerance=None,
        grip_force: float | None = None,
        inertial_mode: str = "counterweight",
        mass_kg: float | None = None,
        **kwargs,
    ) -> None:
        self._rod_half_size = [float(x) for x in rod_half_size]
        self._com_offset = float(com_offset)
        self._density = density
        self._friction = float(friction)
        self._lift_threshold = float(lift_threshold)
        self._hold_steps = int(hold_steps)
        self._level_tolerance = None if level_tolerance is None else float(level_tolerance)
        self._grip_force = None if grip_force is None else float(grip_force)
        self._inertial_mode, self._mass_kg = inertial_mode, mass_kg
        super().__init__(**kwargs)

    def _robosuite_env_kwargs(self) -> dict[str, Any]:
        return {
            "rod_half_size": self._rod_half_size,
            "com_offset": self._com_offset,
            "density": self._density,
            "friction": self._friction,
            "lift_threshold": self._lift_threshold,
            "hold_steps": self._hold_steps,
            "level_tolerance": self._level_tolerance,
            "grip_force": self._grip_force,
            "inertial_mode": self._inertial_mode,
            "mass_kg": self._mass_kg,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info["task_prompt"] = "Pick up the red rod and lift it, keeping it level. Quaternions are WXYZ."
        return obs, info


__all__ = [
    "FrankaRobosuiteRodLiftLowLevel",
    "RodLift",
    "RodObject",
    "rod_mass_properties",
    "fixed_mass_properties",
]
