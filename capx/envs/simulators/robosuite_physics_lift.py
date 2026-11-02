"""Low-level Robosuite Cube Lift with controllable object contact friction.

This module extends :class:`FrankaRobosuiteCubeLiftLowLevel` so that the
*single* physical parameter that is trivial to modify at runtime — the object's
contact friction — can be controlled per episode. It is the first, smallest
verifiable step of the physics-CaP experiments (M1 "Oracle Physics"), before
centre-of-mass and inertia are made controllable.

Why friction and not mass / inertia?
------------------------------------
MuJoCo folds ``body_mass``, ``body_ipos`` and ``body_inertia`` into the
compiled model constants (the mass matrix and related quantities), so changing
them requires correctly re-deriving / re-compiling those constants. In
contrast ``geom_friction`` is read directly when contact constraints are
solved and can be edited on the live ``mjModel`` without touching compiled
constants, which makes it a safe first version.

Real2Sim2Real note
------------------
This subclass only adds *simulator-side* physics knobs (friction). It does not
change the control surface: the observation and the Franka control API
(``goto_pose`` / ``open_gripper`` / ``close_gripper`` / ``sample_grasp_pose``)
are inherited unchanged from the shared low-level interface, so any program
written against this environment keeps the exact same interface as
``FrankaRealLowLevel`` and can be run on the real arm without modification.
The ``friction`` knob has no real-robot counterpart and is intentionally not
exposed through ``info`` / observations, so agents cannot read the hidden
physics from the environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.simulators.robosuite_cube_lift import (
    FrankaRobosuiteCubeLiftLowLevel,
)


class FrankaRobosuitePhysicsLiftLowLevel(FrankaRobosuiteCubeLiftLowLevel):
    """Cube Lift with controllable object contact friction.

    Args:
        friction: Sliding friction coefficient (MuJoCo ``geom_friction[:, 0]``)
            applied to the cube's contact geoms on every reset. Default 0.5.
        **kwargs: Forwarded to :class:`FrankaRobosuiteCubeLiftLowLevel`
            (``controller_cfg``, ``max_steps``, ``seed``, ``viser_debug``,
            ``privileged``, ``enable_render``).
    """

    def __init__(self, friction: float = 0.5, **kwargs: Any) -> None:
        self.friction = float(friction)
        super().__init__(**kwargs)

    def _apply_physics(self) -> None:
        """Write the configured friction onto the cube's contact geoms.

        Only geoms that actually participate in contacts (``contype`` or
        ``conaffinity`` non-zero) are touched, so visual-only geoms and their
        appearance / geometry are left unchanged.
        """
        sim = self.robosuite_env.sim
        model = sim.model
        bid = self.robosuite_env.cube_body_id

        start = int(model.body_geomadr[bid])
        count = int(model.body_geomnum[bid])

        for gid in range(start, start + count):
            if model.geom_contype[gid] != 0 or model.geom_conaffinity[gid] != 0:
                # Sliding friction only; torsional / rolling have different units.
                model.geom_friction[gid, 0] = self.friction

        # Friction is not part of the compiled mass matrix, but forward once so
        # any cached solver state is consistent with the new constraint params.
        sim.forward()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Apply physics *after* the parent reset: robosuite's hard_reset (the
        # default) rebuilds ``self.model`` and ``self.sim`` from scratch inside
        # ``self.robosuite_env.reset()``, so any edit made beforehand is lost.
        obs, info = super().reset(seed=seed, options=options)
        self._apply_physics()
        # Do not expose hidden physics through info / observations.
        return self.get_observation(), info


__all__ = ["FrankaRobosuitePhysicsLiftLowLevel"]
