"""Compile-only inertial tests; no robot task execution or inference."""
import copy
from xml.etree.ElementTree import Element, SubElement, tostring

import mujoco
import numpy as np
import pytest

from capx.envs.simulators.robosuite_rod_lift import RodObject, fixed_mass_properties, rod_mass_properties
from physrsr.physics_spec import PhysicsSpec


@pytest.mark.parametrize("offset", [0., -.04, .08])
def test_compiled_rod_changes_only_com(offset):
    rod = RodObject("rod", [.15, .015, .015], com_offset=offset, density=1000.,
        inertial_mode="fixed_mass_com", mass_kg=.27, friction=1.5)
    root = Element("mujoco")
    world = SubElement(root, "worldbody")
    world.append(copy.deepcopy(rod.get_obj()))
    model = mujoco.MjModel.from_xml_string(tostring(root, encoding="unicode"))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rod.root_body)
    expected_mass, expected_com, expected_inertia = fixed_mass_properties([.15, .015, .015], .27, offset)
    assert model.body_mass[body] == pytest.approx(expected_mass)
    np.testing.assert_allclose(model.body_ipos[body], expected_com, atol=1e-12)
    np.testing.assert_allclose(sorted(model.body_inertia[body]), sorted(expected_inertia), atol=1e-12)
    assert not any("counterweight" in (model.geom(i).name or "") for i in range(model.ngeom))


def test_fixed_inertia_does_not_covary_with_offset_and_legacy_mode_is_preserved():
    rows = [fixed_mass_properties([.15, .015, .015], .27, offset) for offset in (0., -.04, .08)]
    assert all(np.array_equal(r[2], rows[0][2]) for r in rows)
    assert rod_mass_properties((.15, .015, .015), 1000., .08)[0] > .6
    assert PhysicsSpec().as_env_kwargs().get("inertial_mode") is None
    configured = PhysicsSpec(com_offset=.08, inertial_mode="fixed_mass_com", mass_kg=.27).as_env_kwargs()
    assert configured["mass_kg"] == .27
    assert configured["inertial_mode"] == "fixed_mass_com"
