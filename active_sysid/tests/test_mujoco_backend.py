"""MuJoCo 后端测试: 最小门场景 (门铰链 + 把手铰链, 无闩锁)。

只验证后端机制 (通道记录 / 直驱 / 复位确定性 / 与 executor 的衔接),
闩锁语义属于 World 内容, 由 mock 与后续 oracle 场景覆盖。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from active_sysid.backends import MuJoCoBackEnd, create_backend  # noqa: E402
from active_sysid.probes.executor import ProbeExecutor  # noqa: E402
from active_sysid.probes.library import ProbeLibrary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# 门叶 ~12 kg (box 0.8x0.04x2.0 m, density 180), 铰链阻尼 0.4;
# 把手带回位弹簧 (stiffness 0.5, damping 0.08), 与 eval_truth.yaml 同量级
DOOR_XML = """
<mujoco model="test_door">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="frame" pos="0 0 0">
      <!-- 门柱放在叶片扫掠范围之外 (叶片 x in [0, 0.8]) -->
      <geom name="post" type="box" size="0.05 0.05 1.0" pos="-0.06 0 1.0"/>
      <body name="door" pos="0 0 0">
        <joint name="door_hinge" type="hinge" axis="0 0 1"
               range="0 1.508" damping="0.4" frictionloss="0.2"/>
        <geom name="door_leaf" type="box" size="0.4 0.02 1.0"
              pos="0.4 0 1.05" density="180"/>
        <body name="handle" pos="0.75 0 1.0">
          <joint name="handle_hinge" type="hinge" axis="0 0 1"
                 range="-1.2 1.2" damping="0.08" stiffness="0.5" springref="0"/>
          <!-- 把手与门叶几何交叠是建模伪影, 关闭其碰撞参与 (质量/惯量保留) -->
          <geom name="handle_bar" type="capsule" size="0.015 0.06"
                fromto="0 0 -0.08 0 0 0.08" density="800"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

FEATURE_EPS = {"door_immobile_eps": 0.02, "handle_rotates_eps": 0.05,
               "door_opens_eps": 0.20, "return_ratio": 0.6}


@pytest.fixture(scope="module")
def library() -> ProbeLibrary:
    return ProbeLibrary.load(ROOT / "configs" / "probes.yaml")


@pytest.fixture()
def world(tmp_path) -> MuJoCoBackEnd:
    mjcf = tmp_path / "door.mjcf"
    mjcf.write_text(DOOR_XML, encoding="utf-8")
    return MuJoCoBackEnd(
        mjcf, channel_map={"handle_angle": "handle_hinge", "door_angle": "door_hinge"}
    )


def test_channel_map_rejects_unknown_joint(tmp_path):
    mjcf = tmp_path / "door.mjcf"
    mjcf.write_text(DOOR_XML, encoding="utf-8")
    with pytest.raises(ValueError, match="ghost_hinge"):
        MuJoCoBackEnd(mjcf, channel_map={"door_angle": "ghost_hinge"})


def test_rotate_handle_moves_only_handle(world, library):
    traj = world.run_probe(library.get("rotate_handle").instantiate("door_0"))
    assert traj.duration() == pytest.approx(2.0, abs=0.01)  # 采样到 n*dt - dt
    assert np.ptp(traj.channels["handle_angle"]) > 0.3
    assert traj.channels["door_angle"].max() < 0.05
    assert "handle_angle_vel" in traj.channels
    assert "handle_angle_tau_applied" in traj.channels


def test_hold_and_pull_opens_unlocked_door(world, library):
    spec = library.get("hold_and_pull").instantiate(
        "door_0", overrides={"handle_angle": 0.8, "max_force": 15.0, "duration": 1.5}
    )
    traj = world.run_probe(spec)
    assert traj.channels["door_angle"].max() > 0.2
    assert "unlock" not in traj.events  # 无闩锁场景不应产生 unlock 事件


def test_reset_makes_runs_deterministic(world, library):
    spec = library.get("gentle_pull").instantiate(
        "door_0", overrides={"max_force": 15.0}
    )
    first = world.run_probe(spec)
    second = world.run_probe(spec)
    np.testing.assert_allclose(first.channels["door_angle"],
                               second.channels["door_angle"], atol=1e-9)


def test_executor_summary_on_mujoco_channels(world, library):
    executor = ProbeExecutor(feature_eps=FEATURE_EPS)
    evidence = executor.execute(
        library.get("rotate_handle").instantiate("door_0"), world
    )
    assert evidence.summary["handle_rotates"] is True
    assert evidence.summary["door_immobile"] is True


def test_create_backend_factory(tmp_path, world, library):
    traj = world.run_probe(library.get("gentle_pull").instantiate(
        "door_0", overrides={"max_force": 15.0}))
    mjcf = tmp_path / "door2.mjcf"
    mjcf.write_text(DOOR_XML, encoding="utf-8")
    spec = {"backend": "mujoco", "mjcf": str(mjcf),
            "channel_map": {"handle_angle": "handle_hinge", "door_angle": "door_hinge"}}
    factory_world = create_backend(spec)
    traj2 = factory_world.run_probe(library.get("gentle_pull").instantiate(
        "door_0", overrides={"max_force": 15.0}))
    np.testing.assert_allclose(traj.channels["door_angle"],
                               traj2.channels["door_angle"], atol=1e-9)
    with pytest.raises(ValueError, match="未知后端"):
        create_backend({"backend": "real_robot"})
