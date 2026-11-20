"""Checks on the arm world's scene: the arm holds still, the block behaves exactly as in the block world, and only
the intended things touch."""
import mujoco
import numpy as np
import pytest

from boundary_free.arm_world import ARM, BLOCK, TABLE, TRACK_AXIS, ArmWorld, ROD_LENGTH, BALL_RADIUS
from boundary_free.block_world import GRAVITY


def run(world, seconds, push=0.0):
    """Step for a while, holding the arm at home, with a steady push on the block along its track."""
    end = world.time + seconds
    while world.time < end - 1e-9:
        world.data.xfrc_applied[world.block_body, :3] = push * TRACK_AXIS
        world.step()


def touch(model, geom1, geom2):
    """Whether MuJoCo lets two shapes touch: one's contype must share a bit with the other's conaffinity."""
    a, b = model.geom(geom1), model.geom(geom2)
    return bool((a.contype & b.conaffinity) or (b.contype & a.conaffinity))


def test_the_arm_holds_its_home_pose():
    world = ArmWorld()
    tip = world.tip_position
    run(world, seconds=2.0)
    assert np.abs(world.joint_angles - world.home).max() < 1e-3   # radians
    assert np.linalg.norm(world.tip_position - tip) < 1e-3        # metres


def test_gravity_compensation_comes_out_of_the_motors():
    world = ArmWorld()
    arm_joints = [world.model.joint(f"fr3_joint{i}").id for i in range(1, 8)]
    assert all(world.model.jnt_actgravcomp[j] for j in arm_joints)
    assert world.model.ngravcomp > 0
    assert world.model.body_gravcomp[world.block_body] == 0  # the block is held up by its track, not compensated


def test_the_pusher_points_down_at_home():
    world = ArmWorld()
    flange = world.data.site("attachment_site").xpos
    assert world.tip_position == pytest.approx(flange - [0, 0, ROD_LENGTH + BALL_RADIUS], abs=1e-6)


def test_nothing_touches_anything_at_home():
    world = ArmWorld()
    run(world, seconds=0.5)
    assert world.data.ncon == 0


def test_what_may_touch_what():
    model = ArmWorld().model
    assert touch(model, "pusher_ball", "block")
    assert touch(model, "pusher_ball", "table")
    assert touch(model, "fr3_link7_collision", "table")
    assert touch(model, "fr3_link5_collision", "block")
    assert not touch(model, "block", "table")                        # the track's friction is the only friction
    assert not touch(model, "fr3_link7_collision", "fr3_link5_collision")  # the arm does not touch itself
    assert not touch(model, "floor", "block")
    assert (model.geom("table").contype, model.geom("block").contype) == (TABLE, BLOCK)
    assert model.geom("pusher_ball").contype == ARM


def reach(world, point):
    """Joint angles that would put the pusher's ball at `point`: damped least squares on a copy of the arm, geometry
    only. A stand-in for the motion layer of stage G2, just for these tests."""
    model, scratch = world.model, mujoco.MjData(world.model)
    scratch.qpos[:] = world.data.qpos
    tip, jacobian = model.site("pusher_tip").id, np.zeros((3, model.nv))
    for _ in range(200):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        mujoco.mj_jacSite(model, scratch, jacobian, None, tip)
        arm = jacobian[:, :7]
        miss = point - scratch.site_xpos[tip]
        scratch.qpos[:7] += arm.T @ np.linalg.solve(arm @ arm.T + 1e-4 * np.eye(3), miss)
    return scratch.qpos[:7].copy()


def contacts(world):
    """The pairs of shapes touching on the last step, by name."""
    return {tuple(sorted((world.model.geom(c.geom1).name, world.model.geom(c.geom2).name)))
            for c in world.data.contact[:world.data.ncon]}


def test_the_table_stops_the_arm():
    world = ArmWorld()
    world.command(reach(world, np.array([0.35, 0.3, BALL_RADIUS - 0.03])))  # aim the ball 3 cm into the table
    run(world, seconds=2.0)
    assert world.tip_position[2] > BALL_RADIUS - 0.002  # it stops on the table top, give or take a soft contact
    assert ("pusher_ball", "table") in contacts(world)


def test_the_pusher_shoves_the_block_along_its_track():
    world = ArmWorld()
    world.command(reach(world, np.array([0.5, -0.2, 0.05])))  # the ball at block height, left of the block
    run(world, seconds=2.0)
    assert world.position == pytest.approx(0.0, abs=1e-4)   # not touched yet

    world.command(reach(world, np.array([0.5, 0.1, 0.05])))   # sweep through where the block is
    seen = set()
    for _ in range(500):
        world.step()
        seen |= contacts(world)
    assert ("block", "pusher_ball") in seen
    assert world.position > 0.05                             # shoved along its track, away from the pusher


def test_the_block_sticks_below_its_friction_limit():
    world = ArmWorld(friction=0.3, mass=1.0)
    run(world, seconds=1.0, push=0.9 * world.friction_limit)
    assert abs(world.position) < 0.5e-3


def test_the_block_slides_as_in_the_block_world():
    world = ArmWorld(friction=0.3, mass=1.5)
    push = 2 * world.friction_limit
    run(world, seconds=0.3, push=push)
    assert world.velocity / 0.3 == pytest.approx((push - world.friction_limit) / world.mass, rel=0.01)
    assert world.friction_force == pytest.approx(-world.friction_limit, rel=1e-3)
    assert world.data.ncon == 0  # sliding, it still touches nothing


@pytest.mark.parametrize("mass", [0.5, 2.0])
def test_stopping_distance_is_the_same_for_any_mass(mass):
    world = ArmWorld(friction=0.3, mass=mass)
    world.data.qvel[world.model.joint("track").dofadr[0]] = 1.0  # set it sliding at 1 m/s, then let friction stop it
    run(world, seconds=1.0)
    assert world.position == pytest.approx(1.0**2 / (2 * 0.3 * GRAVITY), rel=0.01)
    assert abs(world.velocity) < 1e-3


def test_changing_the_knobs_mid_run_moves_nothing():
    world = ArmWorld(friction=0.3, mass=1.0)
    run(world, seconds=0.3, push=8.0)
    before = (world.position, world.velocity, world.joint_angles.tolist())
    world.set_mass(2.0)
    world.set_friction(0.6)
    assert (world.position, world.velocity, world.joint_angles.tolist()) == before

    velocity = world.velocity
    run(world, seconds=0.1, push=16.0)
    expected = (16.0 - 0.6 * 2.0 * GRAVITY) / 2.0
    assert (world.velocity - velocity) / 0.1 == pytest.approx(expected, rel=0.01)
