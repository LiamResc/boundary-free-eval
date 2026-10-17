"""Checks that the simulated block follows push = mass x acceleration + friction x mass x g."""
import pytest

from boundary_free.block_world import GRAVITY, BlockWorld


def run(world, force, seconds):
    """Push with a constant force for a while."""
    world.push(force)
    end = world.time + seconds
    while world.time < end - 1e-9:
        world.step()


def test_block_sticks_when_the_push_is_below_the_friction_limit():
    world = BlockWorld(friction=0.3, mass=1.0)
    run(world, 0.9 * world.friction_limit, seconds=2.0)
    assert abs(world.position) < 0.5e-3  # less than half a millimetre of creep in two seconds


def test_sliding_acceleration_matches_the_equation():
    world = BlockWorld(friction=0.3, mass=1.5)
    force = 2 * world.friction_limit
    run(world, force, seconds=0.3)
    expected = (force - world.friction_limit) / world.mass
    assert world.velocity / 0.3 == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("mass", [0.5, 2.0])
def test_stopping_distance_is_the_same_for_any_mass(mass):
    world = BlockWorld(friction=0.3, mass=mass)
    world.data.qvel[0] = 1.0  # set it sliding at 1 m/s, then let friction stop it
    run(world, 0.0, seconds=1.0)
    assert world.position == pytest.approx(1.0**2 / (2 * 0.3 * GRAVITY), rel=0.01)
    assert abs(world.velocity) < 1e-3


def test_changing_the_mass_mid_run_changes_friction_and_inertia_without_moving_the_block():
    world = BlockWorld(friction=0.3, mass=1.0)
    run(world, 8.0, seconds=0.3)
    position, velocity = world.position, world.velocity

    world.set_mass(2.0)
    assert (world.position, world.velocity) == (position, velocity)  # nothing jumps

    run(world, 8.0, seconds=0.1)
    expected = (8.0 - 0.3 * 2.0 * GRAVITY) / 2.0
    assert (world.velocity - velocity) / 0.1 == pytest.approx(expected, rel=0.01)


def test_changing_the_friction_mid_run():
    world = BlockWorld(friction=0.3, mass=1.0)
    run(world, 8.0, seconds=0.3)
    velocity = world.velocity

    world.set_friction(0.6)
    run(world, 8.0, seconds=0.1)
    expected = (8.0 - 0.6 * 1.0 * GRAVITY) / 1.0
    assert (world.velocity - velocity) / 0.1 == pytest.approx(expected, rel=0.01)


def test_friction_matches_the_push_while_stuck_and_is_the_limit_while_sliding():
    world = BlockWorld(friction=0.3, mass=1.0)
    run(world, 2.0, seconds=0.5)  # under the 2.94 N limit: stuck
    assert world.friction_force == pytest.approx(-2.0, abs=1e-3)

    run(world, 6.0, seconds=0.3)  # over the limit: sliding towards +x
    assert world.friction_force == pytest.approx(-world.friction_limit, rel=1e-3)


def test_push_is_capped_at_the_limit():
    world = BlockWorld(friction=0.3, mass=1.0)
    run(world, 1000.0, seconds=0.1)
    assert world.applied_push == pytest.approx(world.push_limit)
