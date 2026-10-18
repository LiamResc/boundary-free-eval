"""Checks the pushing controller: the push it asks for, and what static and oracle do with it."""
import pytest

from boundary_free.block_world import GRAVITY, BlockWorld
from boundary_free.controllers import OracleController, PushController, friction_direction, static_controller
from boundary_free.simulate import drive

START, TARGET, SECONDS = -0.3, 0.3, 4.0


def from_start(friction, mass):
    world = BlockWorld(friction=friction, mass=mass)
    world.data.qpos[0] = START
    return world


def test_static_push_when_still_and_10_cm_short():
    # The stage B check question: 25 x 0.1 x 1 kg + 0.3 x 1 kg x g
    assert static_controller().push(position=0.2, velocity=0.0, target=0.3) == pytest.approx(5.443, abs=1e-3)


def test_friction_part_points_the_way_the_block_moves_or_should_move():
    assert friction_direction(velocity=0.2, desired_acceleration=-5.0) == 1.0   # moving: oppose the motion
    assert friction_direction(velocity=0.0, desired_acceleration=-5.0) == -1.0  # still: the way we want to go
    assert friction_direction(velocity=0.0, desired_acceleration=0.0) == 0.0    # still and not asking: nothing


@pytest.mark.parametrize("friction, mass", [(0.15, 1.0), (0.3, 1.0), (0.6, 1.0), (0.3, 0.5), (0.3, 1.5)])
def test_oracle_reaches_the_target_in_any_world(friction, mass):
    world = from_start(friction, mass)
    record = drive(world, OracleController(world), TARGET, SECONDS)
    reached = next(t for t, x in zip(record.time, record.position) if abs(TARGET - x) < 0.01)
    assert reached < 1.5
    assert abs(TARGET - world.position) < 1e-3


def test_static_matches_the_oracle_when_its_beliefs_are_right():
    static_world, oracle_world = from_start(0.3, 1.0), from_start(0.3, 1.0)
    drive(static_world, static_controller(), TARGET, SECONDS)
    drive(oracle_world, OracleController(oracle_world), TARGET, SECONDS)
    assert static_world.position == pytest.approx(oracle_world.position, abs=1e-9)


@pytest.mark.parametrize("stiffness", [10.0, 25.0, 100.0])
def test_static_stalls_where_the_correction_part_covers_the_shortfall(stiffness):
    world = from_start(friction=0.6, mass=1.0)
    drive(world, static_controller(stiffness=stiffness), TARGET, SECONDS)
    shortfall = (0.6 - 0.3) * 1.0 * GRAVITY      # friction the belief left out, in newtons
    expected_miss = shortfall / (1.0 * stiffness)  # where believed mass x stiffness x miss = shortfall
    assert TARGET - world.position == pytest.approx(expected_miss, rel=0.03)


def test_oracle_uses_the_true_values_even_after_they_change():
    world = from_start(0.3, 1.0)
    oracle = OracleController(world)
    world.set_friction(0.6)
    world.set_mass(2.0)
    pushed = oracle.push(position=0.2, velocity=0.0, target=0.3)
    assert pushed == pytest.approx(PushController(0.6, 2.0).push(0.2, 0.0, 0.3))
