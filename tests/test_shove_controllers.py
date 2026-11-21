"""Checks on the shove rule: the release-speed formula, one shove for the oracle, correction shoves for a wrong belief
(same side after an undershoot, the other side after an overshoot), and deciding only when the block is still."""
import pytest

from boundary_free.arm_world import ArmWorld
from boundary_free.motion import block_travel
from boundary_free.shove_controllers import (TOLERANCE, TOP_SPEED, OracleShover, ShoveController, release_speed,
                                             static_shover)


def run(controller, target, seconds):
    """Give the controller a target and let it work for a while; `target` may be a function of time."""
    world = controller.world
    end = world.time + seconds
    while world.time < end - 1e-9:
        controller.update(target(world.time) if callable(target) else target)
        world.step()


def block_after(controller, decision_index):
    """Where the block was when the controller made its next decision (or is now, if there was none)."""
    decisions = controller.decisions
    return decisions[decision_index + 1].block_position if decision_index + 1 < len(decisions) \
        else controller.world.position


@pytest.mark.parametrize("distance", [0.02, 0.1, 0.25])
@pytest.mark.parametrize("friction", [0.3, 0.45, 0.6])
def test_release_speed_is_block_travel_turned_around(distance, friction):
    assert block_travel(release_speed(distance, friction), friction) == pytest.approx(distance)


def test_the_worked_examples():
    assert release_speed(0.10, 0.35) == pytest.approx(0.594, abs=1e-3)
    assert release_speed(0.20, 0.35) == pytest.approx(0.820, abs=1e-3)


def test_the_oracle_gets_there_in_one_shove():
    controller = OracleShover(ArmWorld(friction=0.45))
    run(controller, target=0.2, seconds=5.0)
    assert len(controller.decisions) == 1
    assert abs(controller.world.position - 0.2) < TOLERANCE


def test_believing_friction_weaker_than_it_is_undershoots_then_corrects_from_the_same_side():
    controller = ShoveController(ArmWorld(friction=0.6), friction_belief=0.35)
    run(controller, target=0.2, seconds=6.0)
    first = controller.decisions[0]
    assert block_after(controller, 0) == pytest.approx(block_travel(first.release_speed, 0.6), abs=4e-3)
    assert block_after(controller, 0) < 0.2 - TOLERANCE                  # short of the target
    assert controller.decisions[1].direction == first.direction           # pushed again the same way
    assert abs(controller.world.position - 0.2) <= TOLERANCE


def test_believing_friction_stronger_than_it_is_overshoots_then_corrects_from_the_other_side():
    controller = ShoveController(ArmWorld(friction=0.25), friction_belief=0.35)
    run(controller, target=0.2, seconds=8.0)
    assert block_after(controller, 0) > 0.2 + TOLERANCE                  # past the target
    assert controller.decisions[1].direction == -controller.decisions[0].direction  # pushed back
    assert abs(controller.world.position - 0.2) <= TOLERANCE


def test_within_tolerance_it_does_nothing():
    controller = static_shover(ArmWorld())
    run(controller, target=0.5 * TOLERANCE, seconds=1.0)
    assert controller.decisions == []


def test_it_decides_only_when_the_block_is_still_even_if_the_target_moves_mid_shove():
    controller = OracleShover(ArmWorld(friction=0.45))
    run(controller, target=lambda t: 0.2 if t < 2.4 else -0.1, seconds=9.0)  # the target jumps during the first shove
    assert len(controller.decisions) >= 2
    assert all(abs(d.block_speed) < controller.still_speed for d in controller.decisions)
    assert abs(controller.world.position - (-0.1)) <= TOLERANCE


def test_the_oracle_plans_with_the_true_friction_at_each_decision():
    world = ArmWorld(friction=0.45)
    controller = OracleShover(world)
    run(controller, target=0.15, seconds=4.0)
    world.set_friction(0.6)
    run(controller, target=-0.05, seconds=6.0)
    assert [d.friction_belief for d in controller.decisions] == [0.45, 0.6]


def test_a_long_move_takes_more_than_one_shove():
    controller = OracleShover(ArmWorld(friction=0.6))
    target = -0.4                                    # further than one shove at TOP_SPEED can move it (24 cm)
    run(controller, target=target, seconds=9.0)
    assert controller.decisions[0].release_speed == TOP_SPEED
    assert len(controller.decisions) >= 2
    assert abs(controller.world.position - target) <= TOLERANCE
