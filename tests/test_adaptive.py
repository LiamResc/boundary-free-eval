"""Checks the adaptive controller: it learns only while the block slides, and what that buys it on a stream."""
import pytest

from boundary_free.block_world import BlockWorld
from boundary_free.controllers import adaptive_controller, static_controller
from boundary_free.regret import measure_regret
from boundary_free.simulate import drive
from boundary_free.stream import Change, Schedule, Stream, example_schedule, run, targets_reached


def test_it_learns_the_true_friction_and_mass_during_one_push():
    world = BlockWorld(friction=0.45, mass=1.6)
    world.data.qpos[0] = -0.3
    controller = adaptive_controller(world)
    drive(world, controller, target=0.3, seconds=4.0)
    assert controller.friction_belief == pytest.approx(0.45, abs=0.01)
    assert controller.mass_belief == pytest.approx(1.6, abs=0.01)
    assert abs(0.3 - world.position) < 1e-3


def test_a_stalled_block_teaches_it_nothing():
    # The stage B check: 10 cm short and still, it pushes 5.44 N, under the 5.89 N that friction 0.6 holds with.
    # Nothing slides, so nothing is learned, and it stays stuck just like static.
    world = BlockWorld(friction=0.6, mass=1.0)
    world.data.qpos[0] = 0.2
    controller = adaptive_controller(world)
    drive(world, controller, target=0.3, seconds=2.0)
    assert controller.estimator.points_learned == 0
    assert world.position == pytest.approx(0.2, abs=1e-3)


def test_it_reaches_every_target_on_the_example_stream():
    schedule = example_schedule()
    stream = Stream(schedule)
    record = run(stream, adaptive_controller(stream.world), schedule.length)
    assert all(targets_reached(record, schedule))


def test_its_regret_after_a_change_is_far_below_static():
    schedule = Schedule(targets=(0.2, -0.3, 0.4, -0.1), target_every=6.0, friction_start=0.3, mass_start=1.0,
                        friction_changes=(Change(at=9.0, to=0.6),))
    static = measure_regret(schedule, lambda world: static_controller(), check_every=1.0)
    adaptive = measure_regret(schedule, adaptive_controller, check_every=1.0)
    assert static.rate > 0.03
    assert adaptive.rate < static.rate / 10
