"""Checks the stream: the schedule's values, the targets on the clock, what the controller sees, and branching."""
import pytest

from boundary_free.controllers import OracleController, static_controller
from boundary_free.stream import Change, Schedule, Stream, example_schedule, run, targets_reached, value_at


def test_a_jump_changes_the_value_at_that_moment():
    changes = (Change(at=10.0, to=0.6),)
    assert value_at(0.3, changes, 9.999) == 0.3
    assert value_at(0.3, changes, 10.0) == 0.6
    assert value_at(0.3, changes, 50.0) == 0.6


def test_a_drift_moves_steadily_from_the_old_value_to_the_new():
    changes = (Change(at=10.0, to=2.0, over=4.0),)
    assert value_at(1.0, changes, 10.0) == pytest.approx(1.0)
    assert value_at(1.0, changes, 11.0) == pytest.approx(1.25)
    assert value_at(1.0, changes, 12.0) == pytest.approx(1.5)
    assert value_at(1.0, changes, 14.0) == pytest.approx(2.0)


def test_changes_apply_in_order_and_can_return_to_an_earlier_value():
    changes = (Change(at=10.0, to=0.6), Change(at=20.0, to=0.3))
    assert [value_at(0.3, changes, t) for t in (5.0, 15.0, 25.0)] == [0.3, 0.6, 0.3]


def test_targets_change_on_the_clock_and_the_last_one_stays():
    schedule = Schedule(targets=(0.1, -0.2, 0.3), target_every=6.0, friction_start=0.3, mass_start=1.0)
    assert [schedule.target(t) for t in (0.0, 5.999, 6.0, 12.0, 17.9, 100.0)] == [0.1, 0.1, -0.2, 0.3, 0.3, 0.3]
    assert schedule.length == 18.0


def test_the_example_stream_never_starts_or_ends_a_change_on_a_new_target():
    schedule = example_schedule()
    for change in schedule.friction_changes + schedule.mass_changes:
        assert change.at % schedule.target_every != 0
        assert (change.at + change.over) % schedule.target_every != 0


def test_the_stream_sets_the_hidden_knobs_from_the_schedule():
    schedule = Schedule(targets=(0.2,), target_every=10.0, friction_start=0.3, mass_start=1.0,
                        friction_changes=(Change(at=1.0, to=0.6),), mass_changes=(Change(at=1.0, to=2.0, over=2.0),))
    stream = Stream(schedule)
    run(stream, static_controller(), 2.0)  # one second into the mass drift
    assert stream.world.friction == 0.6
    assert stream.world.mass == pytest.approx(1.5, abs=0.01)


def test_the_controller_is_told_only_position_speed_and_target():
    seen = Stream(example_schedule()).observe()
    assert set(vars(seen)) == {"position", "velocity", "target"}


def test_a_branch_sees_the_same_scheduled_future_as_the_main_run():
    stream = Stream(example_schedule())
    run(stream, static_controller(), 38.0)  # one second before friction jumps
    twin = stream.branch()
    assert run(twin, static_controller(), 4.0).position == run(stream, static_controller(), 4.0).position


def test_a_stream_run_with_branches_matches_one_without():
    expected = run(Stream(example_schedule()), static_controller(), 50.0).position

    stream = Stream(example_schedule())
    controller = static_controller()
    positions = []
    for _ in range(10):  # every 5 s, run a 2 s "what if" with the oracle on a branch
        twin = stream.branch()
        run(twin, OracleController(twin.world), 2.0)
        positions += run(stream, controller, 5.0).position
    assert positions == expected


def test_the_oracle_reaches_every_target_and_static_misses_after_the_change():
    schedule = example_schedule()
    oracle_stream = Stream(schedule)
    oracle = targets_reached(run(oracle_stream, OracleController(oracle_stream.world), schedule.length), schedule)
    static = targets_reached(run(Stream(schedule), static_controller(), schedule.length), schedule)
    assert all(oracle)
    assert all(static[:6]) and not any(static[7:])  # fine before the jump at 39 s, every target missed after it
