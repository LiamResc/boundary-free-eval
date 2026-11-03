"""Checks the told condition: when changes are announced, what the adaptive controller does with an announcement,
and that nothing else changes."""
import numpy as np
import pytest

from boundary_free.controllers import OracleController, adaptive_controller, static_controller
from boundary_free.estimator import START_UNCERTAINTY, FrictionMassEstimator
from boundary_free.stream import Stream, example_schedule, run


def announced_times(stream, seconds):
    """Step the stream with no push, noting the time of every announcement."""
    times = []
    for _ in range(round(seconds / stream.world.timestep)):
        if stream.observe().change_announced:
            times.append(stream.time)
        stream.step(0.0)
    return times


def test_nothing_is_announced_unless_told():
    assert announced_times(Stream(example_schedule()), 120.0) == []


def test_told_hears_every_jump_and_the_start_and_end_of_the_drift_once():
    times = announced_times(Stream(example_schedule(), told=True), 120.0)
    assert times == pytest.approx([39.0, 63.0, 75.0, 81.0], abs=0.003)


def test_a_drift_can_be_announced_every_2_s_instead():
    schedule = example_schedule()
    assert schedule.change_moments(drift_every=2.0) == (39.0, 63.0, 65.0, 67.0, 69.0, 71.0, 73.0, 75.0, 81.0)


def test_a_branch_of_a_told_stream_hears_the_same_announcements_ahead():
    stream = Stream(example_schedule(), told=True)
    run(stream, static_controller(), 38.0)
    assert announced_times(stream.branch(), 4.0) == pytest.approx([39.0], abs=0.003)


def test_an_announcement_makes_adaptive_unsure_again():
    stream = Stream(example_schedule(), told=True)
    controller = adaptive_controller(stream.world)
    run(stream, controller, 38.99)                      # it has learned from many moves by now
    before = controller.estimator.uncertainty.copy()
    run(stream, controller, 0.02)                       # the 39 s announcement arrives (the block is resting)
    assert before[1, 1] < START_UNCERTAINTY / 100       # it was sure of friction x mass ...
    assert np.array_equal(controller.estimator.uncertainty, np.eye(2) * START_UNCERTAINTY)  # ... and now is not


def test_static_and_the_oracle_ignore_announcements():
    for make in (lambda world: static_controller(), OracleController):
        plain, told = Stream(example_schedule()), Stream(example_schedule(), told=True)
        a = run(plain, make(plain.world), 50.0).position
        b = run(told, make(told.world), 50.0).position
        assert a == b


def test_the_mass_belief_never_falls_below_0_3_kg():
    # Points that fit a 0.1 kg block: the belief handed to the controller stops at the floor.
    estimator = FrictionMassEstimator(friction_guess=0.3, mass_guess=1.0)
    for a in np.linspace(-5.0, 5.0, 200):
        estimator.learn(a, 1.0, 0.1 * a + 0.3 * 0.1 * 9.81)
    assert estimator.estimate[0] == pytest.approx(0.1, abs=0.01)
    assert estimator.mass == 0.3
