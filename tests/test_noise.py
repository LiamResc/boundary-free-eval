"""Checks the sensor noise: its size, that a branch sees the same noise, and that 50 ms stretches learn through it."""
import numpy as np
import pytest

from boundary_free.controllers import adaptive_controller, static_controller
from boundary_free.regret import measure_regret
from boundary_free.stream import Change, Schedule, Stream, run

TARGETS = (0.2, -0.3, 0.4, -0.1, 0.3)  # 5 targets, 6 s apart: a 30 s stream


def schedule(speed_noise, friction=0.3, mass=1.0, friction_changes=()):
    return Schedule(targets=TARGETS, target_every=6.0, friction_start=friction, mass_start=mass,
                    friction_changes=friction_changes, speed_noise=speed_noise)


def test_with_no_noise_the_speed_reading_is_exact():
    stream = Stream(schedule(0.0))
    controller = static_controller()
    for _ in range(2000):
        seen = stream.observe()
        assert seen.velocity == stream.world.velocity
        stream.step(controller.push(seen.position, seen.velocity, seen.target))


def test_the_noise_is_the_size_asked_for():
    stream = Stream(schedule(0.01))
    errors = []
    for _ in range(10000):
        errors.append(stream.observe().velocity - stream.world.velocity)
        stream.step(0.0)
    assert np.std(errors) == pytest.approx(0.01, rel=0.03)
    assert abs(np.mean(errors)) < 0.0005


def test_a_branch_sees_the_same_noise_as_the_main_run():
    stream = Stream(schedule(0.01))
    run(stream, static_controller(), 10.0)
    twin = stream.branch()
    for _ in range(500):
        assert twin.observe() == stream.observe()
        stream.step(1.0)
        twin.step(1.0)


def test_with_shared_noise_static_still_has_exactly_zero_regret_before_a_change():
    # Static and the oracle believe the same until 9 s, and read the same noisy speeds, so they push identically.
    record = measure_regret(schedule(0.01, friction_changes=(Change(at=9.0, to=0.6),)),
                            lambda world: static_controller(), check_every=1.0)
    by_time = dict(zip(record.times, record.regret))
    assert all(by_time[t] == 0.0 for t in (0.0, 1.0, 2.0, 3.0, 4.0))


def test_learning_from_50_ms_stretches_finds_friction_and_mass_through_noise():
    stream = Stream(schedule(0.005, friction=0.45, mass=1.6))
    controller = adaptive_controller(stream.world, forgetting=0.999)
    run(stream, controller, 30.0)
    assert controller.friction_belief == pytest.approx(0.45, abs=0.05)
    assert controller.mass_belief == pytest.approx(1.6, abs=0.1)


def test_learning_from_single_steps_is_ruined_by_noise():
    # Noise in each 2 ms acceleration smears the points sideways and flattens the fitted line: far too light a mass.
    stream = Stream(schedule(0.005, friction=0.45, mass=1.6))
    controller = adaptive_controller(stream.world, forgetting=0.999, window=1)
    run(stream, controller, 30.0)
    assert controller.mass_belief < 0.8
