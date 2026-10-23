"""Checks the regret measurement: the scoring, and what regret says about controllers we understand."""
import math

import numpy as np
import pytest

from boundary_free.controllers import OracleController, static_controller
from boundary_free.regret import look_ahead_weights, measure_regret, score
from boundary_free.stream import Change, Schedule, Stream, run

TARGETS = (0.2, -0.3, 0.4, -0.1)  # 4 targets, 6 s apart: a 24 s stream


def schedule(friction_changes=()):
    return Schedule(targets=TARGETS, target_every=6.0, friction_start=0.3, mass_start=1.0,
                    friction_changes=friction_changes)


def static(world):
    return static_controller()


def test_look_ahead_weights_fade_and_add_up_to_one():
    weights = look_ahead_weights(timestep=0.002, look_ahead=4.0, fade_time=1.0)
    assert len(weights) == 2000
    assert math.fsum(weights) == pytest.approx(1.0)
    assert weights[500] / weights[0] == pytest.approx(math.exp(-1.0))  # 1 s ahead counts about a third
    assert np.all(np.diff(weights) < 0)


def test_a_steady_miss_scores_minus_that_miss():
    weights = look_ahead_weights(timestep=0.002)
    assert score([0.05] * len(weights), weights) == pytest.approx(-0.05)


def test_static_has_no_regret_when_its_beliefs_are_right():
    record = measure_regret(schedule(), static, check_every=1.0)
    assert all(r == 0.0 for r in record.regret)


def test_the_oracle_has_no_regret_against_itself():
    record = measure_regret(schedule((Change(at=9.0, to=0.6),)), OracleController, check_every=1.0)
    assert all(r == 0.0 for r in record.regret)


def test_static_has_regret_after_a_change_it_cannot_see():
    record = measure_regret(schedule((Change(at=9.0, to=0.6),)), static, check_every=1.0)
    by_time = dict(zip(record.times, record.regret))
    assert all(by_time[t] == 0.0 for t in (0.0, 1.0, 2.0, 3.0, 4.0))  # look-ahead still ends before 9 s
    assert all(by_time[t] > 0.05 for t in (13.0, 14.0, 15.0))         # static stalls 12 cm short of 0.4; the oracle doesn't


def test_regret_forgives_the_past():
    # Friction is wrong for static from 9 s to 15 s, then right again: from 15 s it does just what the oracle would.
    record = measure_regret(schedule((Change(at=9.0, to=0.6), Change(at=15.0, to=0.3))), static, check_every=1.0)
    by_time = dict(zip(record.times, record.regret))
    assert by_time[14.0] > 0.0
    assert all(by_time[t] == 0.0 for t in (15.0, 16.0, 17.0, 18.0, 19.0, 20.0))


def test_measuring_regret_leaves_the_main_run_untouched():
    plain = run(Stream(schedule((Change(at=9.0, to=0.6),))), static_controller(), 24.0)
    record = measure_regret(schedule((Change(at=9.0, to=0.6),)), static, check_every=1.0)
    assert record.main_run.position == plain.position
