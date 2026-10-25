"""Checks the estimator: it finds friction and mass from clean sliding moments, and what forgetting does."""
import numpy as np
import pytest

from boundary_free.block_world import GRAVITY
from boundary_free.estimator import FrictionMassEstimator

ACCELERATIONS = np.linspace(-8.0, 12.0, 41)  # a spread of accelerations, like speeding up and braking between targets


def moments(friction, mass, direction=1.0, accelerations=ACCELERATIONS):
    """Clean sliding moments, (acceleration, direction, push), exactly on the true line."""
    return [(a, direction, mass * a + friction * mass * GRAVITY * direction) for a in accelerations]


def learn_all(estimator, sliding_moments):
    for acceleration, direction, push in sliding_moments:
        estimator.learn(acceleration, direction, push)


@pytest.mark.parametrize("friction, mass", [(0.3, 1.0), (0.6, 1.0), (0.15, 0.5), (0.45, 1.6)])
def test_it_finds_friction_and_mass_from_clean_sliding_moments(friction, mass):
    estimator = FrictionMassEstimator(friction_guess=0.3, mass_guess=1.0)
    learn_all(estimator, moments(friction, mass, +1.0) + moments(friction, mass, -1.0))
    assert estimator.friction == pytest.approx(friction, abs=1e-3)
    assert estimator.mass == pytest.approx(mass, abs=1e-3)


def test_moments_at_one_acceleration_cannot_tell_friction_from_mass():
    # Sliding at a steady speed: friction 0.3 with 2 kg needs the same push as friction 0.6 with 1 kg.
    estimator = FrictionMassEstimator(friction_guess=0.3, mass_guess=1.0)
    learn_all(estimator, moments(0.3, 2.0, accelerations=[0.0] * 200))
    assert estimator.estimate[1] == pytest.approx(0.6, abs=1e-3)  # friction x mass: learned
    assert estimator.mass == 1.0                                   # mass: never moved from its guess
    assert estimator.friction == pytest.approx(0.6, abs=1e-3)     # so the friction it hands over is wrong
    assert estimator.uncertainty[0, 0] > 7 * 10.0                 # and it grows ever less sure of the mass: windup


def test_without_forgetting_the_fit_blends_old_and_new():
    # Two thirds of the moments come from before friction doubled, so the fit lands a third of the way.
    estimator = FrictionMassEstimator(friction_guess=0.3, mass_guess=1.0, forgetting=1.0)
    learn_all(estimator, moments(0.3, 1.0) * 2 + moments(0.6, 1.0))
    assert estimator.friction == pytest.approx(0.4, abs=2e-3)


def test_with_forgetting_the_fit_follows_a_change():
    estimator = FrictionMassEstimator(friction_guess=0.3, mass_guess=1.0, forgetting=0.99)
    learn_all(estimator, moments(0.3, 1.0) * 10 + moments(0.6, 1.0) * 25)  # 1025 new moments: ten memories' worth
    assert estimator.friction == pytest.approx(0.6, abs=1e-3)
    assert estimator.points_learned == 35 * len(ACCELERATIONS)
