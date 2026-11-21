"""Checks on the three triggers for becoming unsure: the CUSUM, the reading noise it divides by, who listens to an
announcement, and the self-detected version catching a change on the arm."""
import numpy as np
import pytest

from boundary_free.arm_world import ArmWorld
from boundary_free.detection import Cusum
from boundary_free.sensing import BlockSensor
from boundary_free.shove_controllers import AdaptiveShover, OracleShover, reading_noise, static_shover


# ---- The CUSUM ------------------------------------------------------------------------

def test_cusum_rarely_alarms_on_pure_noise():
    detector, rng = Cusum(), np.random.default_rng(0)
    alarms = sum(detector.check(e) for e in rng.standard_normal(10_000))
    assert alarms <= 40      # about 2 per 1000 readings at k = 0.5, h = 5 (playground 16)


@pytest.mark.parametrize("shift", [+3.0, -3.0])
def test_cusum_catches_a_shift_either_way_within_a_few_readings(shift):
    detector, rng = Cusum(), np.random.default_rng(1)
    for e in rng.standard_normal(200):
        detector.check(0.0 * e)   # a quiet start
    waited = next(i + 1 for i, e in enumerate(rng.standard_normal(100) + shift) if detector.check(e))
    assert waited <= 4


def test_after_an_alarm_both_statistics_restart():
    detector = Cusum()
    assert detector.check(10.0)
    assert (detector.upper, detector.lower) == (0.0, 0.0)


def test_reading_noise_comes_from_the_sensor_specification():
    assert reading_noise(0.01, 25, 0.002) == pytest.approx(np.sqrt(2) * 0.01 / (9.81 * 0.05))   # 0.029
    assert reading_noise(0.0, 25, 0.002) == 0.005                                                 # the floor


# ---- Who listens to an announcement ---------------------------------------------------------

def test_only_the_told_version_becomes_unsure_when_told():
    for trigger, listens in (("told", True), ("forgetting only", False), ("self-detected", False)):
        controller = AdaptiveShover(ArmWorld(), trigger=trigger)
        for _ in range(50):
            controller.estimator.learn(0.3)
        settled = controller.estimator.uncertainty
        controller.told_of_change()
        assert (controller.estimator.uncertainty == controller.estimator.start_uncertainty) == listens
        assert (controller.estimator.uncertainty == settled) == (not listens)


def test_static_and_the_oracle_ignore_announcements():
    for controller in (static_shover(ArmWorld()), OracleShover(ArmWorld())):
        controller.told_of_change()          # does nothing, and does not fail
        assert controller.belief() == pytest.approx(0.3)


def test_an_unknown_trigger_is_refused():
    with pytest.raises(ValueError):
        AdaptiveShover(ArmWorld(), trigger="psychic")


# ---- The self-detected version ------------------------------------------------------------------

def test_on_an_alarm_it_becomes_unsure_before_learning_the_reading_that_raised_it():
    world = ArmWorld()
    controller = AdaptiveShover(world, trigger="self-detected", sensor=BlockSensor(world, speed_noise=0.01))
    for _ in range(50):
        controller._use_reading(0.35)
    controller._use_reading(0.6)            # (0.6 - 0.35) / 0.029 = 8.7 noise widths: an alarm at once
    assert [cause for _, cause in controller.resets] == ["alarm"]
    # Unsure first: the reading moves the estimate 92% of the way (learning first would move it only 10%: 0.375).
    assert controller.estimator.friction == pytest.approx(0.35 + 0.917 * 0.25, abs=0.005)


def test_forgetting_only_never_becomes_unsure():
    controller = AdaptiveShover(ArmWorld(), trigger="forgetting only")
    for reading in [0.35] * 50 + [0.9] * 10:
        controller._use_reading(reading)
    assert controller.resets == []


def test_self_detected_catches_a_friction_jump_on_the_arm_and_relearns_quickly():
    world = ArmWorld(friction=0.35)
    controller = AdaptiveShover(world, trigger="self-detected", sensor=BlockSensor(world, speed_noise=0.01, seed=21))
    targets = np.round(np.random.default_rng(6).uniform(-0.3, 0.3, 16), 2)
    while world.time < 96.0 - 1e-9:
        if world.time >= 63.0 - 1e-9 and world.friction == 0.35:
            world.set_friction(0.5)            # unannounced
        controller.update(targets[min(int(world.time // 6.0), 15)])
        world.step()
    alarms = [t for t, cause in controller.resets if cause == "alarm"]
    assert not [t for t in alarms if t < 63.0]                     # no false alarm before the change
    assert alarms and alarms[0] - 63.0 < 6.0                       # caught on the first slide after it
    after = [d.friction_belief for d in controller.decisions if d.time > alarms[0]]
    assert abs(after[0] / 0.5 - 1) < 0.08                          # the very next shove plans with nearly 0.5
