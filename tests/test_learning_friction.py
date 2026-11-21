"""Checks on learning friction from free slides: the one-unknown estimator, the noisy sensor, and the adaptive shove
controller that reads friction from its own slides and plans with it."""
import numpy as np
import pytest

from boundary_free.arm_world import ArmWorld
from boundary_free.estimator import FrictionEstimator
from boundary_free.sensing import BlockSensor
from boundary_free.shove_controllers import AdaptiveShover


def run(controller, targets, every, on_step=None):
    """Work through the targets, one every `every` seconds."""
    world = controller.world
    while world.time < every * len(targets) - 1e-9:
        decided = len(controller.decisions)
        controller.update(targets[min(int(world.time // every), len(targets) - 1)])
        if on_step is not None:
            on_step(len(controller.decisions) > decided)
        world.step()


def random_targets(seed, count=20):
    return np.round(np.random.default_rng(seed).uniform(-0.3, 0.3, count), 2)


# ---- The estimator -----------------------------------------------------------------

def test_a_steady_reading_is_learned():
    estimator = FrictionEstimator(0.3, forgetting=0.9)
    for _ in range(30):
        estimator.learn(0.45)
    assert estimator.friction == pytest.approx(0.45, abs=1e-4)  # the starting guess has all but faded


def test_with_forgetting_the_estimate_closes_a_tenth_of_the_distance_each_reading():
    estimator = FrictionEstimator(0.3, forgetting=0.9)
    for _ in range(100):
        estimator.learn(0.35)
    for _ in range(10):
        estimator.learn(0.5)
    # Once settled, each reading moves it (1 - forgetting) = 10% of the way, so 0.9^10 of the gap remains.
    assert estimator.friction == pytest.approx(0.5 - 0.15 * 0.9**10, abs=2e-3)


def test_becoming_unsure_lets_the_next_reading_almost_replace_the_estimate():
    estimator = FrictionEstimator(0.3, forgetting=0.9)
    for _ in range(100):
        estimator.learn(0.35)
    estimator.become_unsure()
    estimator.learn(0.5)
    assert estimator.friction > 0.485   # 92% of the way in one reading, against 10% without the reset


def test_the_estimate_stays_in_a_sensible_range():
    estimator = FrictionEstimator(0.3)
    for _ in range(20):
        estimator.learn(-1.0)
    assert estimator.friction == 0.05


# ---- The sensor ----------------------------------------------------------------------

def test_without_noise_the_readings_are_exact():
    world = ArmWorld()
    world.data.qvel[world.model.joint("track").dofadr[0]] = 0.3
    sensor = BlockSensor(world)
    assert (sensor.position(), sensor.speed()) == (world.position, world.velocity)


def test_the_noise_has_the_requested_size_and_position_and_speed_noise_are_independent():
    world = ArmWorld()
    sensor = BlockSensor(world, position_noise=0.002, speed_noise=0.01, seed=3)
    errors = []
    for _ in range(5000):
        errors.append((sensor.position() - world.position, sensor.speed() - world.velocity))
        world.step()
    errors = np.array(errors)
    assert errors[:, 0].std() == pytest.approx(0.002, rel=0.05)
    assert errors[:, 1].std() == pytest.approx(0.01, rel=0.05)
    assert abs(np.corrcoef(errors.T)[0, 1]) < 0.05


def test_the_same_seed_gives_the_same_noise_at_the_same_moment():
    main, branch = ArmWorld(), ArmWorld()
    for _ in range(123):
        main.step()
        branch.step()
    same, other = BlockSensor(main, 0.002, 0.01, seed=7), BlockSensor(branch, 0.002, 0.01, seed=7)
    assert (same.position(), same.speed()) == (other.position(), other.speed())
    assert BlockSensor(branch, 0.002, 0.01, seed=8).speed() != same.speed()


# ---- The adaptive controller ---------------------------------------------------------------

def test_with_exact_sensing_every_reading_is_the_true_friction_and_it_plans_with_it():
    controller = AdaptiveShover(ArmWorld(friction=0.45))
    run(controller, random_targets(1, count=6), every=6.0)
    readings = np.array([r for _, r in controller.readings])
    assert len(readings) >= 10
    assert np.abs(readings / 0.45 - 1).max() < 0.005       # never a stretch from the push, which would read nonsense
    assert controller.decisions[0].friction_belief == 0.3  # it starts from the nominal value...
    assert controller.decisions[-1].friction_belief == pytest.approx(0.45, rel=0.005)  # ...and plans with what it learned


@pytest.mark.parametrize("noise", [0.005, 0.01, 0.02])
def test_with_noisy_speed_readings_the_readings_are_unbiased(noise):
    world = ArmWorld(friction=0.35)
    controller = AdaptiveShover(world, sensor=BlockSensor(world, speed_noise=noise, seed=11))
    run(controller, random_targets(2), every=6.0)
    readings = np.array([r for _, r in controller.readings])
    standard_error = readings.std() / np.sqrt(len(readings))
    assert abs(readings.mean() - 0.35) < max(3 * standard_error, 0.004)
    assert readings.std() < 3.5 * noise     # about sqrt(2) x noise / (g x 50 ms) = 2.9 x noise per reading


def test_forgetting_alone_catches_up_slowly_after_a_jump():
    world = ArmWorld(friction=0.35)
    controller = AdaptiveShover(world)
    targets = random_targets(4, count=16)

    def jump(_):
        if world.time >= 48.0 and world.friction == 0.35:
            world.set_friction(0.5)
    run(controller, targets, every=6.0, on_step=jump)
    after = [d.friction_belief for d in controller.decisions if d.time > 48.0]
    assert after[1] < 0.47          # two shoves after the jump it has not caught up...
    assert after[-1] > 0.48         # ...but it gets there


def test_with_position_noise_it_decides_from_averaged_readings():
    world = ArmWorld(friction=0.45)
    controller = AdaptiveShover(world, sensor=BlockSensor(world, position_noise=0.003, seed=5))
    truths = []
    run(controller, random_targets(3, count=8), every=6.0,
        on_step=lambda decided: truths.append(world.position) if decided else None)
    seen = np.array([d.block_position for d in controller.decisions])
    # One reading is off by 3 mm or so; averaged over the 0.1 s still spell (50 readings), about 0.4 mm.
    assert np.abs(seen - np.array(truths)).max() < 1.5e-3
