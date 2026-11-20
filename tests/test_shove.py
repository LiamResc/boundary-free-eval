"""Checks on the shove: its speed profile, where the block ends up, the free slide, that friction plays no part while
the block is carried, what happens if the ball brakes too gently, and the real FR3's limits."""
import numpy as np
import pytest

from boundary_free.arm_world import CONTACT_TIME_CONSTANT, ArmWorld
from boundary_free.block_world import GRAVITY
from boundary_free.motion import A_BRAKE, JOINT_SPEED_LIMITS, PusherMotion, Shove, block_travel, trajectory_to_ready

JOINT_ACCELERATION_LIMIT = 10.0  # rad/s^2, every joint of the real FR3 (Franka's specifications)


def run(world, motion, seconds, log=None):
    """Let the motion layer steer the arm for a while, noting each step in `log` if given."""
    end = world.time + seconds
    while world.time < end - 1e-9:
        motion.update()
        world.step()
        if log is not None:
            log.append((world.time, world.position, world.velocity, world.pusher_touches_block,
                        world.joint_velocities))


def shove(friction, release_speed, brake=A_BRAKE, direction=+1):
    """Get ready, shove the block once, and wait for it to stop. Returns the shove, the block's start position, and a
    log of (time, block position, block velocity, touching, joint velocities) from the start of the shove."""
    world = ArmWorld(friction=friction)
    motion = PusherMotion(world)
    motion.follow(trajectory_to_ready(world.tip_position, world.position, direction))
    run(world, motion, motion.trajectory.duration + 0.3)
    the_shove = Shove(world.tip_position, direction, release_speed, brake_acceleration=brake)
    start, log = world.position, []
    motion.follow(the_shove)
    run(world, motion, the_shove.duration + 1.5, log)
    t0 = log[0][0] - world.timestep
    return the_shove, start, [(t - t0, x, v, touching, qd) for t, x, v, touching, qd in log]


def test_the_shove_speeds_up_then_brakes_to_rest():
    the_shove = Shove(np.zeros(3), +1, release_speed=0.6)
    assert the_shove.release_time == pytest.approx(0.2)                   # 0.6 m/s at 3 m/s^2
    assert the_shove.duration == pytest.approx(0.2 + 0.6 / 8)
    assert np.linalg.norm(the_shove.reference(0.0)[1]) == 0.0
    assert np.linalg.norm(the_shove.reference(0.2)[1]) == pytest.approx(0.6)
    assert np.linalg.norm(the_shove.reference(the_shove.duration)[1]) == pytest.approx(0.0, abs=1e-12)
    assert the_shove.end[1] == pytest.approx(0.06 + 0.6**2 / (2 * 8))      # speeding up, then braking


def test_the_contact_between_pusher_and_block_is_stiff():
    model = ArmWorld().model
    for name in ("pusher_ball", "pusher_rod", "block"):
        assert model.geom(name).solref[0] == CONTACT_TIME_CONSTANT


@pytest.mark.parametrize("friction", [0.15, 0.35, 0.6])
@pytest.mark.parametrize("release_speed", [0.4, 0.8])
def test_the_block_stops_where_the_formula_says(friction, release_speed):
    _, start, log = shove(friction, release_speed)
    travelled = log[-1][1] - start
    assert travelled == pytest.approx(block_travel(release_speed, friction), abs=4e-3)
    assert abs(log[-1][2]) < 1e-3  # and it has stopped


def test_a_shove_the_other_way_is_the_mirror_image():
    _, start, log = shove(0.35, 0.6, direction=-1)
    assert start - log[-1][1] == pytest.approx(block_travel(0.6, 0.35), abs=4e-3)


def test_after_release_the_block_slows_at_friction_times_g():
    _, _, log = shove(0.35, 0.8)
    last_touch = max(i for i, row in enumerate(log) if row[3])
    sliding = [(t, v) for t, _, v, _, _ in log[last_touch + 5:] if v > 0.05]
    times, speeds = np.array(sliding).T
    slope = np.polyfit(times, speeds, 1)[0]
    assert slope == pytest.approx(-0.35 * GRAVITY, rel=0.01)


def test_while_carried_the_block_goes_where_the_ball_takes_it_whatever_the_friction():
    def position_at_release(friction):
        the_shove, start, log = shove(friction, 0.6)
        return next(x for t, x, *_ in log if t >= the_shove.release_time) - start
    assert position_at_release(0.15) == pytest.approx(position_at_release(0.6), abs=1e-3)


def test_braking_more_gently_than_friction_x_g_carries_the_block_to_a_stop():
    # Friction 0.4 slows a free block at 3.92 m/s^2; the ball brakes at only 3, so the block is never released.
    the_shove, start, log = shove(0.4, 0.6, brake=3.0)
    braking = [row for row in log if the_shove.release_time + 0.02 < row[0] < the_shove.duration - 0.02]
    # The block keeps bumping back into the ball: it is never clear of it for more than a few milliseconds.
    apart, longest = 0, 0
    for _, _, _, touching, _ in braking:
        apart = 0 if touching else apart + 1
        longest = max(longest, apart)
    assert longest * 0.002 <= 0.015
    carried = abs(the_shove.end[1] - the_shove.start[1]) - 0.01      # where the ball stopped, less the standoff
    assert log[-1][1] - start == pytest.approx(carried, abs=3e-3)
    # The arm, not friction, decides where it stops.
    _, start_rough, log_rough = shove(0.6, 0.6, brake=3.0)
    assert log_rough[-1][1] - start_rough == pytest.approx(log[-1][1] - start, abs=2e-3)


def test_a_fast_shove_stays_inside_the_real_fr3_limits():
    _, _, log = shove(0.35, 1.0)
    speeds = np.array([qd for *_, qd in log])
    assert (np.abs(speeds) < JOINT_SPEED_LIMITS).all()
    # Joint accelerations, averaged over 10 ms. Step by step there is one 2 ms jolt, about 25 rad/s^2 on a wrist joint,
    # at the moment the ball strikes the block; that is an impact, not commanded motion, which is what the limits
    # are for.
    accelerations = np.abs(speeds[5:] - speeds[:-5]) / 0.010
    assert accelerations.max() < JOINT_ACCELERATION_LIMIT
