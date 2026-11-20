"""Checks on moving the pusher: smooth straight-line references, tracking them closely thanks to velocity
feedforward, getting behind the block without touching it, and staying inside the real FR3's joint speed limits."""
import numpy as np
import pytest

from boundary_free.arm_world import BALL_RADIUS, BLOCK_HALF_SIZE, ArmWorld
from boundary_free.motion import (CLEARANCE, JOINT_SPEED_LIMITS, PusherMotion, Segment, Trajectory, ready_point,
                                  trajectory_to_ready)


def follow(world, motion, trajectory, settle=0.5):
    """Follow a trajectory, then hold its end for `settle` seconds. Returns the errors, ball positions, joint speeds and
    whether the block was ever touched."""
    motion.follow(trajectory)
    errors, tips, speeds, touched = [], [], [], False
    end = world.time + trajectory.duration + settle
    while world.time < end - 1e-9:
        errors.append(np.linalg.norm(motion.update()))
        world.step()
        tips.append(world.tip_position)
        speeds.append(np.abs(world.joint_velocities))
        touched |= any(world.model.geom(c.geom1).name == "block" or world.model.geom(c.geom2).name == "block"
                       for c in world.data.contact[:world.data.ncon])
    return np.array(errors), np.array(tips), np.array(speeds), touched


def test_a_segment_starts_and_ends_at_rest_and_keeps_to_its_top_speed():
    segment = Segment(np.zeros(3), np.array([0.0, 0.3, 0.0]), top_speed=0.3, acceleration=1.0)
    assert segment.duration == pytest.approx(2 * 0.3 + (0.3 - 0.09) / 0.3)  # 0.3 s up, 0.7 s cruising, 0.3 s down
    start, v0 = segment.reference(0.0)
    goal, v1 = segment.reference(segment.duration)
    assert np.allclose(start, segment.start) and np.allclose(goal, segment.goal)
    assert np.allclose(v0, 0) and np.allclose(v1, 0)
    times = np.linspace(0, segment.duration, 200)
    along = [segment.reference(t)[0][1] for t in times]
    assert np.all(np.diff(along) >= 0)
    assert max(np.linalg.norm(segment.reference(t)[1]) for t in times) <= 0.3 + 1e-12


def test_a_short_segment_never_reaches_top_speed():
    segment = Segment(np.zeros(3), np.array([0.04, 0.0, 0.0]), top_speed=0.3, acceleration=1.0)
    assert segment.peak_speed == pytest.approx(0.2)    # sqrt(0.04 x 1)
    assert segment.duration == pytest.approx(0.4)      # 0.2 s up, 0.2 s down


def test_the_ball_follows_a_straight_line_and_stops_on_the_goal():
    world = ArmWorld()
    motion = PusherMotion(world)
    start = world.tip_position
    goal = start + np.array([0.0, 0.2, -0.1])
    errors, tips, _, _ = follow(world, motion, Trajectory.through([start, goal]))
    direction = (goal - start) / np.linalg.norm(goal - start)
    off_line = np.linalg.norm(np.cross(tips - start, direction), axis=1)
    assert off_line.max() < 2e-3                        # metres
    assert np.linalg.norm(tips[-1] - goal) < 0.5e-3


def test_velocity_feedforward_removes_the_lag():
    def cruising_error(lead):
        world = ArmWorld()
        motion = PusherMotion(world)
        if lead is not None:
            motion.lead = np.full(7, lead)
        start = world.tip_position
        trajectory = Trajectory.through([start, start + np.array([0.0, 0.3, 0.0])], top_speed=0.3, acceleration=1.0)
        errors, _, _, _ = follow(world, motion, trajectory, settle=0.0)
        times = np.arange(len(errors)) * world.timestep
        return errors[(times > 0.4) & (times < 0.9)].mean()  # during the cruise at 0.3 m/s

    assert cruising_error(lead=0.0) == pytest.approx(0.1 * 0.3, rel=0.15)  # trails by about 0.1 s x speed: 30 mm
    assert cruising_error(lead=None) < 2e-3


@pytest.mark.parametrize("push_direction", [+1, -1])
def test_getting_ready_to_shove_goes_over_the_block_without_touching_it(push_direction):
    world = ArmWorld()
    motion = PusherMotion(world)
    # First to the ready point on one side, then across to the other side, which means passing over the block.
    for direction in (push_direction, -push_direction):
        trajectory = trajectory_to_ready(world.tip_position, world.position, direction)
        _, tips, speeds, touched = follow(world, motion, trajectory)
        assert np.linalg.norm(tips[-1] - ready_point(world.position, direction)) < 1e-3
        assert not touched
        assert world.position == 0.0                   # the block has not moved at all
        assert (speeds < JOINT_SPEED_LIMITS).all()     # inside the real FR3's joint speed limits
    over = tips[np.abs(tips[:, 1]) < BLOCK_HALF_SIZE + BALL_RADIUS]  # while the ball was above the block's footprint
    assert over[:, 2].min() > 2 * BLOCK_HALF_SIZE + BALL_RADIUS + CLEARANCE / 2


def test_from_the_pushing_side_the_ball_goes_straight_to_the_ready_point():
    world = ArmWorld()
    tip = ready_point(0.0, +1) + np.array([0.0, -0.1, 0.05])  # clear of the block, on the side it pushes from
    trajectory = trajectory_to_ready(tip, 0.0, +1)
    assert len(trajectory.segments) == 1
