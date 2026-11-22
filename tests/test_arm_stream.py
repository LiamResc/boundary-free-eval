"""Checks on the arm world's snapshots, the arm stream and deviation regret on it: exact replay, branches that do
not leak, the schedule, announcements, the oracle's takeover, and regret that is exactly zero when it must be."""
import dataclasses

import numpy as np
import pytest

from boundary_free.arm_stream import ArmStream, arm_schedule, measure_arm_regret, run
from boundary_free.arm_world import ArmWorld
from boundary_free.motion import PusherMotion, Trajectory
from boundary_free.shove_controllers import AdaptiveShover, OracleShover, static_shover


def short(schedule, targets=4):
    """The same schedule, cut down to its first few targets (and the changes that fall inside them)."""
    end = targets * schedule.target_every
    return dataclasses.replace(schedule, targets=schedule.targets[:targets],
                               friction_changes=tuple(c for c in schedule.friction_changes if c.at < end))


# ---- Snapshots of the arm world -----------------------------------------------------------

def moving_world():
    """An arm world with the arm swinging and the block sliding, so there is something to replay."""
    world = ArmWorld(friction=0.35)
    motion = PusherMotion(world)
    motion.follow(Trajectory.through([world.tip_position, world.tip_position + [0.0, 0.15, -0.2]]))
    world.data.qvel[world.model.joint("track").dofadr[0]] = 0.5
    return world, motion


def advance(world, motion, steps):
    trace = []
    for _ in range(steps):
        motion.update()
        world.step()
        trace.append((world.position, *world.joint_angles))
    return np.array(trace)


def test_replay_from_a_snapshot_is_exact():
    world, motion = moving_world()
    advance(world, motion, 200)
    saved = world.snapshot()
    first = advance(world, motion, 300)
    world.restore(saved)          # the motion layer reads the world's clock, so its reference rewinds too
    second = advance(world, motion, 300)
    assert np.array_equal(first, second)


def test_a_branch_starts_at_the_same_moment_and_does_not_leak_back():
    world, motion = moving_world()
    advance(world, motion, 200)
    twin = world.branch()
    assert (twin.time, twin.position) == (world.time, world.position)
    assert np.array_equal(twin.joint_angles, world.joint_angles)
    assert np.array_equal(twin.tip_position, world.tip_position)
    twin.set_friction(0.6)
    for _ in range(100):
        twin.step()
    assert world.friction == 0.35 and world.time < twin.time


# ---- The schedule and the stream -------------------------------------------------------------

def test_the_arm_schedule():
    schedule = arm_schedule(seed=3)
    assert schedule.length == 156.0
    assert all(abs(t) <= 0.3 for t in schedule.targets)
    assert [schedule.friction(t) for t in (10, 50, 69, 78, 90, 130)] == pytest.approx([0.3, 0.45, 0.5, 0.55, 0.45, 0.35])
    assert min(schedule.friction(t) for t in np.arange(39.5, 156, 0.5)) >= 0.35   # never back to static's 0.3
    moments = schedule.change_moments()
    assert moments == (39.0, 63.0, 75.0, 81.0, 111.0)
    assert all(m % schedule.target_every != 0 for m in moments)                    # never on a target switch


def test_the_stream_sets_the_hidden_friction_from_the_schedule():
    schedule = short(arm_schedule(seed=1), targets=8)
    stream = ArmStream(schedule)
    record = run(stream, static_shover(stream.world, sensor=stream.sensor()), 48.0)
    times, frictions = np.array(record.time), np.array(record.friction)
    assert np.all(frictions[times < 39.0 - 0.003] == 0.3) and np.all(frictions[times > 39.0 + 0.003] == 0.45)


@pytest.mark.parametrize("told", [True, False])
def test_only_the_told_stream_announces_each_change_once(told):
    class Listener(OracleShover):
        heard = []
        def told_of_change(self):
            self.heard.append(round(self.world.time, 3))
    Listener.heard = []
    schedule = arm_schedule(seed=2)
    stream = ArmStream(schedule, told=told)
    run(stream, Listener(stream.world, sensor=stream.sensor()), schedule.length)
    assert Listener.heard == ([39.0, 63.0, 75.0, 81.0, 111.0] if told else [])


# ---- Deviation regret on the arm --------------------------------------------------------------

def test_a_controller_taking_over_from_itself_has_exactly_zero_regret():
    schedule = short(arm_schedule(seed=0, speed_noise=0.01, position_noise=0.002), targets=8)
    record = measure_arm_regret(schedule, lambda w, s: static_shover(w, sensor=s),
                                make_deviation=lambda w, s: static_shover(w, sensor=s), check_every=1.0)
    assert len(record.regret) > 30
    assert all(r == 0.0 for r in record.regret)


def test_static_has_exactly_zero_regret_while_its_belief_is_true():
    schedule = short(arm_schedule(seed=0, speed_noise=0.01), targets=6)   # 36 s: friction 0.3 throughout
    record = measure_arm_regret(schedule, lambda w, s: static_shover(w, sensor=s), check_every=1.0)
    assert all(r == 0.0 for r in record.regret)


def test_measuring_regret_leaves_the_main_run_untouched():
    schedule = short(arm_schedule(seed=4, speed_noise=0.01), targets=8)
    make = lambda w, s: AdaptiveShover(w, trigger="self-detected", sensor=s)
    measured = measure_arm_regret(schedule, make, check_every=2.0).main_run
    stream = ArmStream(schedule)
    plain = run(stream, make(stream.world, stream.sensor()), schedule.length)
    assert np.array_equal(measured.position, plain.position)


def test_after_the_jump_static_has_regret_and_the_oracle_does_better():
    schedule = short(arm_schedule(seed=5), targets=10)   # 60 s, with the jump to 0.45 at 39 s
    record = measure_arm_regret(schedule, lambda w, s: static_shover(w, sensor=s), check_every=1.0)
    times, regret = np.array(record.times), np.array(record.regret)
    assert np.all(regret[times < 35.0] == 0.0)            # before the jump's effect can reach the look-ahead
    assert regret[times >= 39.0].mean() > 0.0
