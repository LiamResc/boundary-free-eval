"""The arm world's stream, and deviation regret on it: stages D and E, with the FR3 shoving the block.

The stream is the block world's idea with the arm doing the work: targets arrive on a clock as instructions, the
block's friction follows a hidden schedule written as a rule of time, nothing resets, and only the told condition
hears when something changed. Regret is measured as in stage E: at each checkpoint the arm world is branched, the
oracle takes over (it inherits whatever the controller is in the middle of and makes every decision from then on
with the true friction), and its next few seconds are scored against what the controller actually did.
"""
from dataclasses import dataclass, field

import numpy as np

from boundary_free.arm_world import ArmWorld
from boundary_free.regret import CHECK_EVERY, FADE_TIME, LOOK_AHEAD, RegretRecord, look_ahead_weights, score
from boundary_free.sensing import BlockSensor
from boundary_free.shove_controllers import OracleShover
from boundary_free.stream import CLOCK_ALLOWANCE, Change, Schedule

# ---- 1. The schedule -----------------------------------------------------------------

ARM_TARGET_RANGE = 0.3   # targets fall within this of the track's middle, metres (one shove moves 24-30 cm at most)


def arm_schedule(seed=0, speed_noise=0.0, position_noise=0.0):
    """The arm world's stream: 26 targets, 6 s apart (156 s), and a hidden friction schedule.

    Friction starts at static's 0.3, jumps to 0.45 at 39 s, drifts to 0.55 over 63-75 s, comes back to 0.45 at 81 s
    (a recurring context, but never static's value again) and drops to 0.35 at 111 s. Every value stays between 0.3
    and 0.55: the smallest possible move fits the 1 cm tolerance, and braking always releases the block. No change
    starts or ends on a target switch, so a new target is never a hint. Mass stays at 1 kg: it cannot matter here.
    """
    rng = np.random.default_rng(seed)
    return Schedule(
        targets=tuple(float(x) for x in np.round(rng.uniform(-ARM_TARGET_RANGE, ARM_TARGET_RANGE, 26), 2)),
        target_every=6.0,
        friction_start=0.3,
        mass_start=1.0,
        friction_changes=(Change(at=39.0, to=0.45), Change(at=63.0, to=0.55, over=12.0), Change(at=81.0, to=0.45),
                          Change(at=111.0, to=0.35)),
        speed_noise=speed_noise,
        position_noise=position_noise,
        noise_seed=seed,
    )


# ---- 2. The stream: the arm world, driven by the schedule ------------------------------

class ArmStream:
    """An arm world whose target and hidden friction follow a schedule. told=True is the boundary-informed
    condition: at each of the schedule's change moments the controller is told that something just changed."""

    def __init__(self, schedule, world=None, told=False):
        self.schedule = schedule
        self.world = ArmWorld(friction=schedule.friction(0.0), mass=schedule.mass_start) if world is None else world
        self.told = told
        self._moments = schedule.change_moments()

    @property
    def time(self):
        return self.world.time

    def sensor(self):
        """A sensor on this stream's world with the schedule's noise and seed: a branch's sensor gives the same
        readings at the same moments as the main run's (common random numbers)."""
        return BlockSensor(self.world, position_noise=self.schedule.position_noise,
                           speed_noise=self.schedule.speed_noise, seed=self.schedule.noise_seed)

    def step(self, controller):
        """One step: announce (told only), let the controller act, step the physics, then set the hidden friction
        for the next step. Returns the target the controller was given."""
        # Announced in the step whose span (now - timestep, now] holds a change moment: once each, and the same on a
        # branch, since it depends only on the clock. The allowance shifts the whole span, as in stage F.
        now = self.world.time + CLOCK_ALLOWANCE
        if self.told and any(now - self.world.timestep < m <= now for m in self._moments):
            controller.told_of_change()
        target = self.schedule.target(self.world.time)
        controller.update(target)
        self.world.step()
        friction = self.schedule.friction(self.world.time)
        if friction != self.world.friction:
            self.world.set_friction(friction)
        return target

    def branch(self):
        """A copy of the stream at this moment, with its own world; the schedule is shared."""
        return ArmStream(self.schedule, world=self.world.branch(), told=self.told)


# ---- 3. Running and recording -------------------------------------------------------------

@dataclass
class ArmRecord:
    """What happened, step by step."""
    time: list = field(default_factory=list)
    position: list = field(default_factory=list)   # the block's true position, metres
    target: list = field(default_factory=list)
    friction: list = field(default_factory=list)   # the true friction during the step
    belief: list = field(default_factory=list)     # the controller's friction estimate

    def misses(self):
        """Distance from the target after each step, metres."""
        return np.abs(np.array(self.position) - np.array(self.target))

    def extend(self, other):
        for name in ("time", "position", "target", "friction", "belief"):
            getattr(self, name).extend(getattr(other, name))


def run(stream, controller, seconds):
    """Let a controller work the stream for a while, recording each step."""
    record = ArmRecord()
    end = stream.time + seconds
    while stream.time < end - 1e-9:
        friction = stream.world.friction
        target = stream.step(controller)
        record.time.append(stream.time)
        record.position.append(stream.world.position)
        record.target.append(target)
        record.friction.append(friction)
        record.belief.append(controller.friction_belief)
    return record


# ---- 4. Deviation regret ----------------------------------------------------------------

def measure_arm_regret(schedule, make_controller, make_deviation=None, told=False, look_ahead=LOOK_AHEAD,
                       fade_time=FADE_TIME, check_every=CHECK_EVERY):
    """Run a controller through a whole arm stream, measuring its regret against the oracle at every checkpoint.

    make_controller(world, sensor) returns the controller to measure; make_deviation(world, sensor) returns the
    alternative to compare against on a branch (by default the oracle, with the same noisy sensor: only the friction
    knowledge differs). The deviation takes over from the controller, inheriting whatever it is in the middle of.
    """
    if make_deviation is None:
        def make_deviation(world, sensor):
            return OracleShover(world, sensor=sensor)
    stream = ArmStream(schedule, told=told)
    controller = make_controller(stream.world, stream.sensor())
    weights = look_ahead_weights(stream.world.timestep, look_ahead, fade_time)

    # 1. Run the main stream. At each checkpoint, branch it, let the deviation take over, and score its look-ahead.
    main_run = ArmRecord()
    times, starts, deviation_score = [], [], []
    last_checkpoint = schedule.length - look_ahead
    while stream.time <= last_checkpoint + 1e-9:
        times.append(round(stream.time, 6))
        starts.append(len(main_run.time))
        branch = stream.branch()
        deviation = make_deviation(branch.world, branch.sensor()).take_over_from(controller)
        deviation_score.append(score(run(branch, deviation, look_ahead).misses(), weights))
        main_run.extend(run(stream, controller, check_every))

    # 2. Carry the main run on to the end, so the last checkpoint's look-ahead has happened too.
    main_run.extend(run(stream, controller, schedule.length - stream.time))

    # 3. Score the controller's own future from each checkpoint, and compare.
    misses, steps = main_run.misses(), len(weights)
    controller_score = [score(misses[start:start + steps], weights) for start in starts]
    regret = [d - c for d, c in zip(deviation_score, controller_score)]
    return RegretRecord(times, controller_score, deviation_score, regret, main_run)
