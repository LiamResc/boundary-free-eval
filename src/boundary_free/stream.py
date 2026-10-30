"""The stream: one long run with no resets. Targets arrive on a clock; friction and mass change on a hidden schedule.

The controller is told only the block's position, its speed and the current target. It is never told the
friction or mass, or when they change. The speed reading can carry sensor noise, also written into the schedule.
"""
from dataclasses import dataclass, field

import numpy as np

from boundary_free.block_world import BlockWorld


# ---- 1. The schedule: the script for the stream --------------------------------

# The simulated clock adds up 0.002 s steps, so 30 s can come out as 29.999999999998. Every lookup by time adds
# this tiny allowance first, so that a new target or a hidden change never arrives one step late.
CLOCK_ALLOWANCE = 1e-9


@dataclass(frozen=True)
class Change:
    """One change to a hidden knob: starting at `at` seconds, move to `to`, taking `over` seconds."""
    at: float
    to: float
    over: float = 0.0  # 0 is a sudden jump; more than 0 is a steady drift


def value_at(start, changes, time):
    """The value of a hidden knob at a given time: the start value, with the changes applied in order."""
    value = start
    for change in changes:
        if time < change.at:
            break
        if change.over > 0 and time < change.at + change.over:  # partway through a drift
            return value + (change.to - value) * (time - change.at) / change.over
        value = change.to
    return value


@dataclass(frozen=True)
class Schedule:
    """Everything that happens in a stream, looked up as a rule of time, so a branch sees the same future.

    A recurring condition is a change back to an earlier value; a compound one is changes to both knobs at once.
    """
    targets: tuple             # target positions along the track, in metres, one per interval
    target_every: float        # seconds between new targets; a new one arrives whether or not the last was reached
    friction_start: float
    mass_start: float
    friction_changes: tuple = ()
    mass_changes: tuple = ()
    speed_noise: float = 0.0   # how far off a speed reading typically is, in m/s (the noise's standard deviation)
    noise_seed: int = 0        # which noise: the same seed always gives the same noise

    def target_index(self, time):
        """Which target is current at a given time: 0 for the first (the last one stays once the list runs out)."""
        return min(int((time + CLOCK_ALLOWANCE) // self.target_every), len(self.targets) - 1)

    def target(self, time):
        """The target position at a given time."""
        return self.targets[self.target_index(time)]

    def friction(self, time):
        return value_at(self.friction_start, self.friction_changes, time + CLOCK_ALLOWANCE)

    def mass(self, time):
        return value_at(self.mass_start, self.mass_changes, time + CLOCK_ALLOWANCE)

    @property
    def length(self):
        """How long the stream runs: one interval per target, in seconds."""
        return len(self.targets) * self.target_every

    def speed_noise_table(self, timestep):
        """The error added to the speed reading at each step, written in advance from the seed. Every stream built
        from this schedule, main run or branch, makes the same table, so a branch sees exactly the noise the main
        run sees from that moment on (common random numbers)."""
        steps = round(self.length / timestep) + 1
        if self.speed_noise == 0:
            return np.zeros(steps)
        return np.random.default_rng(self.noise_seed).normal(0.0, self.speed_noise, steps)


def example_schedule(seed=0, speed_noise=0.0):
    """The stream used in the playground demos: 20 targets, 6 s apart, 120 s in all.

    Friction jumps from 0.3 to 0.6 at 39 s and comes back at 81 s (a recurring condition). Mass drifts from
    1.0 to 1.5 kg between 63 and 75 s. The changes start and end partway through pushes, never on a new
    target, so a new target is never a hint that something changed.
    """
    rng = np.random.default_rng(seed)
    return Schedule(
        targets=tuple(float(x) for x in np.round(rng.uniform(-0.6, 0.6, size=20), 2)),
        target_every=6.0,
        friction_start=0.3,
        mass_start=1.0,
        friction_changes=(Change(at=39.0, to=0.6), Change(at=81.0, to=0.3)),
        mass_changes=(Change(at=63.0, to=1.5, over=12.0),),
        speed_noise=speed_noise,
        noise_seed=seed,
    )


# ---- 2. The stream: the block world, driven by the schedule ----------------------

@dataclass(frozen=True)
class Observation:
    """Everything a controller is told: nothing about friction or mass."""
    position: float
    velocity: float
    target: float


class Stream:
    """A block world whose target and hidden knobs follow a schedule."""

    def __init__(self, schedule, world=None, start_position=0.0):
        self.schedule = schedule
        if world is None:
            world = BlockWorld(friction=schedule.friction(0.0), mass=schedule.mass(0.0))
            world.data.qpos[0] = start_position
        self.world = world
        self.speed_noise = schedule.speed_noise_table(world.timestep)

    @property
    def time(self):
        return self.world.time

    def observe(self):
        """What the controller sees now. The speed reading carries this step's noise; position is read exactly."""
        world = self.world
        step = min(round(world.time / world.timestep), len(self.speed_noise) - 1)
        return Observation(world.position, world.velocity + self.speed_noise[step], self.schedule.target(world.time))

    def step(self, push):
        """Apply a push for one timestep, then set the hidden knobs to what the schedule says for the next one.

        So the world always holds the values for the step about to be taken, and the oracle, which reads them
        while working out its push, is never a step behind a change.
        """
        world = self.world
        world.push(push)
        world.step()
        friction, mass = self.schedule.friction(world.time), self.schedule.mass(world.time)
        if friction != world.friction:
            world.set_friction(friction)
        if mass != world.mass:
            world.set_mass(mass)

    def branch(self):
        """A copy of this moment for a "what if" run. The schedule, noise included, is a rule of time, so it is
        simply shared."""
        return Stream(self.schedule, world=self.world.branch())


# ---- 3. Running a controller on a stream ------------------------------------------

@dataclass
class StreamRecord:
    """What happened on each step. friction and mass are recorded for analysis only; controllers never see them."""
    time: list = field(default_factory=list)
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    seen_velocity: list = field(default_factory=list)  # the speed reading the controller pushed on, noise and all
    target: list = field(default_factory=list)
    target_index: list = field(default_factory=list)  # which target was current: 0 for the first
    push: list = field(default_factory=list)
    friction: list = field(default_factory=list)
    mass: list = field(default_factory=list)


def run(stream, controller, seconds, on_step=None):
    """Let a controller work the stream for a while, recording each step."""
    record = StreamRecord()
    end = stream.time + seconds
    while stream.time < end - 1e-9:
        target_index = stream.schedule.target_index(stream.time)
        friction, mass = stream.world.friction, stream.world.mass  # the hidden values during this step
        seen = stream.observe()
        stream.step(controller.push(seen.position, seen.velocity, seen.target))

        world = stream.world
        record.time.append(world.time)
        record.position.append(world.position)
        record.velocity.append(world.velocity)
        record.seen_velocity.append(seen.velocity)
        record.target.append(seen.target)
        record.target_index.append(target_index)
        record.push.append(world.applied_push)
        record.friction.append(friction)
        record.mass.append(mass)
        if on_step is not None:
            on_step()
    return record


def targets_reached(record, schedule, within=0.01):
    """For each target, whether the block got within `within` metres of it before the next target arrived.

    This scores every target as if it were a separate task: the "episodic" view of the stream.
    """
    index, position, target = map(np.array, (record.target_index, record.position, record.target))
    reached = []
    for k in range(len(schedule.targets)):
        during = index == k
        if during.any():
            reached.append(bool(np.any(np.abs(position[during] - target[during]) < within)))
    return reached
