"""Pushing controllers: one rule, with different beliefs about the block's friction and mass.

The rule, every step:
  1. desired acceleration = stiffness x miss - damping x speed
                            (correction part)    (braking part)
  2. push = believed mass x desired acceleration
          + believed friction x believed mass x g x direction    (computed push for friction)

Static, oracle and adaptive use the same rule and differ only in their beliefs.
"""
import math

from boundary_free.block_world import GRAVITY
from boundary_free.estimator import FORGETTING, FrictionMassEstimator

STIFFNESS = 25.0            # desired acceleration per metre of miss (per second squared)
STILL_SPEED = 1e-3          # below this speed (m/s), treat the block as still
ASKING_ACCELERATION = 0.05  # below this desired acceleration (m/s^2), treat the controller as not asking for motion
SLIDING_SPEED = 0.01        # above this speed (m/s), the block is surely sliding, so friction is friction x mass x g
WINDOW = 25                 # sliding steps per learning point: 50 ms, which makes speed-reading noise matter 25x less
NOMINAL_FRICTION, NOMINAL_MASS = 0.3, 1.0  # the values a system set up once would assume


class PushController:
    """Pushes the block towards a target, using its beliefs about the block's friction and mass."""

    def __init__(self, friction_belief, mass_belief, stiffness=STIFFNESS, damping=None):
        self.friction_belief = friction_belief
        self.mass_belief = mass_belief
        self.stiffness = stiffness
        # Unless told otherwise, brake just enough that the block never overshoots the target:
        # 2 x the square root of the stiffness (engineers call this critical damping). For 25, that is 10.
        self.damping = 2 * math.sqrt(stiffness) if damping is None else damping

    def push(self, position, velocity, target):
        """The push to apply now, in newtons, from what a real robot could observe."""
        # 1. The acceleration we want: correct the miss, and brake against the speed.
        desired = self.stiffness * (target - position) - self.damping * velocity

        # 2. The push that would give that acceleration, if our beliefs are right.
        direction = friction_direction(velocity, desired)
        return (self.mass_belief * desired
                + self.friction_belief * self.mass_belief * GRAVITY * direction)


def friction_direction(velocity, desired_acceleration):
    """Which way the computed push for friction should point: +1, -1, or 0 for none.

    Friction opposes motion, so the push against it points the way the block is moving. If the block is
    still, it points the way we want it to go. If we are not asking for motion, there is nothing to beat,
    and pushing anyway would shove the block back and forth at the target.
    """
    if abs(velocity) > STILL_SPEED:
        return math.copysign(1.0, velocity)
    if abs(desired_acceleration) > ASKING_ACCELERATION:
        return math.copysign(1.0, desired_acceleration)
    return 0.0


def static_controller(**settings):
    """Always believes friction 0.3 and mass 1 kg, like a system set up once and never updated."""
    return PushController(friction_belief=NOMINAL_FRICTION, mass_belief=NOMINAL_MASS, **settings)


class OracleController(PushController):
    """Told the true friction and mass at every step. No real robot could be; it is a yardstick."""

    def __init__(self, world, **settings):
        super().__init__(world.friction, world.mass, **settings)
        self.world = world

    def push(self, position, velocity, target):
        self.friction_belief, self.mass_belief = self.world.friction, self.world.mass  # the cheat
        return super().push(position, velocity, target)


class AdaptiveController(PushController):
    """Works its beliefs out from what it sees: it learns from what its pushes did while the block slid.

    It starts from static's beliefs and is told nothing about the true friction and mass, or when they change.
    It learns from stretches of `window` sliding steps: the average push over the stretch against the speed
    change across it. Only the two speed readings at the ends carry noise, so a longer stretch divides that noise
    by a longer time. window=1 learns from every single step, as in stage F part 1.
    """

    def __init__(self, timestep, push_limit, forgetting=FORGETTING, window=WINDOW, **settings):
        super().__init__(friction_belief=NOMINAL_FRICTION, mass_belief=NOMINAL_MASS, **settings)
        # Forgetting is counted per step, so one learning point, a stretch of `window` steps, forgets that many.
        self.estimator = FrictionMassEstimator(NOMINAL_FRICTION, NOMINAL_MASS, forgetting ** window)
        self.timestep = timestep
        self.push_limit = push_limit
        self.window = window
        self.last = None            # (speed reading, push) at the previous step: the memory that learning needs
        self.stretch_pushes = []    # the pushes so far in the current stretch of sliding one way
        self.stretch_start = None   # the speed reading at the start of that stretch
        self.last_point = None      # the latest (acceleration, average push) it learned from, for plotting

    def push(self, position, velocity, target):
        # 1. Learn from what its pushes did. Collect `window` steps of the block sliding one way, then learn from
        #    their average push and the speed change across them. A step that is not sliding breaks the stretch.
        if self.last is not None:
            last_velocity, last_push = self.last
            if sliding_same_way(last_velocity, velocity):
                if not self.stretch_pushes:
                    self.stretch_start = last_velocity
                self.stretch_pushes.append(last_push)
                if len(self.stretch_pushes) == self.window:
                    acceleration = (velocity - self.stretch_start) / (self.window * self.timestep)
                    average_push = sum(self.stretch_pushes) / self.window
                    self.estimator.learn(acceleration, math.copysign(1.0, velocity), average_push)
                    self.friction_belief, self.mass_belief = self.estimator.friction, self.estimator.mass
                    self.last_point = (acceleration, average_push)
                    self.stretch_pushes = []
            else:
                self.stretch_pushes = []

        # 2. Push using the current beliefs, capped at what the pusher can give, so it knows what was applied.
        push = super().push(position, velocity, target)
        push = max(-self.push_limit, min(self.push_limit, push))
        self.last = (velocity, push)
        return push


def sliding_same_way(speed_before, speed_after):
    """Whether the block was clearly sliding, in one direction, for a whole step. Only then is the friction
    exactly friction x mass x g: a still block's friction just matches the push, whatever the push is."""
    return (abs(speed_before) > SLIDING_SPEED and abs(speed_after) > SLIDING_SPEED
            and (speed_before > 0) == (speed_after > 0))


def adaptive_controller(world, **settings):
    """An adaptive controller for this world. It reads only the timestep and the push limit, which a real
    robot knows about itself; never the friction or mass."""
    return AdaptiveController(timestep=world.timestep, push_limit=world.push_limit, **settings)
