"""Pushing controllers: one rule, with different beliefs about the block's friction and mass.

The rule, every step:
  1. desired acceleration = stiffness x miss - damping x speed
                            (correction part)    (braking part)
  2. push = believed mass x desired acceleration
          + believed friction x believed mass x g x direction    (computed push for friction)

Static and oracle use the same rule and differ only in their beliefs.
"""
import math

from boundary_free.block_world import GRAVITY

STIFFNESS = 25.0            # desired acceleration per metre of miss (per second squared)
STILL_SPEED = 1e-3          # below this speed (m/s), treat the block as still
ASKING_ACCELERATION = 0.05  # below this desired acceleration (m/s^2), treat the controller as not asking for motion


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
    return PushController(friction_belief=0.3, mass_belief=1.0, **settings)


class OracleController(PushController):
    """Told the true friction and mass at every step. No real robot could be; it is a yardstick."""

    def __init__(self, world, **settings):
        super().__init__(world.friction, world.mass, **settings)
        self.world = world

    def push(self, position, velocity, target):
        self.friction_belief, self.mass_belief = self.world.friction, self.world.mass  # the cheat
        return super().push(position, velocity, target)
