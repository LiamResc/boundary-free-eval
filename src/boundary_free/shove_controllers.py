"""The shove rule: the policy the arm world's controllers share. Static, oracle and adaptive differ only in the
friction estimate they plug into it, and each acts as if its estimate were true (certainty equivalence).

Whenever the block is still and more than TOLERANCE from the target:
  1. push from the side away from the target;
  2. release at the speed that block_travel says moves the block exactly the distance to the target, with the
     friction estimate: v = sqrt(2 (D + STANDOFF) / (1 / A_PUSH + 1 / (friction x g)));
  3. get the ball ready beside the block, shove, and wait for the block to stop again.
A wrong estimate leaves the carried part alone and makes the slide (estimate / true friction) times the planned
one, so the block undershoots or overshoots, and the rule shoves again: a correction shove, from the same side
after an undershoot, from the other side (over the block) after an overshoot.

Mass plays no part: the arm carries the block whatever it weighs, and a free slide's length does not depend on it.
"""
import math
from dataclasses import dataclass

from boundary_free.block_world import GRAVITY
from boundary_free.controllers import NOMINAL_FRICTION
from boundary_free.motion import A_PUSH, STANDOFF, PusherMotion, Shove, trajectory_to_ready

TOLERANCE = 0.01    # within this of the target counts as there, metres
TOP_SPEED = 1.0     # the fastest release, m/s; a longer move takes more than one shove
# The slowest release that still reaches the block: the ball must cover the STANDOFF while speeding up. A little
# above it, so the ball always arrives.
MIN_SPEED = 1.05 * math.sqrt(2 * A_PUSH * STANDOFF)
STILL_SPEED = 1e-3  # below this speed reading (m/s), the block counts as still...
STILL_TIME = 0.1    # ...once it has stayed below it this long, seconds


def release_speed(distance, friction):
    """The release speed that moves the block `distance` metres with this friction: block_travel solved for v."""
    return math.sqrt(2 * (distance + STANDOFF) / (1 / A_PUSH + 1 / (friction * GRAVITY)))


@dataclass(frozen=True)
class ShoveDecision:
    """One decision to shove, as the controller saw it."""
    time: float              # seconds
    block_position: float    # where the block was, along its track, metres
    block_speed: float       # its speed reading then, m/s (always below STILL_SPEED: it decides only when still)
    target: float            # where it was meant to go, metres
    direction: int           # +1 pushes along the track's +y, -1 along -y
    release_speed: float     # m/s
    friction_belief: float   # the estimate it planned with


class ShoveController:
    """The shove rule, steering the arm through its motion layer. Call update(target) once per step."""

    def __init__(self, world, friction_belief, tolerance=TOLERANCE, top_speed=TOP_SPEED):
        self.world = world
        self.motion = PusherMotion(world)
        self.friction_belief = friction_belief
        self.tolerance, self.top_speed = tolerance, top_speed
        self.phase = "waiting"      # waiting -> getting ready -> shoving -> waiting
        self.decisions = []         # every ShoveDecision, in order
        self._still_since = None
        self._plan = None

    def belief(self):
        """The friction estimate to plan with. Static keeps its own; the oracle and adaptive override this."""
        return self.friction_belief

    def update(self, target):
        """Decide, or carry on with the current shove, then give the arm this step's motor commands."""
        world, motion = self.world, self.motion

        # 1. Waiting: once the block has been still for STILL_TIME, decide whether, where and how hard to shove.
        if self.phase == "waiting" and self._block_has_settled():
            miss = target - world.position
            if abs(miss) > self.tolerance:
                self._decide(target, miss)
                motion.follow(trajectory_to_ready(world.tip_position, world.position, self._plan.direction))
                self.phase = "getting ready"

        # 2. Getting ready: when the ball reaches the ready point, shove.
        elif self.phase == "getting ready" and motion.done:
            motion.follow(Shove(world.tip_position, self._plan.direction, self._plan.release_speed))
            self.phase = "shoving"

        # 3. Shoving: when the ball has braked to rest, wait for the block to stop sliding.
        elif self.phase == "shoving" and motion.done:
            self.phase, self._still_since = "waiting", None

        # Steer the arm (before the first shove it simply holds its pose).
        if motion.trajectory is not None:
            motion.update()

    def _decide(self, target, miss):
        friction = self.belief()
        speed = min(self.top_speed, max(MIN_SPEED, release_speed(abs(miss), friction)))
        self._plan = ShoveDecision(time=self.world.time, block_position=self.world.position,
                                   block_speed=self.world.velocity, target=target, direction=1 if miss > 0 else -1,
                                   release_speed=speed, friction_belief=friction)
        self.decisions.append(self._plan)

    def _block_has_settled(self):
        """Whether the block's speed reading has stayed below STILL_SPEED for STILL_TIME."""
        if abs(self.world.velocity) >= STILL_SPEED:
            self._still_since = None
            return False
        if self._still_since is None:
            self._still_since = self.world.time
        return self.world.time - self._still_since >= STILL_TIME - 1e-9


def static_shover(world, **settings):
    """Always believes the nominal friction (0.3), like a system set up once and never updated."""
    return ShoveController(world, friction_belief=NOMINAL_FRICTION, **settings)


class OracleShover(ShoveController):
    """Told the true friction at every decision. No real robot could be; it is the yardstick (and the deviation)."""

    def __init__(self, world, **settings):
        super().__init__(world, friction_belief=world.friction, **settings)

    def belief(self):
        self.friction_belief = self.world.friction  # the cheat
        return self.friction_belief
