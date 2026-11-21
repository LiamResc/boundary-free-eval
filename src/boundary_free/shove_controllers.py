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

The controllers see the block only through a sensor: noisy readings of its position and speed. The arm's own
position comes from its joint encoders, exactly. Mass plays no part: the arm carries the block whatever it weighs,
and a free slide's length does not depend on it.
"""
import math
from dataclasses import dataclass

from boundary_free.block_world import GRAVITY
from boundary_free.controllers import NOMINAL_FRICTION, SLIDING_SPEED, WINDOW, still_speed_for
from boundary_free.detection import Cusum
from boundary_free.estimator import FRICTION_FORGETTING, FrictionEstimator
from boundary_free.motion import A_PUSH, STANDOFF, PusherMotion, Shove, trajectory_to_ready
from boundary_free.sensing import BlockSensor

TOLERANCE = 0.01    # within this of the target counts as there, metres
TOP_SPEED = 1.0     # the fastest release, m/s; a longer move takes more than one shove
# The slowest release that still reaches the block: the ball must cover the STANDOFF while speeding up. A little
# above it, so the ball always arrives.
MIN_SPEED = 1.05 * math.sqrt(2 * A_PUSH * STANDOFF)
STILL_TIME = 0.1    # the block counts as still once its speed reading has stayed under the still threshold this long
FREE_SLIDE_MARGIN = 0.02  # seconds after braking starts before a slide counts as free (the block has left the ball)
TRIGGERS = ("forgetting only", "told", "self-detected")  # what makes the adaptive controller become unsure
READING_NOISE_FLOOR = 0.005  # the smallest reading noise CUSUM assumes, so a noise-free sensor does not alarm at trivia


def release_speed(distance, friction):
    """The release speed that moves the block `distance` metres with this friction: block_travel solved for v."""
    return math.sqrt(2 * (distance + STANDOFF) / (1 / A_PUSH + 1 / (friction * GRAVITY)))


def reading_noise(speed_noise, window, timestep):
    """How far one friction reading is typically off, from the speed sensor's specified noise: the two noisy speed
    readings at a stretch's ends differ by about sqrt(2) x noise, divided by the stretch's duration and by g."""
    return max(READING_NOISE_FLOOR, math.sqrt(2) * speed_noise / (GRAVITY * window * timestep))


@dataclass(frozen=True)
class ShoveDecision:
    """One decision to shove, as the controller saw it."""
    time: float              # seconds
    block_position: float    # where it saw the block: its position readings averaged over the still spell, metres
    block_speed: float       # its speed reading then, m/s (always under the still threshold: it decides only when still)
    target: float            # where it was meant to go, metres
    direction: int           # +1 pushes along the track's +y, -1 along -y
    release_speed: float     # m/s
    friction_belief: float   # the estimate it planned with


class ShoveController:
    """The shove rule, steering the arm through its motion layer. Call update(target) once per step."""

    def __init__(self, world, friction_belief, tolerance=TOLERANCE, top_speed=TOP_SPEED, sensor=None):
        self.world = world
        self.sensor = BlockSensor(world) if sensor is None else sensor
        # Speed readings under this count as "still": 3 times the sensor's noise, and never under 1 mm/s (stage F).
        self.still_speed = still_speed_for(self.sensor.speed_noise)
        self.motion = PusherMotion(world)
        self.friction_belief = friction_belief
        self.tolerance, self.top_speed = tolerance, top_speed
        self.phase = "waiting"      # waiting -> getting ready -> shoving -> waiting
        self.decisions = []         # every ShoveDecision, in order
        self._plan = None
        self._still_since, self._still_sum, self._still_count = None, 0.0, 0

    def belief(self):
        """The friction estimate to plan with. Static keeps its own; the oracle and adaptive override this."""
        return self.friction_belief

    def told_of_change(self):
        """The benchmark announces that something just changed (only in the told condition). Static and the oracle
        have no use for it."""

    def update(self, target):
        """Read the sensor, learn (adaptive only), decide or carry on with the current shove, then steer the arm."""
        world, motion = self.world, self.motion

        # 1. What it sees of the block now.
        position, speed = self.sensor.position(), self.sensor.speed()

        # 2. Learn from what it sees (only the adaptive controller does anything here).
        self._learn(speed)

        # 3. Waiting: once the block has been still for STILL_TIME, decide whether, where and how hard to shove,
        #    from its position readings averaged over the still spell.
        if self.phase == "waiting" and self._block_has_settled(position, speed):
            seen = self._still_sum / self._still_count
            miss = target - seen
            if abs(miss) > self.tolerance:
                self._decide(target, miss, seen, speed)
                motion.follow(trajectory_to_ready(world.tip_position, seen, self._plan.direction))
                self.phase = "getting ready"

        # 4. Getting ready: when the ball reaches the ready point, shove.
        elif self.phase == "getting ready" and motion.done:
            motion.follow(Shove(world.tip_position, self._plan.direction, self._plan.release_speed))
            self.phase = "shoving"

        # 5. Shoving: when the ball has braked to rest, wait for the block to stop sliding.
        elif self.phase == "shoving" and motion.done:
            self.phase = "waiting"
            self._still_since, self._still_sum, self._still_count = None, 0.0, 0

        # 6. Steer the arm (before the first shove it simply holds its pose).
        if motion.trajectory is not None:
            motion.update()

    def _learn(self, speed):
        """Nothing to learn for a controller whose estimate is fixed or handed to it."""

    def _decide(self, target, miss, seen, speed):
        friction = self.belief()
        release = min(self.top_speed, max(MIN_SPEED, release_speed(abs(miss), friction)))
        self._plan = ShoveDecision(time=self.world.time, block_position=seen, block_speed=speed, target=target,
                                   direction=1 if miss > 0 else -1, release_speed=release, friction_belief=friction)
        self.decisions.append(self._plan)

    def _block_has_settled(self, position, speed):
        """Whether the speed reading has stayed under the still threshold for STILL_TIME; meanwhile, add up the
        position readings so the decision can use their average."""
        if abs(speed) >= self.still_speed:
            self._still_since, self._still_sum, self._still_count = None, 0.0, 0
            return False
        if self._still_since is None:
            self._still_since = self.world.time
        self._still_sum += position
        self._still_count += 1
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


class AdaptiveShover(ShoveController):
    """Learns the friction from its own free slides, and plans with what it has learned.

    During a free slide only friction acts, so the block slows at exactly friction x g, whatever its mass. From
    FREE_SLIDE_MARGIN after braking starts, while the block's speed reading is clearly above zero, every `window`
    steps (50 ms) give a reading: the speed change over the stretch, divided by its duration and by g. Only the two
    speed readings at the stretch's ends carry noise, so a longer stretch divides that noise by a longer time
    (as in stage F). The readings feed a FrictionEstimator, which starts from the nominal friction.

    One response to a change, three triggers. The response is to become unsure, so the next readings replace the old
    estimate. What triggers it:
      - "forgetting only": nothing; old readings just fade;
      - "told": the benchmark's announcement of each change (told_of_change);
      - "self-detected": its own CUSUM on each reading's prediction error (the reading minus the current estimate,
        before learning from it), in units of the reading noise. On an alarm it becomes unsure first and then learns
        the reading, so the first evidence of the change counts fully.
    """

    def __init__(self, world, trigger="forgetting only", forgetting=FRICTION_FORGETTING, window=WINDOW,
                 detector=None, **settings):
        super().__init__(world, friction_belief=NOMINAL_FRICTION, **settings)
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}")
        self.trigger = trigger
        self.estimator = FrictionEstimator(NOMINAL_FRICTION, forgetting)
        self.window = window
        # Learn only while the speed reading says the block is clearly sliding: above the still threshold, and at
        # least 1 cm/s.
        self.learning_speed = max(SLIDING_SPEED, self.still_speed)
        self.detector = (detector or Cusum()) if trigger == "self-detected" else None
        self.reading_noise = reading_noise(self.sensor.speed_noise, window, world.timestep)
        self.readings = []   # (time, friction reading) for every stretch learned from, for checking and plotting
        self.resets = []     # (time, "told" or "alarm") for every time it became unsure
        self._stretch = []   # speed readings so far in the current stretch

    def belief(self):
        self.friction_belief = self.estimator.friction
        return self.friction_belief

    def told_of_change(self):
        """Only the told version listens. A stretch half-collected across the change mixes before and after, so drop
        it."""
        if self.trigger == "told":
            self._become_unsure("told")
            self._stretch = []

    def _learn(self, speed):
        shove = self.motion.trajectory
        free = (isinstance(shove, Shove) and self.motion.elapsed >= shove.release_time + FREE_SLIDE_MARGIN
                and speed * self._plan.direction > self.learning_speed)
        if not free:
            self._stretch = []
            return
        self._stretch.append(speed)
        if len(self._stretch) == self.window + 1:
            slowing = -(self._stretch[-1] - self._stretch[0]) * self._plan.direction / (self.window * self.world.timestep)
            self._use_reading(slowing / GRAVITY)
            self._stretch = [self._stretch[-1]]  # the next stretch starts where this one ended

    def _use_reading(self, reading):
        """Check one friction reading for a change (self-detected only), then learn from it."""
        if self.detector is not None:
            error = (reading - self.estimator.friction) / self.reading_noise  # before learning from it
            if self.detector.check(error):
                self._become_unsure("alarm")                                # then learn, below
        self.estimator.learn(reading)
        self.readings.append((self.world.time, reading))

    def _become_unsure(self, cause):
        """The one response all three versions share."""
        self.estimator.become_unsure()
        self.resets.append((self.world.time, cause))
