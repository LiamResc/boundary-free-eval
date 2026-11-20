"""Moving the pusher: the task-space position controller of stage 1, steering the arm world's ball tip.

Every step, as in playground/06, plus one addition (velocity feedforward):
  1. The reference, where the ball should be right now, moves along a straight-line path, speeding up and slowing
     down smoothly.
  2. The end-effector position error: the reference minus where the ball is.
  3. A fresh Jacobian for the arm's current pose: how the ball moves per small turn of each joint.
  4. Damped least squares: the smallest joint turns that the Jacobian says would cover the error, and the joint
     speeds that would move the ball at the reference's velocity.
  5. Each motor is commanded to its joint's current angle, plus its turn, plus a lead of its reference speed times
     damping / stiffness (0.1 s for every FR3 motor). That lead is the velocity feedforward. Without it, a joint
     keeps moving only by trailing its command, so the ball trails the reference by 0.1 s x its speed (stage 1's
     10 mm at 0.1 m/s). With it, the motor's damping works against the difference between the joint's speed and
     its reference speed, not against all of its speed, so nothing has to trail.

Paths to a shove go over the block, never through it.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from boundary_free.arm_world import BALL_RADIUS, BLOCK_HALF_SIZE, TRACK_AXIS, TRACK_X

CAUTION = 1e-4   # the damping in damped least squares: keeps the answer calm in awkward poses (as in 06)
MAX_TURN = 0.2   # safety limit on any motor's turn, in radians (as in 06); normal moves never reach it

# The real FR3's joint speed limits, in rad/s (Franka's specifications, checked 2026-10-12). The simulated arm does
# not enforce them, so motions must stay inside them by design.
JOINT_SPEED_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])

MOVE_SPEED = 0.3          # how fast the ball moves when repositioning (not shoving), m/s
MOVE_ACCELERATION = 1.0   # how quickly it speeds up and slows down, m/s^2
CLEARANCE = 0.05          # crossing over the block, the ball's bottom passes this far above its top, metres
GAP = 0.01                # ready to shove, the ball waits this far from the block's face, metres


# ---- 1. Paths: where the ball should be, moment by moment ------------------------

@dataclass(frozen=True)
class Segment:
    """A straight line from start to goal, at up to top_speed, speeding up from rest and slowing down to rest at
    `acceleration`: a trapezoidal speed profile (ramp up, cruise, ramp down)."""
    start: np.ndarray
    goal: np.ndarray
    top_speed: float
    acceleration: float

    @property
    def length(self):
        return float(np.linalg.norm(self.goal - self.start))

    @property
    def peak_speed(self):
        """The fastest the reference goes: top_speed, unless the line is too short to reach it."""
        return min(self.top_speed, np.sqrt(self.length * self.acceleration))

    @property
    def duration(self):
        ramp = self.peak_speed / self.acceleration
        cruise = (self.length - self.peak_speed * ramp) / self.peak_speed if self.length > 0 else 0.0
        return 2 * ramp + cruise

    def reference(self, t):
        """Where the ball should be t seconds into the segment, and its velocity: (metres, metres per second)."""
        if self.length == 0:
            return self.start.copy(), np.zeros(3)
        direction = (self.goal - self.start) / self.length
        peak, ramp, end = self.peak_speed, self.peak_speed / self.acceleration, self.duration
        t = min(max(t, 0.0), end)
        if t < ramp:                                   # speeding up
            covered, speed = 0.5 * self.acceleration * t**2, self.acceleration * t
        elif t > end - ramp:                           # slowing down
            left = end - t
            covered, speed = self.length - 0.5 * self.acceleration * left**2, self.acceleration * left
        else:                                          # cruising
            covered, speed = 0.5 * peak * ramp + peak * (t - ramp), peak
        return self.start + covered * direction, speed * direction


class Path:
    """Straight segments, one after another."""

    def __init__(self, segments):
        self.segments = list(segments)

    @classmethod
    def through(cls, points, top_speed=MOVE_SPEED, acceleration=MOVE_ACCELERATION):
        """A path through these points, in order, stopping at each."""
        points = [np.asarray(p, dtype=float) for p in points]
        return cls(Segment(a, b, top_speed, acceleration) for a, b in zip(points, points[1:]))

    @property
    def duration(self):
        return sum(segment.duration for segment in self.segments)

    @property
    def end(self):
        return self.segments[-1].goal

    def reference(self, t):
        """Where the ball should be t seconds into the path, and its velocity. After the end: the end, at rest."""
        for segment in self.segments:
            if t <= segment.duration:
                return segment.reference(t)
            t -= segment.duration
        return self.end.copy(), np.zeros(3)


def ready_point(block_position, push_direction):
    """Where the ball waits before shoving the block towards push_direction (+1 along the track's +y, -1 along -y):
    level with the block's centre, a GAP away from the face it will push."""
    centre = np.array([TRACK_X, 0.0, BLOCK_HALF_SIZE]) + block_position * TRACK_AXIS
    return centre - push_direction * (BLOCK_HALF_SIZE + BALL_RADIUS + GAP) * TRACK_AXIS


def path_to_ready(tip, block_position, push_direction):
    """A path that takes the ball from `tip` to the ready point. If the ball is already clear of the block on the
    side it pushes from, it goes straight there; otherwise it goes up, across above the block, and down."""
    ready = ready_point(block_position, push_direction)
    behind = (np.dot(tip, TRACK_AXIS) - block_position) * push_direction  # negative on the side it pushes from
    if behind < -(BLOCK_HALF_SIZE + BALL_RADIUS):
        return Path.through([tip, ready])
    above = 2 * BLOCK_HALF_SIZE + CLEARANCE + BALL_RADIUS  # the ball's centre height when crossing over
    up = np.array([tip[0], tip[1], max(tip[2], above)])
    across = np.array([ready[0], ready[1], above])
    return Path.through([p for p, q in zip([tip, up, across, ready], [None, tip, up, across])
                         if q is None or np.linalg.norm(p - q) > 1e-9])


# ---- 2. The task-space position controller ---------------------------------------

class PusherMotion:
    """Steers the arm so the pusher's ball follows a path."""

    def __init__(self, world):
        self.world = world
        self._tip = world.model.site("pusher_tip").id
        self._jacobian = np.zeros((3, world.model.nv))
        # Each motor's damping / stiffness (kv / kp), read from the model: 0.1 s for every FR3 motor.
        self.lead = -world.model.actuator_biasprm[:, 2] / world.model.actuator_gainprm[:, 0]
        self.path, self._started = None, 0.0

    def follow(self, path):
        """Start following a path now."""
        self.path, self._started = path, self.world.time

    @property
    def done(self):
        """Whether the reference has reached the end of the path (the ball may still be settling)."""
        return self.path is None or self.world.time - self._started >= self.path.duration

    def update(self):
        """Work out this step's motor commands and give them to the arm. Returns the end-effector position error."""
        world = self.world
        # 1. The reference: where the ball should be now, and how fast it should be moving.
        reference, velocity = self.path.reference(world.time - self._started)

        # 2. The end-effector position error, in metres.
        error = reference - world.tip_position

        # 3. A fresh Jacobian for the current pose: one column per arm joint, metres per radian.
        mujoco.mj_jacSite(world.model, world.data, self._jacobian, None, self._tip)
        jacobian = self._jacobian[:, world.arm_dofs]

        # 4. Damped least squares, once for the turns that cover the error and once for the joint speeds that give
        #    the reference velocity. Scale the turns down together if any is beyond the safety limit.
        solve = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + CAUTION * np.eye(3))
        turns = solve @ error
        biggest = np.abs(turns).max()
        if biggest > MAX_TURN:
            turns *= MAX_TURN / biggest
        speeds = solve @ velocity

        # 5. Command each motor: current angle, plus turn, plus lead (velocity feedforward).
        world.command(world.joint_angles + turns + self.lead * speeds)
        return error
