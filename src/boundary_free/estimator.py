"""The estimator: learns a block's friction and mass from how it responds to pushes while it slides.

While the block slides:   push = mass x acceleration + (friction x mass) x g x direction

So every sliding moment is a point on a straight line through (acceleration, push): the slope is the mass, and
the push at zero acceleration is the friction force. The estimator fits that line by recursive least squares:
it nudges its fit with each new point instead of refitting every point it has seen. A forgetting factor makes
older points count for less and less, so the fit can follow changes.

The estimator only does the arithmetic. Deciding which moments count as sliding is the controller's job.
"""
import numpy as np

from boundary_free.block_world import GRAVITY

FORGETTING = 0.99         # each learning step, older points' weight is multiplied by this; memory is about 1/(1 - this) points
START_UNCERTAINTY = 10.0  # how unsure it is of its starting guesses: bigger means the first points move it further
MASS_RANGE = (0.1, 10.0)  # kg: beliefs handed to the controller are kept inside these, however odd the fit
FRICTION_RANGE = (0.0, 2.0)


class FrictionMassEstimator:
    """Recursive least squares on push = mass x acceleration + (friction x mass) x g x direction."""

    def __init__(self, friction_guess, mass_guess, forgetting=FORGETTING, start_uncertainty=START_UNCERTAINTY):
        # 1. What it believes: the line's slope (the mass) and its height at zero acceleration, divided by g
        #    (friction x mass). Fitting friction x mass rather than friction keeps the equation a straight line.
        self.estimate = np.array([mass_guess, friction_guess * mass_guess])
        # How unsure it is about those two numbers, and how the two uncertainties are linked (the covariance).
        self.uncertainty = np.eye(2) * start_uncertainty
        self.forgetting = forgetting
        self.points_learned = 0

    # ---- 2. Learning from one sliding moment ---------------------------------------

    def learn(self, acceleration, direction, push):
        """Nudge the fit towards one point: the block slid in `direction` (+1 or -1) and sped up by
        `acceleration` (m/s^2) under `push` (N)."""
        x = np.array([acceleration, GRAVITY * direction])  # the point's position, in the line's terms
        predicted = x @ self.estimate                       # the push the current fit says this needed
        P = self.uncertainty
        gain = P @ x / (self.forgetting + x @ P @ x)        # how far to move: further when more unsure
        self.estimate = self.estimate + gain * (push - predicted)  # new = old + gain x (what happened - predicted)
        # More sure after the point; then dividing by the forgetting factor makes it a little less sure again,
        # which is what lets later points keep moving the fit.
        self.uncertainty = (P - np.outer(gain, x @ P)) / self.forgetting
        self.points_learned += 1

    # ---- 3. What it believes, as the controller wants it -----------------------------

    @property
    def mass(self):
        return float(np.clip(self.estimate[0], *MASS_RANGE))

    @property
    def friction(self):
        return float(np.clip(self.estimate[1] / self.mass, *FRICTION_RANGE))
