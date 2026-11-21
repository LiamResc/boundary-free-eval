"""Change detection: a two-sided CUSUM on standardized prediction errors.

Each prediction error is divided by its typical size when nothing has changed, so 1 means "as far off as a reading
usually is". Two CUSUM statistics add the errors up, minus the reference value k each time: the upper one collects
errors leaning positive, the lower one errors leaning negative. When either passes the decision threshold h, it raises
an alarm and both restart from zero. k is about half the smallest shift worth detecting; h trades detection delay
against false alarms (see playground/16_cusum.py).
"""
ALLOWANCE = 0.5   # the reference value k, in units of the noise
THRESHOLD = 5.0   # the decision threshold h


class Cusum:
    """A two-sided CUSUM. Feed it standardized prediction errors one at a time with check()."""

    def __init__(self, allowance=ALLOWANCE, threshold=THRESHOLD):
        self.allowance, self.threshold = allowance, threshold
        self.upper = self.lower = 0.0

    def check(self, error):
        """Add one standardized prediction error; True if that raises an alarm (both statistics then restart)."""
        self.upper = max(0.0, self.upper + error - self.allowance)  # evidence that the errors lean positive
        self.lower = max(0.0, self.lower - error - self.allowance)  # evidence that they lean negative
        if self.upper > self.threshold or self.lower > self.threshold:
            self.upper = self.lower = 0.0
            return True
        return False
