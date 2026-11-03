"""The boundary-gap experiment: how much is it worth to the adaptive controller to be told when things change?

For each noise level:
  1. Tune. Each condition, told and not told, gets its own FORGETTING: the one with the lowest average regret on
     the training streams. The test streams play no part, so neither condition is tuned to the streams it is
     judged on.
  2. Compare. Both conditions run on the same fresh test streams and are compared stream by stream (paired):
     most of what differs between streams, such as where the targets fall, the noise and the stalls, affects
     both alike and cancels out.
  3. Summarise. The typical result (the median gap, and the share of streams where told did better) and the
     risk (the mean, and the worst stream), since a few failures can dominate an average.

Streams differ in their targets and their noise (example_schedule's seed); all share the same hidden changes.
"""
from dataclasses import dataclass

import numpy as np

from boundary_free.controllers import adaptive_controller, still_speed_for
from boundary_free.regret import measure_regret
from boundary_free.stream import example_schedule

FORGETTINGS = (0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99)
TRAINING_STREAMS = tuple(range(100, 108))
TEST_STREAMS = tuple(range(300, 340))
CHECK_EVERY = 2.0  # seconds between regret checkpoints: sparser than the default, as there are hundreds of runs
CONDITIONS = ("not told", "told")


def adaptive_regret(seed, noise, condition, forgetting, check_every=CHECK_EVERY, drift_every=None):
    """Adaptive's regret rate (m) on one stream: the example stream's changes, with this seed's targets and noise."""
    def make(world):
        return adaptive_controller(world, forgetting=forgetting, still_speed=still_speed_for(noise))
    schedule = example_schedule(seed=seed, speed_noise=noise)
    return measure_regret(schedule, make, told=(condition == "told"), drift_every=drift_every,
                          check_every=check_every).rate


@dataclass
class GapResult:
    """Both conditions at one noise level, on the test streams."""
    noise: float
    forgetting: dict        # the tuned FORGETTING for each condition
    not_told: np.ndarray    # regret rate (m) on each test stream
    told: np.ndarray        # the same streams, in the same order

    @property
    def gap(self):
        """For each test stream, not told's regret minus told's: positive means being told helped."""
        return self.not_told - self.told

    @property
    def told_better(self):
        """The share of test streams on which told did better."""
        return float(np.mean(self.gap > 0))

    def gap_interval(self, draws=5000, seed=0):
        """A 95% interval for the mean gap, by resampling the test streams (the bootstrap)."""
        rng = np.random.default_rng(seed)
        means = [rng.choice(self.gap, len(self.gap)).mean() for _ in range(draws)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boundary_gap(noises, pool, forgettings=FORGETTINGS, training=TRAINING_STREAMS, test=TEST_STREAMS,
                 drift_every=None):
    """Run the whole experiment for each noise level, spreading the runs over `pool` (a process pool).
    Returns {noise: GapResult}."""
    def measure(jobs):
        return dict(zip(jobs, pool.map(adaptive_regret, *zip(*jobs), [CHECK_EVERY] * len(jobs),
                                       [drift_every] * len(jobs))))

    # 1. Tune on the training streams: each condition's FORGETTING with the lowest average regret.
    rates = measure([(seed, noise, condition, f) for noise in noises for condition in CONDITIONS
                     for f in forgettings for seed in training])
    tuned = {(noise, condition): min(forgettings,
                                     key=lambda f: np.mean([rates[(s, noise, condition, f)] for s in training]))
             for noise in noises for condition in CONDITIONS}

    # 2. Compare both conditions on the same test streams, each with its own tuned FORGETTING.
    rates = measure([(seed, noise, condition, tuned[(noise, condition)]) for noise in noises
                     for condition in CONDITIONS for seed in test])

    # 3. Gather them stream by stream.
    return {noise: GapResult(noise, {c: tuned[(noise, c)] for c in CONDITIONS},
                             np.array([rates[(s, noise, "not told", tuned[(noise, "not told")])] for s in test]),
                             np.array([rates[(s, noise, "told", tuned[(noise, "told")])] for s in test]))
            for noise in noises}
