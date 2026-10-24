"""Regret: from a given moment, how much better the oracle would have done over the next few seconds.

At each checkpoint along a stream:
  1. branch the stream and let the oracle take over the branch for the look-ahead;
  2. score the oracle's future, and the controller's own future in the main run, over that look-ahead;
  3. regret = oracle's score - controller's score.

A step's reward is minus the miss. A score is the rewards over the look-ahead, with nearer moments counting
more (the discounted return of Sutton & Barto chapter 3), scaled so that it reads as a weighted average
miss, in metres. This is Jin's deviation regret with one deviation, the oracle.
"""
import math
from dataclasses import dataclass

import numpy as np

from boundary_free.controllers import OracleController
from boundary_free.stream import Stream, StreamRecord, run

LOOK_AHEAD = 4.0   # seconds each checkpoint looks ahead
FADE_TIME = 1.0    # how fast later moments count less: a moment this far ahead counts about a third as much
CHECK_EVERY = 0.5  # seconds between checkpoints


def look_ahead_weights(timestep, look_ahead=LOOK_AHEAD, fade_time=FADE_TIME):
    """How much each step of the look-ahead counts (relative to the current checkpoint): 
    fading with time, and adding up to 1.
    
    :param timestep: the simulation's timestep, in seconds (ex: 0.002)
    :param look_ahead: how far ahead to look, in seconds
    :param fade_time: how fast later moments count less, in seconds (a moment this
    
    :return: a numpy array of normalized weights, one per timestep, that add up to 1.0"""
    discount = math.exp(-timestep / fade_time)  # the per-step discount, gamma
    weights = discount ** np.arange(round(look_ahead / timestep)) # 0.998, 0.996, ... (discount ** 1999 = 0.018)
    return weights / weights.sum() # normalized weights


def score(misses, weights):
    """The score of a stretch of the future: minus the weighted average miss, in metres. Higher is better."""
    # math.fsum adds up exactly, so the same misses always give exactly the same score.
    # strict=True stops with an error if the stretch of misses is not exactly as long as the look-ahead.
    return -math.fsum(w * m for w, m in zip(weights, misses, strict=True))


@dataclass
class RegretRecord:
    """Regret at each checkpoint, plus the main run it was measured on."""
    times: list             # checkpoint times, in seconds
    controller_score: list  # the controller's own score from each checkpoint
    oracle_score: list      # the oracle's score from each checkpoint, on a branch
    regret: list            # oracle's score minus controller's score, in metres (positive: the oracle did better)
    main_run: StreamRecord  # everything that happened in the main run

    @property
    def total(self):
        """Regret added up over all checkpoints."""
        return math.fsum(self.regret)

    @property
    def rate(self):
        """Average regret per checkpoint."""
        return self.total / len(self.regret)


def measure_regret(schedule, make_controller, make_deviation=OracleController,
                   look_ahead=LOOK_AHEAD, fade_time=FADE_TIME, check_every=CHECK_EVERY):
    """Run a controller through a whole stream, measuring its regret against the oracle at every checkpoint.

    make_controller is given the main stream's world and returns the controller to measure.
    make_deviation is given a branch's world and returns the alternative to compare against.
    """
    stream = Stream(schedule)
    controller = make_controller(stream.world)
    weights = look_ahead_weights(stream.world.timestep, look_ahead, fade_time)

    # 1. Run the main stream. At each checkpoint, branch it and see what the oracle would have done.
    main_run = StreamRecord()
    times, starts, oracle_score = [], [], []
    last_checkpoint = schedule.length - look_ahead
    while stream.time <= last_checkpoint + 1e-9:
        # The clock adds up 0.002 s steps, so 1 s can come out as 0.9999999999999998; rounding tidies that.
        times.append(round(stream.time, 6))
        starts.append(len(main_run.time))  # where this checkpoint's future will begin in the main run's record
        branch = stream.branch()
        future = run(branch, make_deviation(branch.world), look_ahead)
        oracle_score.append(score(_misses(future), weights))
        _extend(main_run, run(stream, controller, check_every))

    # 2. Carry the main run on to the end, so the last checkpoint's look-ahead has happened too.
    _extend(main_run, run(stream, controller, schedule.length - stream.time))

    # 3. Score the controller's own future from each checkpoint, over the same stretch of time, and compare.
    misses, steps = _misses(main_run), len(weights)
    controller_score = [score(misses[start:start + steps], weights) for start in starts]
    regret = [o - c for o, c in zip(oracle_score, controller_score)]
    return RegretRecord(times, controller_score, oracle_score, regret, main_run)


def _misses(record):
    """The miss at every step of a record, in metres, ignoring which side of the target the block is on."""
    return np.abs(np.array(record.position) - np.array(record.target))


def _extend(record, more):
    """Add the steps recorded in `more` onto the end of `record`."""
    for name in vars(record):
        getattr(record, name).extend(getattr(more, name))
