"""What the arm world's controllers see of the block: noisy readings of its position and speed along the track.

The noise is drawn ahead of time from a seed, one number per simulation step, and looked up by the step's number.
So a branch built from the same seed sees exactly the same noise at the same moments as the main run (common random
numbers, as in stage F): any difference between the two comes from what the controllers did, not from luck.
"""
import numpy as np

CHUNK = 10_000  # steps of noise drawn at a time (20 s); each chunk has its own seed, so any step can be looked up


class BlockSensor:
    """The block's position (metres) and speed (m/s) along its track, each with its own Gaussian noise."""

    def __init__(self, world, position_noise=0.0, speed_noise=0.0, seed=0):
        self.world = world
        self.position_noise, self.speed_noise, self.seed = position_noise, speed_noise, seed
        self._chunks = {}

    def position(self):
        """The block's position reading now."""
        return self.world.position + self.position_noise * self._noise(0)

    def speed(self):
        """The block's speed reading now."""
        return self.world.velocity + self.speed_noise * self._noise(1)

    def _noise(self, which):
        """A standard normal draw for this step: column 0 for position, column 1 for speed."""
        step = int(round(self.world.time / self.world.timestep))
        chunk, row = divmod(step, CHUNK)
        if chunk not in self._chunks:
            self._chunks[chunk] = np.random.default_rng([self.seed, chunk]).standard_normal((CHUNK, 2))
        return self._chunks[chunk][row, which]
