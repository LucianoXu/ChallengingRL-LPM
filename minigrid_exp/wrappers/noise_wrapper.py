import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """Corrupt a fraction of *cells* in the MiniGrid symbolic image.

    `noise_prob` is the per-cell Bernoulli probability that a grid cell (a
    location in the H x W egocentric view) is corrupted — i.e. the expected
    fraction of cells perturbed, NOT a per-channel/per-element probability. A
    corrupted cell is re-drawn as a unit: each of its 3 channels (object, color,
    state) gets an independent uniform draw within that channel's valid MiniGrid
    range (object in [0,10], color in [0,5], state in [0,2]), so the noise never
    injects encodings the agent could not otherwise see.

    Owns a seeded numpy Generator for reproducibility. `last_cell_mask` exposes
    the most recent (H, W) boolean cell mask for testing/observability.
    """

    def __init__(self, env, noise_prob: float = 0.10, seed: int | None = None):
        super().__init__(env)
        self.noise_prob = noise_prob
        self._rng = np.random.default_rng(seed)
        # Per-channel inclusive max for the (object, color, state) encoding.
        self._channel_max = np.array(
            [max(OBJECT_TO_IDX.values()),
             max(COLOR_TO_IDX.values()),
             max(STATE_TO_IDX.values())],
            dtype=np.int64,
        )
        self.last_cell_mask = None

    def observation(self, obs):
        obs = obs.copy()
        image = obs["image"]
        h, w = image.shape[:2]

        cell_mask = self._rng.random((h, w)) < self.noise_prob
        self.last_cell_mask = cell_mask
        n = int(cell_mask.sum())
        if n:
            image = image.copy()  # only allocate when we actually corrupt
            for c in range(image.shape[2]):
                image[cell_mask, c] = self._rng.integers(
                    0, int(self._channel_max[c]) + 1, size=n, dtype=image.dtype)
            obs["image"] = image
        return obs
