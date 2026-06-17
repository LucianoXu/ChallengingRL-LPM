import gymnasium as gym
import numpy as np


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """
    Adds random noise to the MiniGrid image observation.

    MiniGrid observations are dictionaries:
    {
        "image": ...,
        "direction": ...,
        "mission": ...
    }

    We only modify the image part.
    """

    def __init__(self, env, noise_prob=0.10):
        super().__init__(env)
        self.noise_prob = noise_prob

    def observation(self, obs):
        obs = obs.copy()

        image = obs["image"].copy()

        noise_mask = np.random.random(size=image.shape) < self.noise_prob

        random_values = np.random.randint(
            low=0,
            high=10,
            size=image.shape,
            dtype=image.dtype,
        )

        image[noise_mask] = random_values[noise_mask]

        obs["image"] = image

        return obs