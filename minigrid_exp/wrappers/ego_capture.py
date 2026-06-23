import gymnasium as gym


class EgoCaptureWrapper(gym.ObservationWrapper):
    """Records the egocentric image the policy receives, leaving the obs unchanged.

    Placed AFTER ``ObservationNoiseWrapper`` and BEFORE ``ImgObsWrapper`` in the
    render-time wrapper stack so that ``last_image`` is exactly the (possibly
    noisy) 7x7x3 image the policy saw this step. Pure pass-through: the dict obs
    and the observation_space are unmodified, so training/eval behavior is not
    affected if this is ever inserted there.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_image = None

    def observation(self, obs):
        # MiniGrid dict obs: {"image", "direction", "mission"}. Copy so a later
        # in-place noise draw (there is none downstream, but be safe) can't alias.
        self.last_image = obs["image"].copy()
        return obs
