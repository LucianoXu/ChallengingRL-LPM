import numpy as np
import maze_envs


def test_make_env_each_variant_resets_and_steps():
    for variant in ["nonoise", "noisy_tv", "action_noise"]:
        env = maze_envs.make_env(variant, seed=0)
        obs, info = env.reset(seed=0)
        assert obs.shape == (120, 160, 3)
        assert obs.dtype == np.uint8
        obs, r, term, trunc, info = env.step(2)  # move_forward
        assert r == 0.0
        assert "pos" in info and len(info["pos"]) == 2
        env.close()


def test_obs_scale_changes_resolution():
    env = maze_envs.make_env("nonoise", seed=0, obs_scale=0.5)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (60, 80, 3)
    env.close()
