import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers.env_factory import make_env
from wrappers.intrinsic_models import build_shared_model
from wrappers.intrinsic_vec_wrapper import IntrinsicVecWrapper

ENV = "MiniGrid-Empty-8x8-v0"


def _venv(n=2):
    fns = [lambda s=ENV, k=i: make_env(s, intrinsic=False, noise=False, seed=k,
                                       training=True, method="none") for i in range(n)]
    return DummyVecEnv(fns)


def _wrap(method="lpm", n_steps=8, n=2):
    venv = _venv(n)
    obs_dim = venv.observation_space.shape[0]
    num_actions = venv.action_space.n
    model = build_shared_model(method, obs_dim, num_actions, reward_scale=1.0, seed=0)
    return IntrinsicVecWrapper(venv, model, n_steps=n_steps), model


def test_single_shared_model():
    w, model = _wrap("lpm")
    assert w.model is model  # ONE model for all envs


def test_update_called_once_per_rollout(monkeypatch):
    w, model = _wrap("rnd", n_steps=5, n=2)
    calls = {"n": 0}
    orig = model.update

    def counted():
        calls["n"] += 1
        return orig()

    monkeypatch.setattr(model, "update", counted)
    w.reset()
    for _ in range(10):  # exactly 2 rollouts of n_steps=5
        w.step_async(np.array([w.action_space.sample() for _ in range(2)]))
        w.step_wait()
    assert calls["n"] == 2


def test_reward_is_extrinsic_plus_intrinsic():
    w, _ = _wrap("rnd", n_steps=4)
    w.reset()
    w.step_async(np.array([w.action_space.sample() for _ in range(2)]))
    _, rews, _, _ = w.step_wait()
    assert rews.shape == (2,) and np.all(np.isfinite(rews))


def test_icm_shared_wrapper_reward_is_finite():
    w, _ = _wrap("icm", n_steps=4)
    w.reset()
    w.step_async(np.array([w.action_space.sample() for _ in range(2)]))
    _, rews, _, _ = w.step_wait()
    assert rews.shape == (2,) and np.all(np.isfinite(rews))


def test_make_vector_env_uses_shared_wrapper():
    import train
    from config import PPO_HYPERPARAMS
    env = train.make_vector_env(ENV, intrinsic=True, noise=False, seed=0,
                                training=True, n_envs=2, log_dir=None,
                                run_name="t", method="lpm", beta=0.01)
    # VecMonitor -> IntrinsicVecWrapper -> VecEnv ; exactly one shared model.
    inner = env
    found = None
    while hasattr(inner, "venv"):
        if isinstance(inner, IntrinsicVecWrapper):
            found = inner
        inner = inner.venv
    assert found is not None and found.n_steps == PPO_HYPERPARAMS["n_steps"]
    env.close()


def test_ep_intrinsic_logged_at_episode_end():
    # MiniGrid-Empty truncates at max_steps; run long enough to end an episode.
    w, _ = _wrap("rnd", n_steps=4, n=2)
    w.reset()
    saw_ep = False
    for _ in range(600):
        w.step_async(np.array([w.action_space.sample() for _ in range(2)]))
        _, _, dones, infos = w.step_wait()
        for d, info in zip(dones, infos):
            if d:
                assert "ep_intrinsic" in info and "ep_extrinsic" in info
                saw_ep = True
        if saw_ep:
            break
    assert saw_ep
