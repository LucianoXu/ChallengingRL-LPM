import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from wrappers.intrinsic_models import SharedRNDModel, SharedLPMModel, SharedICMModel, build_shared_model

OBS, ACT, B = 12, 3, 4


def _obs(rng):
    return rng.standard_normal((B, OBS)).astype(np.float32)


def test_rnd_reward_shape_and_finite():
    m = SharedRNDModel(OBS, ACT, reward_scale=1.0, seed=0)
    rng = np.random.default_rng(0)
    r = m.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng))
    assert r.shape == (B,)
    # raw RND bonus is a non-negative squared error; reward_rms divides by std
    # only (no mean subtraction), so the first batch stays finite.
    assert np.all(np.isfinite(r))


def test_rnd_update_decreases_predictor_loss():
    m = SharedRNDModel(OBS, ACT, reward_scale=1.0, train_epochs=4, train_batch=8, seed=0)
    rng = np.random.default_rng(1)
    fixed_next = _obs(rng)
    for _ in range(5):  # feed the SAME next-obs repeatedly; training should reduce loss
        m.reward(_obs(rng), rng.integers(0, ACT, B), fixed_next)
    first = m.update()["loss"]
    last = first
    for _ in range(10):
        m.reward(_obs(rng), rng.integers(0, ACT, B), fixed_next)
        last = m.update()["loss"]
    assert last < first


def test_lpm_gated_until_buffer_fills():
    bs = 6
    m = SharedLPMModel(OBS, ACT, reward_scale=1.0, buffer_size=bs, seed=0)
    rng = np.random.default_rng(2)
    rewards = []
    for _ in range(4):  # B=4 per call; buffer fills after ceil(6/4)=2 calls
        rewards.append(m.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng)))
    assert np.all(rewards[0] == 0.0), "must be gated to 0 before buffer fills"
    assert any(np.any(r != 0.0) for r in rewards[1:]), "must activate after buffer fills"


def test_lpm_update_returns_losses():
    m = SharedLPMModel(OBS, ACT, reward_scale=1.0, buffer_size=4, seed=0)
    rng = np.random.default_rng(3)
    for _ in range(3):
        m.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng))
    d = m.update()
    assert "fwd_loss" in d and "err_loss" in d


def test_icm_reward_shape_and_update_losses():
    m = SharedICMModel(OBS, ACT, reward_scale=1.0, train_epochs=2, train_batch=4, seed=0)
    rng = np.random.default_rng(4)
    r = m.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng))
    assert r.shape == (B,)
    assert np.all(np.isfinite(r))
    d = m.update()
    assert "fwd_loss" in d and "inv_loss" in d


def test_build_shared_model_dispatch():
    assert isinstance(build_shared_model("rnd", OBS, ACT, 0.01), SharedRNDModel)
    assert isinstance(build_shared_model("lpm", OBS, ACT, 0.01), SharedLPMModel)
    assert isinstance(build_shared_model("icm", OBS, ACT, 0.01), SharedICMModel)
