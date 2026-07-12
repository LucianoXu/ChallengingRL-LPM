import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch as th
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


@pytest.mark.parametrize("method", ["rnd", "lpm", "icm"])
def test_state_roundtrip_preserves_learning_state(method):
    source = build_shared_model(
        method, OBS, ACT, reward_scale=0.01, seed=7,
        train_epochs=2, train_batch=4,
        **({"buffer_size": 6} if method == "lpm" else {}),
    )
    rng = np.random.default_rng(8)
    for _ in range(3):
        source.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng))
    source.update()  # populate optimizer state
    source.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng))

    restored = build_shared_model(
        method, OBS, ACT, reward_scale=0.01, seed=999,
        train_epochs=2, train_batch=4,
        **({"buffer_size": 6} if method == "lpm" else {}),
    )
    restored.load_state_dict(source.state_dict())

    assert np.array_equal(restored.obs_rms.mean, source.obs_rms.mean)
    assert np.array_equal(restored.obs_rms.var, source.obs_rms.var)
    assert restored.obs_rms.count == source.obs_rms.count
    assert np.array_equal(restored.reward_rms.mean, source.reward_rms.mean)
    assert np.array_equal(restored.reward_rms.var, source.reward_rms.var)
    assert restored.reward_rms.count == source.reward_rms.count
    if method == "lpm":
        assert len(restored.fwd_opt.state) == len(source.fwd_opt.state) > 0
        assert len(restored.err_opt.state) == len(source.err_opt.state) > 0
        assert len(restored.buf) == len(source.buf)
        assert len(restored._rollout) == len(source._rollout)
    else:
        assert len(restored.opt.state) == len(source.opt.state) > 0
        if method == "rnd":
            assert len(restored._rollout_next) == len(source._rollout_next)
        else:
            assert len(restored._rollout) == len(source._rollout)

    probe_prev = _obs(rng)
    probe_actions = rng.integers(0, ACT, B)
    probe_next = _obs(rng)
    expected = source.reward(probe_prev, probe_actions, probe_next)
    actual = restored.reward(probe_prev, probe_actions, probe_next)
    assert np.allclose(actual, expected, rtol=0, atol=1e-7)

    source_tensors = {
        key: value for key, value in source.state_dict().items()
        if isinstance(value, dict) and any(isinstance(v, th.Tensor) for v in value.values())
    }
    restored_state = restored.state_dict()
    for group, tensors in source_tensors.items():
        if group not in restored_state:
            continue
        for key, value in tensors.items():
            if isinstance(value, th.Tensor):
                assert th.equal(restored_state[group][key], value)
