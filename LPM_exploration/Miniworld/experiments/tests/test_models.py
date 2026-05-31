import numpy as np
import pytest
import torch
import models

OBS = (120, 160, 3)
INP = (3, 120, 160)


def _frame():
    return np.random.randint(0, 256, size=OBS, dtype=np.uint8)


def test_feature_extractor_shape():
    fe = models.CNNFeatureExtractor(INP)
    x = torch.rand(2, *OBS)  # (N,H,W,C)
    out = fe(x)
    assert out.shape == (2, fe.feature_size)


def test_mse_model_reward_and_update():
    m = models.MSEModel(INP, num_actions=5, device="cpu")
    r = m.reward(_frame(), _frame(), 2)
    assert isinstance(r, float) and r >= 0
    states = torch.rand(4, *OBS)
    nstates = torch.rand(4, *OBS)
    actions = torch.randint(0, 5, (4,))
    losses = m.update(states, nstates, actions)
    assert "pred_loss" in losses


def test_lpm_model_reward_bounded_and_update():
    m = models.LPMModel(INP, num_actions=5, device="cpu")
    r = m.reward(_frame(), _frame(), 1)
    assert isinstance(r, float) and r <= 0.5
    states = torch.rand(6, *OBS); nstates = torch.rand(6, *OBS)
    actions = torch.randint(0, 5, (6,))
    losses = m.update(states, nstates, actions)
    assert "pred_loss" in losses and "unc_loss" in losses


@pytest.mark.parametrize("name,expect", [
    ("icm", "fwd_loss"), ("rnd", "rnd_loss"),
])
def test_icm_rnd_reward_and_update(name, expect):
    m = models.build_model(name, INP, 5, device="cpu")
    r = m.reward(_frame(), _frame(), 3)
    assert isinstance(r, float) and r >= 0
    losses = m.update(torch.rand(4, *OBS), torch.rand(4, *OBS), torch.randint(0, 5, (4,)))
    assert expect in losses


def test_none_model_zero_reward():
    m = models.build_model("none", INP, 5, device="cpu")
    assert m.reward(_frame(), _frame(), 0) == 0.0
    assert m.update(torch.rand(2, *OBS), torch.rand(2, *OBS), torch.randint(0, 5, (2,))) == {}


def test_build_model_all_names():
    for n in ["lpm", "rnd", "icm", "mse", "none"]:
        assert isinstance(models.build_model(n, INP, 5, "cpu"), models.IntrinsicModel)
