import math

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


def test_lpm_reward_is_logspace_difference_eq3():
    # Paper Eq (1)-(3): epsilon = log(MSE); error model g_phi predicts that
    # log-error; intrinsic reward r = g_phi - epsilon (a difference of log-errors),
    # with NO eta and NO 0.5 clip, gated to 0 until the error queue is full.
    m = models.LPMModel(INP, num_actions=5, device="cpu", buffer_size=3)
    m.unc = lambda s, a: torch.tensor([[5.0]])  # g_phi := predicted log-error
    # Alg 1 line 6: reward is 0 until |D| = buffer_size.
    assert m.reward(_frame(), _frame(), 1) == 0.0
    assert m.reward(_frame(), _frame(), 1) == 0.0
    s, ns = _frame(), _frame()
    mse, _, _ = m._mse(s, ns, 1)
    r = m.reward(s, ns, 1)  # buffer now full -> r = g_phi - log(MSE)
    assert abs(r - (5.0 - math.log(mse + 1e-6))) < 1e-4
    assert r > 0.5  # log-space reward is NOT clipped at 0.5 (the raw form was)


def test_lpm_raw_mode_reproduces_clipped_notebook_form():
    # reward_space="raw" preserves the pre-fix upstream-notebook reward for
    # reproducing the earlier runs: r = min(0.5, eta*exp(g_phi) - MSE).
    m = models.LPMModel(INP, num_actions=5, device="cpu", reward_space="raw")
    r = m.reward(_frame(), _frame(), 1)
    assert isinstance(r, float) and r <= 0.5


def test_lpm_update_returns_pred_and_unc_losses():
    m = models.LPMModel(INP, num_actions=5, device="cpu")
    m.reward(_frame(), _frame(), 1)
    losses = m.update(torch.rand(6, *OBS), torch.rand(6, *OBS),
                      torch.randint(0, 5, (6,)))
    assert "pred_loss" in losses and "unc_loss" in losses


def test_lpm_error_model_lr_is_paper_consistent():
    # Empirically (overfit probe), the notebook's error-model lr 1e-2 drives g_phi
    # into the [-10,10] clamp under the log-space objective (zero gradient -> dead
    # error model, reward becomes a constant offset), while 1e-3 fits the log-error
    # targets cleanly. C.2 only ever specifies 1e-3, so the faithful log reward
    # uses 1e-3; raw mode keeps 1e-2 to reproduce the pre-fix runs.
    assert models.LPMModel(INP, 5, device="cpu").unc_opt.param_groups[0]["lr"] == 1e-3
    raw = models.LPMModel(INP, 5, device="cpu", reward_space="raw")
    assert raw.unc_opt.param_groups[0]["lr"] == 1e-2
    tuned = models.LPMModel(INP, 5, device="cpu", pred_lr=3e-4, unc_lr=7e-4)
    assert tuned.pred_opt.param_groups[0]["lr"] == 3e-4
    assert tuned.unc_opt.param_groups[0]["lr"] == 7e-4


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


def test_build_model_forwards_hyperparameters():
    m = models.build_model("lpm", INP, 5, "cpu", buffer_size=7, unc_lr=2e-3)
    assert m.buffer_size == 7
    assert m.unc_opt.param_groups[0]["lr"] == 2e-3
