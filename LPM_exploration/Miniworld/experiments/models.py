"""Intrinsic-reward models for the maze exploration comparison.

`CNNFeatureExtractor`, the decoder, and the LPM prediction/uncertainty nets are
lifted from the upstream maze notebooks (LPM_exploration/Miniworld/*.ipynb),
generalised behind a uniform `IntrinsicModel` interface and parameterised by
`device` instead of a module global. ICM (canonical inverse+forward dynamics)
and RND are NEW additions (the notebooks contain neither) — see UPSTREAM.md.

Convention: observation frames are HWC uint8; `input_shape` is the channel-first
triple (C, H, W). The shared encoder does `x/255` then permute(0,3,1,2), so
tensors are fed as (N, H, W, C) float.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """Nature-style CNN. Verbatim from the notebooks (normalises + permutes)."""

    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def co(size, k, s):
            return (size - k) // s + 1
        cw = co(co(co(input_shape[2], 8, 4), 4, 2), 3, 1)
        ch = co(co(co(input_shape[1], 8, 4), 4, 2), 3, 1)
        self.feature_size = cw * ch * 64

    def forward(self, x):
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)            # (N,H,W,C) -> (N,C,H,W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.reshape(x.size(0), -1)


class IntrinsicModel:
    """Uniform interface. `reward` returns a raw scalar; the A2C agent
    normalises across methods before combining."""

    def reward(self, state, next_state, action) -> float:
        raise NotImplementedError

    def update(self, states, next_states, actions) -> dict:
        return {}


class _Decoder(nn.Module):
    """Predicts next-state pixels from (features, action). Notebook decoder."""

    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.num_actions = num_actions
        c, h, w = input_shape
        self.fe = CNNFeatureExtractor(input_shape)
        self.fwd = nn.Sequential(
            nn.Linear(self.fe.feature_size + num_actions, 512), nn.ReLU())
        ho, wo = h // 8, w // 8
        self.dec = nn.Sequential(
            nn.Linear(512, ho * wo * 64), nn.ReLU(),
            nn.Unflatten(1, (64, ho, wo)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, c, 4, 2, 1), nn.Sigmoid(),
            nn.Upsample(size=(h, w), mode="bilinear", align_corners=False),
        )

    def forward(self, state, action):
        f = self.fe(state)
        a = F.one_hot(action, self.num_actions).float()
        return self.dec(self.fwd(torch.cat([f, a], dim=1)))


class MSEModel(IntrinsicModel):
    """Raw next-state prediction error — the pure noisy-TV victim."""

    def __init__(self, input_shape, num_actions, device="cpu", lr=1e-3):
        self.device = device
        self.num_actions = num_actions
        self.net = _Decoder(input_shape, num_actions).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def reward(self, state, next_state, action):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a = torch.LongTensor([action]).to(self.device)
        with torch.no_grad():
            pred = self.net(s, a)
            tgt = ns.permute(0, 3, 1, 2) / 255.0
            return float(((tgt - pred) ** 2).mean().item())

    def update(self, states, next_states, actions):
        pred = self.net(states, actions)
        tgt = next_states.permute(0, 3, 1, 2) / 255.0
        loss = F.mse_loss(pred, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"pred_loss": float(loss.item())}


class _UncertaintyNet(nn.Module):
    """Predicts log of expected prediction error. Notebook uncertainty net."""

    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.fe = CNNFeatureExtractor(input_shape)
        self.net = nn.Sequential(
            nn.Linear(self.fe.feature_size + num_actions, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, state, action):
        f = self.fe(state)
        a = F.one_hot(action, self.num_actions).float()
        out = self.net(torch.cat([f, a], dim=1))
        return torch.clamp(out, -10.0, 10.0)


class LPMModel(IntrinsicModel):
    """Learning Progress Monitoring (Hou et al. 2026).

    reward_space="log" (default, paper-faithful, Eq 1-3 + Algorithm 1):
        epsilon = log(MSE) of the dynamics model       [Eq 1]
        g_phi (the error/uncertainty net) predicts that log-error  [Eq 2]
        intrinsic reward  r = g_phi - epsilon           [Eq 3]
    a difference of log-errors with NO eta and NO clip; r is gated to 0 until the
    error queue is full (|D| = buffer_size, Alg 1 line 6), and g_phi is updated
    together with the dynamics model every update (update_unc_every=1, Alg 1).

    reward_space="raw" reproduces the form that actually shipped in the paper's
    upstream notebooks: r = min(0.5, eta*exp(g_phi) - MSE), no gating, g_phi
    updated every 5th dynamics update. Kept only to reproduce the pre-fix runs.
    """

    def __init__(self, input_shape, num_actions, device="cpu", eta=1.0,
                 buffer_size=100, update_unc_every=None, reward_space="log"):
        self.device = device
        self.num_actions = num_actions
        self.eta = eta
        self.reward_space = reward_space
        self.pred = _Decoder(input_shape, num_actions).to(device)
        self.pred_opt = optim.Adam(self.pred.parameters(), lr=1e-3)
        self.unc = _UncertaintyNet(input_shape, num_actions).to(device)
        # Error-model lr: the notebook's 1e-2 overshoots g_phi into the [-10,10]
        # clamp under the log-space objective (zero-gradient -> dead error model),
        # so the faithful log reward uses 1e-3 (the only lr C.2 specifies). Raw
        # mode keeps 1e-2 to reproduce the pre-fix runs exactly.
        self.unc_opt = optim.Adam(self.unc.parameters(),
                                  lr=1e-2 if reward_space == "raw" else 1e-3)
        self.buf = []  # (state, action, mse)
        self.buffer_size = buffer_size
        # Alg 1 updates f_theta and g_phi together every cycle; the legacy raw
        # notebook updated g_phi only every 5th dynamics update.
        if update_unc_every is None:
            update_unc_every = 5 if reward_space == "raw" else 1
        self.update_unc_every = update_unc_every
        self._since = 0

    def _mse(self, state, next_state, action):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a = torch.LongTensor([action]).to(self.device)
        with torch.no_grad():
            pred = self.pred(s, a)
            tgt = ns.permute(0, 3, 1, 2) / 255.0
            return float(((tgt - pred) ** 2).mean().item()), s, a

    def reward(self, state, next_state, action):
        actual, s, a = self._mse(state, next_state, action)
        self.buf.append((state, action, actual))
        if len(self.buf) > self.buffer_size:
            self.buf.pop(0)
        with torch.no_grad():
            g = float(self.unc(s, a).item())   # g_phi predicts the log-MSE [Eq 2]
        if self.reward_space == "raw":
            # Legacy upstream-notebook form (raw-MSE space, eta, 0.5 clip).
            return float(min(0.5, self.eta * np.exp(g) - actual))
        # Paper-faithful log-space reward (Eq 1-3): r = g_phi - log(MSE), gated
        # to 0 until the error queue is full (|D| = d, Alg 1 line 6).
        if len(self.buf) < self.buffer_size:
            return 0.0
        eps = float(np.log(actual + 1e-6))     # epsilon = log(MSE) [Eq 1]
        return float(g - eps)                  # r = g_phi - epsilon [Eq 3]

    def update(self, states, next_states, actions):
        pred = self.pred(states, actions)
        tgt = next_states.permute(0, 3, 1, 2) / 255.0
        ploss = F.mse_loss(pred, tgt)
        self.pred_opt.zero_grad(); ploss.backward(); self.pred_opt.step()
        self._since += 1
        uloss = 0.0
        if self._since >= self.update_unc_every and self.buf:
            n = min(32, len(self.buf))
            idx = np.random.choice(len(self.buf), n, replace=False)
            bs = torch.FloatTensor(np.array([self.buf[i][0] for i in idx])).to(self.device)
            ba = torch.LongTensor([self.buf[i][1] for i in idx]).to(self.device)
            be = [self.buf[i][2] for i in idx]
            logp = self.unc(bs, ba)
            logt = torch.log(torch.FloatTensor(be).to(self.device) + 1e-6).unsqueeze(1)
            ul = F.mse_loss(logp, logt)
            self.unc_opt.zero_grad(); ul.backward(); self.unc_opt.step()
            uloss = float(ul.item()); self._since = 0
        return {"pred_loss": float(ploss.item()), "unc_loss": uloss}


class ICMModel(IntrinsicModel):
    """Canonical ICM: inverse + forward dynamics in feature space (NEW)."""

    def __init__(self, input_shape, num_actions, device="cpu", lr=1e-3, beta=0.2):
        self.device = device
        self.num_actions = num_actions
        self.beta = beta
        self.fe = CNNFeatureExtractor(input_shape).to(device)
        d = self.fe.feature_size
        self.inverse = nn.Sequential(
            nn.Linear(2 * d, 256), nn.ReLU(), nn.Linear(256, num_actions)).to(device)
        self.forward_net = nn.Sequential(
            nn.Linear(d + num_actions, 256), nn.ReLU(), nn.Linear(256, d)).to(device)
        params = (list(self.fe.parameters()) + list(self.inverse.parameters())
                  + list(self.forward_net.parameters()))
        self.opt = optim.Adam(params, lr=lr)

    def reward(self, state, next_state, action):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a = torch.LongTensor([action]).to(self.device)
        with torch.no_grad():
            phi, nphi = self.fe(s), self.fe(ns)
            ah = F.one_hot(a, self.num_actions).float()
            pred = self.forward_net(torch.cat([phi, ah], dim=1))
            return float(((pred - nphi) ** 2).mean().item())

    def update(self, states, next_states, actions):
        phi, nphi = self.fe(states), self.fe(next_states)
        ah = F.one_hot(actions, self.num_actions).float()
        pred = self.forward_net(torch.cat([phi, ah], dim=1))
        fwd_loss = F.mse_loss(pred, nphi.detach())
        logits = self.inverse(torch.cat([phi, nphi], dim=1))
        inv_loss = F.cross_entropy(logits, actions)
        loss = self.beta * fwd_loss + (1 - self.beta) * inv_loss
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"fwd_loss": float(fwd_loss.item()), "inv_loss": float(inv_loss.item())}


class RNDModel(IntrinsicModel):
    """Random Network Distillation: predictor matches a frozen random target (NEW)."""

    def __init__(self, input_shape, num_actions, device="cpu", lr=1e-3, emb=256):
        self.device = device
        self.target = CNNFeatureExtractor(input_shape).to(device)
        d = self.target.feature_size
        self.target_head = nn.Linear(d, emb).to(device)
        for p in list(self.target.parameters()) + list(self.target_head.parameters()):
            p.requires_grad_(False)
        self.pred = CNNFeatureExtractor(input_shape).to(device)
        self.pred_head = nn.Linear(d, emb).to(device)
        self.opt = optim.Adam(
            list(self.pred.parameters()) + list(self.pred_head.parameters()), lr=lr)

    def _t(self, x):
        return self.target_head(self.target(x))

    def _p(self, x):
        return self.pred_head(self.pred(x))

    def reward(self, state, next_state, action):
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(((self._p(ns) - self._t(ns)) ** 2).mean().item())

    def update(self, states, next_states, actions):
        with torch.no_grad():
            tgt = self._t(next_states)
        loss = F.mse_loss(self._p(next_states), tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"rnd_loss": float(loss.item())}


class NoneModel(IntrinsicModel):
    """No intrinsic reward — plain A2C random-walk baseline."""

    def __init__(self, *a, **k):
        pass

    def reward(self, state, next_state, action):
        return 0.0

    def update(self, states, next_states, actions):
        return {}


def build_model(name, input_shape, num_actions, device="cpu"):
    name = name.lower()
    if name == "lpm":
        return LPMModel(input_shape, num_actions, device)
    if name == "icm":
        return ICMModel(input_shape, num_actions, device)
    if name == "rnd":
        return RNDModel(input_shape, num_actions, device)
    if name == "mse":
        return MSEModel(input_shape, num_actions, device)
    if name == "none":
        return NoneModel()
    raise ValueError(f"unknown method {name!r}")
