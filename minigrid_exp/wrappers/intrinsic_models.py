from __future__ import annotations

import numpy as np
import torch as th

from wrappers.rnd_wrapper import RunningMeanStd, RNDNetwork
from wrappers.lpm_wrapper import _MLP


def _resolve_device(device):
    if device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


class _SharedBase:
    """Common obs-normalization + reward-normalization for shared intrinsic models.

    A single instance is owned by IntrinsicVecWrapper and fed ALL n_envs
    transitions each step (so obs_rms/reward_rms are global, unlike the old
    per-env wrappers). reward() is called every step (no-grad) and stashes the
    rollout's transitions; update() trains once per rollout and clears the stash.
    """

    def __init__(self, obs_dim, reward_scale, normalize_observations,
                 normalize_rewards, observation_clip, train_epochs, train_batch,
                 device, seed):
        self.obs_dim = int(obs_dim)
        self.reward_scale = float(reward_scale)
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.observation_clip = observation_clip
        self.train_epochs = int(train_epochs)
        self.train_batch = int(train_batch)
        self.device = _resolve_device(device)
        if seed is not None:
            th.manual_seed(seed)
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.reward_rms = RunningMeanStd(shape=())

    def _normalize_obs(self, flat):  # flat: (B, obs_dim) float32
        if not self.normalize_observations:
            return flat.astype(np.float32)
        z = (flat - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(z, -self.observation_clip, self.observation_clip).astype(np.float32)

    def _normalize_reward(self, raw):  # raw: (B,) float64
        if not self.normalize_rewards:
            return raw.astype(np.float32)
        self.reward_rms.update(raw)
        return (raw / np.sqrt(self.reward_rms.var + 1e-8)).astype(np.float32)

    def _minibatches(self, n):
        idx = np.arange(n)
        for _ in range(self.train_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.train_batch):
                yield idx[s:s + self.train_batch]


class SharedRNDModel(_SharedBase):
    def __init__(self, obs_dim, num_actions, reward_scale, learning_rate=1e-4,
                 hidden_dim=128, output_dim=128, normalize_observations=True,
                 normalize_rewards=True, observation_clip=5.0, train_epochs=4,
                 train_batch=256, device="cpu", seed=None):
        super().__init__(obs_dim, reward_scale, normalize_observations,
                         normalize_rewards, observation_clip, train_epochs,
                         train_batch, device, seed)
        self.target = RNDNetwork(self.obs_dim, hidden_dim, output_dim).to(self.device)
        self.predictor = RNDNetwork(self.obs_dim, hidden_dim, output_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = th.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self._rollout_next = []  # list of (B, obs_dim) normalized-next-obs arrays

    def reward(self, prev_obs, actions, next_obs):
        flat = np.asarray(next_obs, dtype=np.float32)
        self.obs_rms.update(flat)
        norm = self._normalize_obs(flat)
        t = th.as_tensor(norm, dtype=th.float32, device=self.device)
        with th.no_grad():
            raw = ((self.predictor(t) - self.target(t)) ** 2).mean(dim=1).cpu().numpy()
        self._rollout_next.append(norm)
        bonus = self._normalize_reward(raw.astype(np.float64))
        return (self.reward_scale * bonus).astype(np.float32)

    def update(self):
        if not self._rollout_next:
            return {"loss": 0.0}
        X = np.concatenate(self._rollout_next, axis=0)
        self._rollout_next = []
        xt = th.as_tensor(X, dtype=th.float32, device=self.device)
        with th.no_grad():
            tgt = self.target(xt)
        last = 0.0
        for mb in self._minibatches(X.shape[0]):
            loss = ((self.predictor(xt[mb]) - tgt[mb]) ** 2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            last = float(loss.item())
        return {"loss": last}


class SharedLPMModel(_SharedBase):
    def __init__(self, obs_dim, num_actions, reward_scale, learning_rate=1e-3,
                 hidden_dim=128, buffer_size=100, normalize_observations=True,
                 normalize_rewards=True, observation_clip=5.0, train_epochs=4,
                 train_batch=256, device="cpu", seed=None):
        super().__init__(obs_dim, reward_scale, normalize_observations,
                         normalize_rewards, observation_clip, train_epochs,
                         train_batch, device, seed)
        self.num_actions = int(num_actions)
        self.forward_model = _MLP(self.obs_dim + self.num_actions, hidden_dim, self.obs_dim).to(self.device)
        self.error_model = _MLP(self.obs_dim + self.num_actions, hidden_dim, 1).to(self.device)
        self.fwd_opt = th.optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        self.err_opt = th.optim.Adam(self.error_model.parameters(), lr=learning_rate)
        self.buffer_size = int(buffer_size)
        self.buf = []                 # (norm_prev: (obs_dim,), action:int, mse:float)
        self._rollout = []            # (norm_prev:(obs_dim,), action:int, norm_next:(obs_dim,))

    def _sa(self, norm_obs, actions):  # norm_obs:(B,obs_dim), actions:(B,)
        a = np.zeros((norm_obs.shape[0], self.num_actions), dtype=np.float32)
        a[np.arange(norm_obs.shape[0]), np.asarray(actions, dtype=int)] = 1.0
        return th.as_tensor(np.concatenate([norm_obs, a], axis=1), dtype=th.float32, device=self.device)

    def reward(self, prev_obs, actions, next_obs):
        prev = np.asarray(prev_obs, dtype=np.float32)
        nxt = np.asarray(next_obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=int)
        self.obs_rms.update(nxt)
        nprev = self._normalize_obs(prev)
        nnext = self._normalize_obs(nxt)
        sa = self._sa(nprev, actions)
        target = th.as_tensor(nnext, dtype=th.float32, device=self.device)
        with th.no_grad():
            mse = ((self.forward_model(sa) - target) ** 2).mean(dim=1).cpu().numpy()
            g = th.clamp(self.error_model(sa), -10.0, 10.0).squeeze(1).cpu().numpy()
        B = prev.shape[0]
        for i in range(B):
            self.buf.append((nprev[i], int(actions[i]), float(mse[i])))
            self._rollout.append((nprev[i], int(actions[i]), nnext[i]))
        if len(self.buf) > self.buffer_size:
            self.buf = self.buf[-self.buffer_size:]
        if len(self.buf) < self.buffer_size:
            return np.zeros(B, dtype=np.float32)   # gated (Alg 1 L6)
        raw = g - np.log(mse + 1e-6)
        bonus = self._normalize_reward(raw.astype(np.float64))
        return (self.reward_scale * bonus).astype(np.float32)

    def update(self):
        fwd_last = 0.0
        if self._rollout:
            bs = np.stack([r[0] for r in self._rollout])
            ba = np.array([r[1] for r in self._rollout], dtype=int)
            bn = np.stack([r[2] for r in self._rollout])
            self._rollout = []
            sa = self._sa(bs, ba)
            tgt = th.as_tensor(bn, dtype=th.float32, device=self.device)
            for mb in self._minibatches(bs.shape[0]):
                loss = ((self.forward_model(sa[mb]) - tgt[mb]) ** 2).mean()
                self.fwd_opt.zero_grad(); loss.backward(); self.fwd_opt.step()
                fwd_last = float(loss.item())
        err_last = 0.0
        if self.buf:
            for _ in range(self.train_epochs):
                n = min(32, len(self.buf))
                idx = np.random.choice(len(self.buf), n, replace=False)
                bs = np.stack([self.buf[i][0] for i in idx])
                ba = np.array([self.buf[i][1] for i in idx], dtype=int)
                be = np.array([self.buf[i][2] for i in idx], dtype=np.float32)
                logp = th.clamp(self.error_model(self._sa(bs, ba)), -10.0, 10.0)
                logt = th.log(th.as_tensor(be, device=self.device) + 1e-6).unsqueeze(1)
                loss = ((logp - logt) ** 2).mean()
                self.err_opt.zero_grad(); loss.backward(); self.err_opt.step()
                err_last = float(loss.item())
        return {"fwd_loss": fwd_last, "err_loss": err_last}


class SharedICMModel(_SharedBase):
    """Intrinsic Curiosity Module for the shared PPO vector-wrapper path.

    Reward is forward-model prediction error in inverse-dynamics feature space:
    mean((phi(next_obs) - f(phi(prev_obs), action))^2).
    """

    def __init__(self, obs_dim, num_actions, reward_scale, learning_rate=1e-3,
                 hidden_dim=128, feature_dim=128, forward_loss_weight=0.2,
                 normalize_observations=True, normalize_rewards=True,
                 observation_clip=5.0, train_epochs=4, train_batch=256,
                 device="cpu", seed=None):
        super().__init__(obs_dim, reward_scale, normalize_observations,
                         normalize_rewards, observation_clip, train_epochs,
                         train_batch, device, seed)
        self.num_actions = int(num_actions)
        self.forward_loss_weight = float(forward_loss_weight)
        self.encoder = _MLP(self.obs_dim, hidden_dim, feature_dim).to(self.device)
        self.forward_model = _MLP(feature_dim + self.num_actions, hidden_dim, feature_dim).to(self.device)
        self.inverse_model = _MLP(2 * feature_dim, hidden_dim, self.num_actions).to(self.device)
        params = (list(self.encoder.parameters()) + list(self.forward_model.parameters())
                  + list(self.inverse_model.parameters()))
        self.opt = th.optim.Adam(params, lr=learning_rate)
        self._rollout = []  # (norm_prev:(obs_dim,), action:int, norm_next:(obs_dim,))

    def _action_one_hot(self, actions):
        a = np.zeros((len(actions), self.num_actions), dtype=np.float32)
        a[np.arange(len(actions)), np.asarray(actions, dtype=int)] = 1.0
        return th.as_tensor(a, dtype=th.float32, device=self.device)

    def reward(self, prev_obs, actions, next_obs):
        prev = np.asarray(prev_obs, dtype=np.float32)
        nxt = np.asarray(next_obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=int)
        self.obs_rms.update(nxt)
        nprev = self._normalize_obs(prev)
        nnext = self._normalize_obs(nxt)
        pt = th.as_tensor(nprev, dtype=th.float32, device=self.device)
        nt = th.as_tensor(nnext, dtype=th.float32, device=self.device)
        ah = self._action_one_hot(actions)
        with th.no_grad():
            phi_prev = self.encoder(pt)
            phi_next = self.encoder(nt)
            pred_next = self.forward_model(th.cat([phi_prev, ah], dim=1))
            raw = ((pred_next - phi_next) ** 2).mean(dim=1).cpu().numpy()
        for i in range(prev.shape[0]):
            self._rollout.append((nprev[i], int(actions[i]), nnext[i]))
        bonus = self._normalize_reward(raw.astype(np.float64))
        return (self.reward_scale * bonus).astype(np.float32)

    def update(self):
        if not self._rollout:
            return {"fwd_loss": 0.0, "inv_loss": 0.0}
        bs = np.stack([r[0] for r in self._rollout])
        ba = np.array([r[1] for r in self._rollout], dtype=int)
        bn = np.stack([r[2] for r in self._rollout])
        self._rollout = []
        st = th.as_tensor(bs, dtype=th.float32, device=self.device)
        nt = th.as_tensor(bn, dtype=th.float32, device=self.device)
        at = th.as_tensor(ba, dtype=th.long, device=self.device)
        ah = self._action_one_hot(ba)
        fwd_last = inv_last = 0.0
        for mb in self._minibatches(bs.shape[0]):
            phi_prev = self.encoder(st[mb])
            phi_next = self.encoder(nt[mb])
            inv_logits = self.inverse_model(th.cat([phi_prev, phi_next], dim=1))
            inv_loss = th.nn.functional.cross_entropy(inv_logits, at[mb])
            pred_next = self.forward_model(th.cat([phi_prev, ah[mb]], dim=1))
            fwd_loss = ((pred_next - phi_next.detach()) ** 2).mean()
            loss = ((1.0 - self.forward_loss_weight) * inv_loss
                    + self.forward_loss_weight * fwd_loss)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            fwd_last = float(fwd_loss.item())
            inv_last = float(inv_loss.item())
        return {"fwd_loss": fwd_last, "inv_loss": inv_last}


def build_shared_model(method, obs_dim, num_actions, reward_scale, device="cpu", seed=None, **kwargs):
    if method == "rnd":
        return SharedRNDModel(obs_dim, num_actions, reward_scale, device=device, seed=seed, **kwargs)
    if method == "lpm":
        return SharedLPMModel(obs_dim, num_actions, reward_scale, device=device, seed=seed, **kwargs)
    if method == "icm":
        return SharedICMModel(obs_dim, num_actions, reward_scale, device=device, seed=seed, **kwargs)
    raise ValueError(f"build_shared_model: unsupported method {method!r}")
