"""A2C actor-critic extracted from the maze notebooks (verbatim policy/update),
generalized to any IntrinsicModel with uniform running-std reward normalization.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class A2CNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        from models import CNNFeatureExtractor
        self.feature_extractor = CNNFeatureExtractor(input_shape)
        fs = self.feature_extractor.feature_size
        self.fc_actor = nn.Sequential(
            nn.Linear(fs, 512), nn.ReLU(), nn.Linear(512, num_actions))
        self.fc_critic = nn.Sequential(
            nn.Linear(fs, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        f = self.feature_extractor(x)
        return self.fc_actor(f), self.fc_critic(f)


class Memory:
    def __init__(self):
        self.clear()

    def add(self, s, a, r, ir, ns, d, lp, v):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.intrinsic_rewards.append(ir); self.next_states.append(ns)
        self.dones.append(d); self.logprobs.append(lp); self.values.append(v)

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.intrinsic_rewards, self.next_states, self.dones = [], [], []
        self.logprobs, self.values = [], []

    def __len__(self):
        return len(self.states)


class RunningNorm:
    """Welford running mean/std for scalar intrinsic rewards."""

    def __init__(self):
        self.n = 0; self.mean = 0.0; self.m2 = 0.0

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self):
        return (self.m2 / self.n) ** 0.5 if self.n > 1 else 1.0

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)


class A2CAgent:
    def __init__(self, policy, num_actions, device="cpu", gamma=0.99, gae_lambda=0.95,
                 lambda_intrinsic=0.1, entropy_coef=0.05, value_loss_coef=0.5,
                 max_grad_norm=0.5, lr=0.01, normalize_intrinsic=True):
        self.policy = policy
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma; self.gae_lambda = gae_lambda
        self.lambda_intrinsic = lambda_intrinsic
        self.entropy_coef = entropy_coef; self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.RMSprop(policy.parameters(), lr=lr, alpha=0.99, eps=1e-8)
        self.memory = Memory()
        self.rn = RunningNorm()
        self.normalize_intrinsic = normalize_intrinsic

    def select_action(self, state):
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(st)
            dist = Categorical(F.softmax(logits, dim=-1))
            a = dist.sample()
            return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def compute_gae(self, rewards, values, dones, next_value=0.0):
        returns, gae = [], 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            next_value = values[i]
            returns.insert(0, gae + values[i])
        return returns

    def update(self):
        if len(self.memory) == 0:
            return {}
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        dones = np.array(self.memory.dones, dtype=float)
        ext = np.array(self.memory.rewards, dtype=float)
        intr = np.array(self.memory.intrinsic_rewards, dtype=float)
        if self.normalize_intrinsic:
            for v in intr:
                self.rn.update(float(v))
            intr = np.array([self.rn.normalize(float(v)) for v in intr])
        combined = ext + self.lambda_intrinsic * intr
        with torch.no_grad():
            last = torch.FloatTensor(self.memory.next_states[-1]).unsqueeze(0).to(self.device)
            _, lv = self.policy(last); last_value = float(lv.item())
        returns = torch.FloatTensor(
            self.compute_gae(combined, self.memory.values, dones, last_value)).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        logits, values = self.policy(states)
        dist = Categorical(F.softmax(logits, dim=-1))
        adv = returns - values.squeeze()
        policy_loss = -(dist.log_prob(actions) * adv.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        entropy = dist.entropy().mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.memory.clear()
        return {"policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item())}
