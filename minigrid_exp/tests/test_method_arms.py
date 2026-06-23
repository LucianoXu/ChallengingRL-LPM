import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

from wrappers.env_factory import make_env
from wrappers.rnd_wrapper import RNDIntrinsicRewardWrapper
from wrappers.lpm_wrapper import LPMIntrinsicRewardWrapper


def _has_wrapper(env, cls):
    inner = env
    while True:
        if isinstance(inner, cls):
            return True
        if not hasattr(inner, "env"):
            return False
        inner = inner.env


def test_rnd_lstm_builds_rnd_wrapper():
    env = make_env("MiniGrid-Empty-8x8-v0", intrinsic=True, training=True,
                   method="rnd_lstm", beta=0.005)
    assert _has_wrapper(env, RNDIntrinsicRewardWrapper)


def test_lpm_lstm_builds_lpm_wrapper():
    env = make_env("MiniGrid-Empty-8x8-v0", intrinsic=True, training=True,
                   method="lpm_lstm", beta=0.001)
    assert _has_wrapper(env, LPMIntrinsicRewardWrapper)


def test_recurrent_ppo_constructs_and_steps():
    # Full integration: RND wrapper (base of rnd_lstm) + RecurrentPPO +
    # MlpLstmPolicy + shared-LSTM kwargs all instantiate and train one rollout.
    env = DummyVecEnv([lambda: make_env(
        "MiniGrid-Empty-8x8-v0", intrinsic=True, training=True,
        method="rnd_lstm", beta=0.005)])
    model = RecurrentPPO(
        "MlpLstmPolicy", env, n_steps=64, batch_size=64, verbose=0,
        policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1,
                           shared_lstm=True, enable_critic_lstm=False))
    model.learn(total_timesteps=64)
