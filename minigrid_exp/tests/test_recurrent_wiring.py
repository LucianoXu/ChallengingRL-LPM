import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import train


def test_mlp_method_uses_ppo():
    cfg = train.get_algorithm_config("rnd")
    assert cfg["class"] is PPO
    assert cfg["policy"] == "MlpPolicy"


def test_recurrent_method_uses_recurrent_ppo():
    cfg = train.get_algorithm_config("rnd_lstm")
    assert cfg["class"] is RecurrentPPO
    assert cfg["policy"] == "MlpLstmPolicy"
    assert cfg["policy_kwargs"]["shared_lstm"] is True
    assert cfg["policy_kwargs"]["enable_critic_lstm"] is False
    assert cfg["policy_kwargs"]["lstm_hidden_size"] == 128
    assert cfg["policy_kwargs"]["n_lstm_layers"] == 1
