from stable_baselines3 import DQN, PPO

from config import ALGORITHM_NAME, DQN_EXPLORATION_STRATEGY
from method_utils import is_recurrent
from ucb_dqn import UCBDQN


def get_dqn_class():
    if DQN_EXPLORATION_STRATEGY == "epsilon_greedy":
        return DQN
    if DQN_EXPLORATION_STRATEGY == "ucb":
        return UCBDQN
    raise ValueError(
        f"Unsupported DQN exploration strategy: {DQN_EXPLORATION_STRATEGY}"
    )


def get_algorithm_class(method: str = "none"):
    if ALGORITHM_NAME == "ppo":
        if is_recurrent(method):
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO
        return PPO
    if ALGORITHM_NAME == "dqn":
        return get_dqn_class()
    raise ValueError(f"Unsupported algorithm: {ALGORITHM_NAME}")
