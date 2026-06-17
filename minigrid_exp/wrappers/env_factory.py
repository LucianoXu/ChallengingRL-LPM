import gymnasium as gym
import minigrid
from gymnasium import spaces
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FlattenObservation

from config import (
    ALGORITHM_NAME,
    DQN_DOOR_OPEN_BONUS,
    DQN_DOORKEY_SUBGOAL_REWARDS,
    DQN_DEFAULT_ACTIONS,
    DQN_EMPTY_ACTIONS,
    DQN_KEY_PICKUP_BONUS,
    DQN_RESTRICT_ACTIONS,
    DQN_USE_FLAT_OBS,
    LPM_BUFFER_SIZE,
    LPM_HIDDEN_DIM,
    LPM_LEARNING_RATE,
    LPM_REWARD_SCALE,
    PPO_RESTRICT_ACTIONS,
    PPO_USE_FLAT_OBS,
    RND_DEVICE,
    RND_HIDDEN_DIM,
    RND_LEARNING_RATE,
    RND_NORMALIZE_OBSERVATIONS,
    RND_NORMALIZE_REWARDS,
    RND_OBSERVATION_CLIP,
    RND_OUTPUT_DIM,
    RND_REWARD_SCALE,
)
from wrappers.noise_wrapper import ObservationNoiseWrapper
from wrappers.rnd_wrapper import RNDIntrinsicRewardWrapper
from wrappers.lpm_wrapper import LPMIntrinsicRewardWrapper
from wrappers.reward_split_wrapper import EpisodeRewardSplitWrapper


MINIGRID_ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


class MiniGridActionSubsetWrapper(gym.ActionWrapper):
    def __init__(self, env, action_map):
        super().__init__(env)
        self.action_map = tuple(action_map)
        self.action_space = spaces.Discrete(len(self.action_map))

    def action(self, action):
        return self.action_map[int(action)]


class DoorKeySubgoalRewardWrapper(gym.Wrapper):
    def __init__(self, env, key_pickup_bonus: float, door_open_bonus: float):
        super().__init__(env)
        self.key_pickup_bonus = key_pickup_bonus
        self.door_open_bonus = door_open_bonus
        self.key_picked_up = False
        self.door_opened = False

    def reset(self, **kwargs):
        self.key_picked_up = False
        self.door_opened = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        carrying = self.unwrapped.carrying
        if not self.key_picked_up and carrying is not None and carrying.type == "key":
            reward += self.key_pickup_bonus
            self.key_picked_up = True
            info["key_pickup_bonus"] = self.key_pickup_bonus

        if not self.door_opened and self._has_open_door():
            reward += self.door_open_bonus
            self.door_opened = True
            info["door_open_bonus"] = self.door_open_bonus

        return obs, reward, terminated, truncated, info

    def _has_open_door(self):
        grid = self.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "door" and cell.is_open:
                    return True
        return False


def get_action_map(env_id: str):
    if ALGORITHM_NAME == "ppo":
        restrict_actions = PPO_RESTRICT_ACTIONS
    else:
        restrict_actions = DQN_RESTRICT_ACTIONS

    if not restrict_actions:
        return None

    if "MiniGrid-Empty" in env_id or "MiniGrid-FourRooms" in env_id:
        return DQN_EMPTY_ACTIONS

    return DQN_DEFAULT_ACTIONS


def get_action_name(env, action):
    action_map = getattr(env, "action_map", None)
    base_action = action_map[int(action)] if action_map is not None else int(action)
    return MINIGRID_ACTION_NAMES.get(base_action, str(base_action))


def make_env(
    env_id: str,
    intrinsic: bool = False,
    noise: bool = False,
    seed: int = 0,
    noise_prob: float = 0.10,
    training: bool = False,
    method: str = "rnd",
    beta: float | None = None,
):
    """
    Create MiniGrid environment with optional intrinsic reward and optional noise.
    """

    env = gym.make(env_id, render_mode="rgb_array")

    env.reset(seed=seed)

    if training and DQN_DOORKEY_SUBGOAL_REWARDS and "DoorKey" in env_id:
        env = DoorKeySubgoalRewardWrapper(
            env,
            key_pickup_bonus=DQN_KEY_PICKUP_BONUS,
            door_open_bonus=DQN_DOOR_OPEN_BONUS,
        )

    if noise:
        env = ObservationNoiseWrapper(env, noise_prob=noise_prob)

    if ALGORITHM_NAME == "ppo":
        use_flat_obs = PPO_USE_FLAT_OBS
    else:
        use_flat_obs = DQN_USE_FLAT_OBS

    if use_flat_obs:
        env = FlatObsWrapper(env)
    else:
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)

    # NOTE: the intrinsic wrapper is applied here, BEFORE MiniGridActionSubsetWrapper
    # (added further below). So the intrinsic wrappers see the FULL MiniGrid action
    # space (Discrete(7)) at construction, and receive the *mapped base* action in
    # step(). The LPM wrapper therefore one-hots the base-action space (num_actions=7),
    # not the policy's reduced subset — intentional and in-range; do not assume
    # num_actions matches the agent's action_space.n.
    if training and intrinsic:
        if method == "rnd":
            env = RNDIntrinsicRewardWrapper(
                env,
                reward_scale=RND_REWARD_SCALE if beta is None else beta,
                learning_rate=RND_LEARNING_RATE,
                hidden_dim=RND_HIDDEN_DIM,
                output_dim=RND_OUTPUT_DIM,
                normalize_observations=RND_NORMALIZE_OBSERVATIONS,
                normalize_rewards=RND_NORMALIZE_REWARDS,
                observation_clip=RND_OBSERVATION_CLIP,
                device=RND_DEVICE,
                seed=seed,
            )
        elif method == "lpm":
            env = LPMIntrinsicRewardWrapper(
                env,
                reward_scale=LPM_REWARD_SCALE if beta is None else beta,
                learning_rate=LPM_LEARNING_RATE,
                hidden_dim=LPM_HIDDEN_DIM,
                buffer_size=LPM_BUFFER_SIZE,
                normalize_observations=RND_NORMALIZE_OBSERVATIONS,
                normalize_rewards=RND_NORMALIZE_REWARDS,
                observation_clip=RND_OBSERVATION_CLIP,
                device=RND_DEVICE,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported intrinsic reward method: {method}")

    action_map = get_action_map(env_id)
    if action_map is not None:
        env = MiniGridActionSubsetWrapper(env, action_map)

    env = EpisodeRewardSplitWrapper(env)   # outermost: emits ep_extrinsic / ep_intrinsic
    return env
