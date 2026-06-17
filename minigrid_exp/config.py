from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
EXPR_DATA = REPO_ROOT / "expr_data" / "minigrid"

RESULTS_DIR = EXPR_DATA / "results"

ALGORITHM_NAME = "ppo"
ALGORITHM_LABELS = {
    "dqn": "DQN",
    "ppo": "PPO",
}

LOGS_DIR = RESULTS_DIR / "logs" / ALGORITHM_NAME
MODELS_DIR = RESULTS_DIR / "models" / ALGORITHM_NAME
VIDEOS_DIR = RESULTS_DIR / "videos" / ALGORITHM_NAME
PLOTS_DIR = EXPR_DATA / "figures" / ALGORITHM_NAME
EVAL_RESULTS_PATH = RESULTS_DIR / f"{ALGORITHM_NAME}_evaluation_results.csv"
EVAL_SUMMARY_PATH = RESULTS_DIR / f"{ALGORITHM_NAME}_evaluation_summary.csv"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ENVIRONMENTS = {
    "easy": [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-DoorKey-5x5-v0",
    ],
    "medium": [
        "MiniGrid-FourRooms-v0",
        "MiniGrid-DoorKey-8x8-v0",
    ],
    "hard": [
        "MiniGrid-MultiRoom-N6-v0",
        "MiniGrid-KeyCorridorS3R3-v0",
    ],
}

SEEDS = [1, 2, 3]

TOTAL_TIMESTEPS = 5_000_000
PARALLEL_WORKERS = 2

DQN_POLICY = "MlpPolicy"
DQN_USE_FLAT_OBS = True
DQN_RESTRICT_ACTIONS = True
DQN_DEFAULT_ACTIONS = (0, 1, 2, 3, 5)
DQN_EMPTY_ACTIONS = (0, 1, 2)
DQN_EVAL_FREQ = 5_000
DQN_EVAL_EPISODES = 10
DQN_EXPLORATION_STRATEGY = "ucb"
DQN_EXPLORATION_LABELS = {
    "epsilon_greedy": "epsilon-greedy",
    "ucb": "UCB",
}
DQN_UCB_COEFFICIENT = 1.0
DQN_UCB_STATE_ROUND_DECIMALS = None
INTRINSIC_REWARD_METHOD = "rnd"
RND_REWARD_SCALE = 0.05
RND_LEARNING_RATE = 1e-4
RND_HIDDEN_DIM = 128
RND_OUTPUT_DIM = 128
RND_NORMALIZE_OBSERVATIONS = True
RND_NORMALIZE_REWARDS = True
RND_OBSERVATION_CLIP = 5.0
RND_DEVICE = "auto"
LPM_REWARD_SCALE = 0.05
LPM_LEARNING_RATE = 1e-3
LPM_HIDDEN_DIM = 128
LPM_BUFFER_SIZE = 100
DQN_DOORKEY_SUBGOAL_REWARDS = True
DQN_KEY_PICKUP_BONUS = 0.20
DQN_DOOR_OPEN_BONUS = 0.40
DQN_POLICY_KWARGS = {}

DQN_HYPERPARAMS = {
    "buffer_size": 100_000,
    "learning_starts": 5_000,
    "batch_size": 64,
    "gamma": 0.99,
    "learning_rate": 2.5e-4,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 500,
    "exploration_fraction": 1.00,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.20,
}

PPO_POLICY = "MlpPolicy"
PPO_N_ENVS = 8
PPO_USE_FLAT_OBS = True
PPO_RESTRICT_ACTIONS = True
PPO_EVAL_FREQ = 5_000
PPO_EVAL_EPISODES = 10
PPO_POLICY_KWARGS = {}

PPO_HYPERPARAMS = {
    "learning_rate": 2.5e-4,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
}

ALGORITHM_LABEL = ALGORITHM_LABELS[ALGORITHM_NAME]
if ALGORITHM_NAME == "dqn":
    ALGORITHM_LABEL = (
        f"{ALGORITHM_LABEL} ({DQN_EXPLORATION_LABELS[DQN_EXPLORATION_STRATEGY]})"
    )

VARIANTS = [
    {"name": "baseline_no_noise",  "intrinsic": False, "noise": False},
    {"name": "intrinsic_no_noise", "intrinsic": True,  "noise": False},
    {"name": "baseline_noise",     "intrinsic": False, "noise": True},
    {"name": "intrinsic_noise",    "intrinsic": True,  "noise": True},
]
