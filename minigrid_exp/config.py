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

# One (more interesting) environment per difficulty tier.
ENVIRONMENTS = {
    "easy": ["MiniGrid-DoorKey-5x5-v0"],
    "medium": ["MiniGrid-FourRooms-v0"],
    # MultiRoom-N6: long-horizon navigation (6-room chain to a sparse goal),
    # memory-free (doors just toggle open, no key) so an MLP can solve it, but
    # exploration-hard. KeyCorridorS3R3 was unsolvable by the MLP agent (needs
    # key-carry memory -> 0 reward for all methods at 3M).
    "hard": ["MiniGrid-MultiRoom-N6-v0"],
}

SEEDS = [1, 2, 3]

TOTAL_TIMESTEPS = 5_000_000
PARALLEL_WORKERS = 2
CHUNK_STEPS = 300_000

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
# beta sweep (FourRooms, 500k): 0.05 catastrophically drowns the sparse extrinsic
# signal (return -> 0); usable range ~0.001-0.005. RND peaks ~0.005.
RND_REWARD_SCALE = 0.005
RND_LEARNING_RATE = 1e-4
RND_HIDDEN_DIM = 128
RND_OUTPUT_DIM = 128
RND_NORMALIZE_OBSERVATIONS = True
RND_NORMALIZE_REWARDS = True
RND_OBSERVATION_CLIP = 5.0
RND_DEVICE = "auto"
# LPM declines monotonically with beta on FourRooms; use a small value.
LPM_REWARD_SCALE = 0.001
LPM_LEARNING_RATE = 1e-3
LPM_HIDDEN_DIM = 128
LPM_BUFFER_SIZE = 100
# PPO entropy-coefficient for the non-intrinsic "entropy" exploration arm
# (exploration via policy stochasticity; no reward shaping). none-arm uses 0.0.
ENTROPY_COEF = 0.01
# Count-based (UCB-style) exploration bonus = COUNT_REWARD_SCALE / sqrt(N(obs)).
COUNT_REWARD_SCALE = 0.05
# Disabled: keep DoorKey purely sparse (no key/door shaping) for a clean
# "does intrinsic motivation help on sparse reward" comparison.
DQN_DOORKEY_SUBGOAL_REWARDS = False
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
# ImgObs (147-dim 7x7x3 view) instead of FlatObs (2835-dim, ~95% constant
# mission-string padding). Same spatial representation, ~19x lighter.
PPO_USE_FLAT_OBS = False
# Vectorized-env backend: "subproc" runs the n_envs envs (+ their per-env
# intrinsic wrappers) in parallel processes -> uses ~n_envs cores per run and
# parallelizes the LPM/RND per-step compute; "dummy" runs them sequentially in
# one process (~1 core). Use "subproc" + jobs≈cores/n_envs to saturate the box.
PPO_VEC_ENV = "subproc"
PPO_RESTRICT_ACTIONS = True
PPO_EVAL_FREQ = 5_000
PPO_EVAL_EPISODES = 10
PPO_POLICY_KWARGS = {}

# Recurrent (LSTM) policy for the rnd_lstm / lpm_lstm arms (sb3-contrib
# RecurrentPPO). shared_lstm=True + enable_critic_lstm=False: one LSTM feeds
# actor+critic (~halves recurrent compute vs a separate critic LSTM). hidden=128
# (obs is 147-dim; 256 is overkill).
PPO_LSTM_POLICY = "MlpLstmPolicy"
PPO_LSTM_POLICY_KWARGS = {
    "lstm_hidden_size": 128,
    "n_lstm_layers": 1,
    "shared_lstm": True,
    "enable_critic_lstm": False,
}

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
