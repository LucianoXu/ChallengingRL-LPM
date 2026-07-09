import shutil

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from algorithms import get_algorithm_class
from config import (
    ALGORITHM_NAME,
    CHUNK_STEPS,
    DQN_EVAL_EPISODES,
    DQN_EVAL_FREQ,
    DQN_EXPLORATION_STRATEGY,
    DQN_HYPERPARAMS,
    DQN_POLICY,
    DQN_POLICY_KWARGS,
    DQN_UCB_COEFFICIENT,
    DQN_UCB_STATE_ROUND_DECIMALS,
    ENTROPY_COEF,
    MODELS_DIR,
    PPO_EVAL_EPISODES,
    PPO_VEC_ENV,
    PPO_EVAL_FREQ,
    PPO_HYPERPARAMS,
    PPO_N_ENVS,
    PPO_LSTM_POLICY,
    PPO_LSTM_POLICY_KWARGS,
    PPO_POLICY,
    PPO_POLICY_KWARGS,
)
from method_utils import is_recurrent
from wrappers.env_factory import make_env


def get_algorithm_config(method: str = "none"):
    if ALGORITHM_NAME == "ppo":
        recurrent = is_recurrent(method)
        return {
            "class": get_algorithm_class(method),
            "policy": PPO_LSTM_POLICY if recurrent else PPO_POLICY,
            "policy_kwargs": dict(
                PPO_LSTM_POLICY_KWARGS if recurrent else PPO_POLICY_KWARGS),
            "hyperparams": dict(PPO_HYPERPARAMS),
            "eval_freq": max(PPO_EVAL_FREQ // PPO_N_ENVS, 1),
            "eval_episodes": PPO_EVAL_EPISODES,
            "n_envs": PPO_N_ENVS,
        }

    if ALGORITHM_NAME == "dqn":
        policy_kwargs = dict(DQN_POLICY_KWARGS)
        hyperparams = dict(DQN_HYPERPARAMS)
        if DQN_EXPLORATION_STRATEGY == "ucb":
            hyperparams.update(
                {
                    "exploration_fraction": 0.0,
                    "exploration_initial_eps": 0.0,
                    "exploration_final_eps": 0.0,
                    "ucb_coefficient": DQN_UCB_COEFFICIENT,
                    "ucb_state_round_decimals": DQN_UCB_STATE_ROUND_DECIMALS,
                }
            )

        return {
            "class": get_algorithm_class(),
            "policy": DQN_POLICY,
            "policy_kwargs": policy_kwargs,
            "hyperparams": hyperparams,
            "eval_freq": DQN_EVAL_FREQ,
            "eval_episodes": DQN_EVAL_EPISODES,
            "n_envs": 1,
        }

    raise ValueError(f"Unsupported algorithm: {ALGORITHM_NAME}")


def make_vector_env(env_id, intrinsic, noise, seed, training, n_envs, log_dir, run_name,
                    method="rnd", beta=None, noise_prob=0.10):
    from method_utils import base_intrinsic, is_intrinsic
    from wrappers.intrinsic_models import build_shared_model
    from wrappers.intrinsic_vec_wrapper import IntrinsicVecWrapper
    from config import (
        ICM_FEATURE_DIM,
        ICM_FORWARD_LOSS_WEIGHT,
        ICM_HIDDEN_DIM,
        ICM_LEARNING_RATE,
        ICM_REWARD_SCALE,
        LPM_REWARD_SCALE,
        RND_DEVICE,
        RND_REWARD_SCALE,
    )

    # RND/LPM/ICM intrinsic is handled by ONE shared IntrinsicVecWrapper over the
    # whole VecEnv (single model, per-rollout update). So the per-env envs are
    # built WITHOUT their own intrinsic wrapper. count (dormant) still goes per-env.
    vec_handled = bool(training and intrinsic and is_intrinsic(method))
    per_env_intrinsic = bool(intrinsic and not vec_handled)

    env_fns = []
    for env_index in range(n_envs):
        env_seed = seed + env_index

        def _make_env(env_seed=env_seed):
            return make_env(
                env_id=env_id,
                intrinsic=per_env_intrinsic,
                noise=noise,
                seed=env_seed,
                noise_prob=noise_prob,
                training=training,
                method=method,
                beta=beta,
            )

        env_fns.append(_make_env)

    # SubprocVecEnv runs the n_envs envs in parallel processes; DummyVecEnv runs
    # them sequentially in one process. SB3 picks a safe default start method
    # (forkserver/spawn) when unspecified. The shared intrinsic model lives ABOVE
    # the VecEnv in the main process, so it sees all n_envs transitions at once.
    if PPO_VEC_ENV == "subproc" and n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    if vec_handled:
        base = base_intrinsic(method)
        default_scales = {
            "rnd": RND_REWARD_SCALE,
            "lpm": LPM_REWARD_SCALE,
            "icm": ICM_REWARD_SCALE,
        }
        scale = default_scales[base] if beta is None else beta
        model_kwargs = {}
        if base == "icm":
            model_kwargs = {
                "learning_rate": ICM_LEARNING_RATE,
                "hidden_dim": ICM_HIDDEN_DIM,
                "feature_dim": ICM_FEATURE_DIM,
                "forward_loss_weight": ICM_FORWARD_LOSS_WEIGHT,
            }
        model = build_shared_model(
            base, obs_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n, reward_scale=scale,
            device=RND_DEVICE, seed=seed, **model_kwargs)
        env = IntrinsicVecWrapper(env, model, n_steps=PPO_HYPERPARAMS["n_steps"])

    if log_dir is not None:
        return VecMonitor(env, filename=str(log_dir / run_name),
                          info_keywords=("ep_extrinsic", "ep_intrinsic"))

    return VecMonitor(env, info_keywords=("ep_extrinsic", "ep_intrinsic"))


def _read_progress(run_name: str) -> int:
    """Return steps completed so far from the sidecar file, or 0 if absent."""
    sidecar = MODELS_DIR / f"{run_name}.progress"
    if sidecar.exists():
        try:
            return int(sidecar.read_text().strip())
        except (ValueError, OSError):
            return 0
    return 0


def _write_progress(run_name: str, steps: int) -> None:
    """Overwrite the sidecar file with the new progress value."""
    sidecar = MODELS_DIR / f"{run_name}.progress"
    sidecar.write_text(str(steps))


def train_agent(
    env_id: str,
    variant_name: str,
    intrinsic: bool,
    noise: bool,
    seed: int,
    total_timesteps: int,
    log_dir,
    model_dir,
    method: str = "rnd",
    beta: float | None = None,
    tag: str | None = None,
    chunk_steps: int = CHUNK_STEPS,
    noise_prob: float = 0.10,
):
    suffix = f"__{tag}" if tag else ""
    run_name = f"{env_id}__{variant_name}__{method}__seed_{seed}{suffix}"
    run_name = run_name.replace("/", "_")

    # --- Resume / early-exit logic ---
    progress = _read_progress(run_name)
    if progress >= total_timesteps:
        # Cell already complete -- nothing to do.
        ckpt_path = model_dir / f"{run_name}.zip"
        return ckpt_path

    algorithm_config = get_algorithm_config(method)
    if ALGORITHM_NAME == "ppo" and method == "entropy":
        algorithm_config["hyperparams"] = {
            **algorithm_config["hyperparams"], "ent_coef": ENTROPY_COEF,
        }
    n_envs = algorithm_config["n_envs"]

    if n_envs == 1:
        env = make_env(
            env_id=env_id,
            intrinsic=intrinsic,
            noise=noise,
            seed=seed,
            noise_prob=noise_prob,
            training=True,
            method=method,
            beta=beta,
        )
        env = Monitor(env, filename=str(log_dir / run_name),
                      info_keywords=("ep_extrinsic", "ep_intrinsic"))
    else:
        env = make_vector_env(
            env_id=env_id,
            intrinsic=intrinsic,
            noise=noise,
            seed=seed,
            training=True,
            n_envs=n_envs,
            log_dir=log_dir,
            run_name=run_name,
            method=method,
            beta=beta,
            noise_prob=noise_prob,
        )

    eval_env = make_env(
        env_id=env_id,
        intrinsic=intrinsic,
        noise=noise,
        seed=seed + 10_000,
        noise_prob=noise_prob,
        training=False,
        method=method,
        beta=beta,
    )
    eval_env = Monitor(eval_env)

    # --- Load checkpoint or build fresh model ---
    ckpt_path = model_dir / f"{run_name}.zip"
    model_class = algorithm_config["class"]
    device = algorithm_config["hyperparams"].get("device", "auto")

    if ckpt_path.exists():
        model = model_class.load(str(ckpt_path), env=env, device=device)
        # Re-apply entropy override when resuming
        if ALGORITHM_NAME == "ppo" and method == "entropy":
            model.ent_coef = ENTROPY_COEF
    else:
        model = model_class(
            policy=algorithm_config["policy"],
            env=env,
            verbose=1,
            seed=seed,
            tensorboard_log=str(log_dir),
            policy_kwargs=algorithm_config["policy_kwargs"],
            **algorithm_config["hyperparams"],
        )

    # --- Chunk size ---
    this_chunk = min(chunk_steps, total_timesteps - progress)

    # --- Unique eval dirs per chunk ---
    chunk_tag = f"c{progress}"
    eval_log_path = log_dir / "eval" / run_name / chunk_tag
    eval_log_path.mkdir(parents=True, exist_ok=True)
    best_model_dir = model_dir / "best" / run_name / chunk_tag
    best_model_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_path),
        eval_freq=algorithm_config["eval_freq"],
        n_eval_episodes=algorithm_config["eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=this_chunk,
        reset_num_timesteps=(progress == 0),
        tb_log_name=run_name,
        callback=eval_callback,
    )

    # --- Save checkpoint and update progress ---
    model.save(str(ckpt_path))
    _write_progress(run_name, progress + this_chunk)

    env.close()
    eval_env.close()

    return ckpt_path
