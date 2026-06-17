import shutil

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from algorithms import get_algorithm_class
from config import (
    ALGORITHM_NAME,
    DQN_EVAL_EPISODES,
    DQN_EVAL_FREQ,
    DQN_EXPLORATION_STRATEGY,
    DQN_HYPERPARAMS,
    DQN_POLICY,
    DQN_POLICY_KWARGS,
    DQN_UCB_COEFFICIENT,
    DQN_UCB_STATE_ROUND_DECIMALS,
    ENTROPY_COEF,
    PPO_EVAL_EPISODES,
    PPO_EVAL_FREQ,
    PPO_HYPERPARAMS,
    PPO_N_ENVS,
    PPO_POLICY,
    PPO_POLICY_KWARGS,
)
from wrappers.env_factory import make_env


def get_algorithm_config():
    if ALGORITHM_NAME == "ppo":
        return {
            "class": get_algorithm_class(),
            "policy": PPO_POLICY,
            "policy_kwargs": dict(PPO_POLICY_KWARGS),
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
                    method="rnd", beta=None):
    env_fns = []

    for env_index in range(n_envs):
        env_seed = seed + env_index

        def _make_env(env_seed=env_seed):
            return make_env(
                env_id=env_id,
                intrinsic=intrinsic,
                noise=noise,
                seed=env_seed,
                training=training,
                method=method,
                beta=beta,
            )

        env_fns.append(_make_env)

    env = DummyVecEnv(env_fns)
    if log_dir is not None:
        return VecMonitor(env, filename=str(log_dir / run_name),
                          info_keywords=("ep_extrinsic", "ep_intrinsic"))

    return VecMonitor(env, info_keywords=("ep_extrinsic", "ep_intrinsic"))


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
):
    suffix = f"__{tag}" if tag else ""
    run_name = f"{env_id}__{variant_name}__{method}__seed_{seed}{suffix}"
    run_name = run_name.replace("/", "_")

    algorithm_config = get_algorithm_config()
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
        )

    eval_env = make_env(
        env_id=env_id,
        intrinsic=intrinsic,
        noise=noise,
        seed=seed + 10_000,
        training=False,
        method=method,
        beta=beta,
    )
    eval_env = Monitor(eval_env)

    best_model_dir = model_dir / "best" / run_name
    best_model_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir / "eval" / run_name),
        eval_freq=algorithm_config["eval_freq"],
        n_eval_episodes=algorithm_config["eval_episodes"],
        deterministic=True,
        render=False,
    )

    model_class = algorithm_config["class"]
    model = model_class(
        policy=algorithm_config["policy"],
        env=env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(log_dir),
        policy_kwargs=algorithm_config["policy_kwargs"],
        **algorithm_config["hyperparams"],
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=run_name,
        callback=eval_callback,
    )

    model_path = model_dir / f"{run_name}.zip"
    last_model_dir = model_dir / "last"
    last_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(last_model_dir / f"{run_name}.zip")

    best_model_path = best_model_dir / "best_model.zip"
    if best_model_path.exists():
        shutil.copy2(best_model_path, model_path)
    else:
        model.save(model_path)

    env.close()
    eval_env.close()

    return model_path
