import os
import random
import shutil

import numpy as np
import torch as th

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


def _find_intrinsic_wrapper(env):
    from wrappers.intrinsic_vec_wrapper import IntrinsicVecWrapper

    current = env
    while current is not None:
        if isinstance(current, IntrinsicVecWrapper):
            return current
        current = getattr(current, "venv", None)
    return None


def _resume_path(model_dir, run_name: str, slot: int, suffix: str):
    return model_dir / f"{run_name}.resume{slot}.{suffix}"


def _atomic_write_text(path, value: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(value)
    os.replace(tmp, path)


def _active_resume_slot(run_name: str, model_dir, progress: int):
    if progress <= 0:
        return None
    for slot in (0, 1):
        marker = _resume_path(model_dir, run_name, slot, "progress")
        model_path = _resume_path(model_dir, run_name, slot, "zip")
        state_path = _resume_path(model_dir, run_name, slot, "state.pt")
        try:
            slot_progress = int(marker.read_text().strip())
        except (FileNotFoundError, OSError, ValueError):
            continue
        if slot_progress == progress and model_path.exists() and state_path.exists():
            return slot
    return None


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": th.get_rng_state(),
    }
    if th.cuda.is_available():
        state["torch_cuda"] = th.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    th.set_rng_state(state["torch"])
    if "torch_cuda" in state and th.cuda.is_available():
        th.cuda.set_rng_state_all(state["torch_cuda"])


def _load_torch_checkpoint(path, device):
    try:
        return th.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch versions before weights_only was added
        return th.load(path, map_location=device)


def _sync_canonical_model(source, destination) -> None:
    tmp = destination.with_name(f"{destination.stem}.tmp.zip")
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)


def _read_progress(run_name: str, model_dir=MODELS_DIR) -> int:
    """Return steps completed so far from the sidecar file, or 0 if absent."""
    sidecar = model_dir / f"{run_name}.progress"
    if sidecar.exists():
        try:
            return int(sidecar.read_text().strip())
        except (ValueError, OSError):
            return 0
    return 0


def _write_progress(run_name: str, steps: int, model_dir=MODELS_DIR) -> None:
    """Overwrite the sidecar file with the new progress value."""
    sidecar = model_dir / f"{run_name}.progress"
    _atomic_write_text(sidecar, str(steps))


def _save_resume_checkpoint(
    model,
    env,
    run_name: str,
    model_dir,
    completed_steps: int,
    active_slot,
    expect_intrinsic: bool,
):
    """Commit PPO and curiosity state together, then advance progress atomically."""
    slot = 0 if active_slot is None else 1 - active_slot
    model_path = _resume_path(model_dir, run_name, slot, "zip")
    state_path = _resume_path(model_dir, run_name, slot, "state.pt")
    marker_path = _resume_path(model_dir, run_name, slot, "progress")
    tmp_model = model_path.with_name(f"{model_path.stem}.tmp.zip")
    tmp_state = state_path.with_name(f"{state_path.name}.tmp")

    wrapper = _find_intrinsic_wrapper(env)
    if expect_intrinsic and wrapper is None:
        raise RuntimeError("Intrinsic run has no shared intrinsic wrapper to checkpoint")
    state = {
        "version": 1,
        "progress": int(completed_steps),
        "intrinsic": None if wrapper is None else wrapper.checkpoint_state(),
        "rng": _rng_state(),
    }

    try:
        model.save(str(tmp_model))
        th.save(state, tmp_state)
        os.replace(tmp_model, model_path)
        os.replace(tmp_state, state_path)
        _atomic_write_text(marker_path, str(completed_steps))
        _write_progress(run_name, completed_steps, model_dir=model_dir)
        _sync_canonical_model(model_path, model_dir / f"{run_name}.zip")
    finally:
        for path in (tmp_model, tmp_state):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return slot


def _restore_resume_state(env, state_path, progress: int, device, expect_intrinsic: bool):
    state = _load_torch_checkpoint(state_path, device)
    if state.get("version") != 1 or int(state.get("progress", -1)) != progress:
        raise RuntimeError(
            f"Resume state {state_path} does not match committed progress {progress}"
        )
    wrapper = _find_intrinsic_wrapper(env)
    saved_intrinsic = state.get("intrinsic")
    if expect_intrinsic:
        if wrapper is None or saved_intrinsic is None:
            raise RuntimeError("Committed checkpoint is missing intrinsic-reward state")
        wrapper.load_checkpoint_state(saved_intrinsic)
    elif saved_intrinsic is not None:
        raise RuntimeError("Baseline checkpoint unexpectedly contains intrinsic-reward state")
    _restore_rng_state(state["rng"])


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
    progress = _read_progress(run_name, model_dir=model_dir)
    active_slot = _active_resume_slot(run_name, model_dir, progress)
    ckpt_path = model_dir / f"{run_name}.zip"
    if progress >= total_timesteps:
        # Cell already complete -- nothing to do.
        if active_slot is not None:
            _sync_canonical_model(
                _resume_path(model_dir, run_name, active_slot, "zip"), ckpt_path
            )
        return ckpt_path
    if progress > 0 and active_slot is None:
        raise RuntimeError(
            f"{run_name} has progress={progress} but no complete-state resume slot. "
            "This is a legacy partial run and cannot be resumed safely; use a fresh output directory."
        )

    algorithm_config = get_algorithm_config(method)
    if ALGORITHM_NAME == "ppo" and method == "entropy":
        algorithm_config["hyperparams"] = {
            **algorithm_config["hyperparams"], "ent_coef": ENTROPY_COEF,
        }
    n_envs = algorithm_config["n_envs"]

    # A resumed process starts new episodes, so give each chunk a deterministic
    # but distinct training RNG stream instead of replaying seed..seed+n_envs.
    chunk_seed = seed + progress
    monitor_run_name = f"{run_name}__c{progress}"
    if n_envs == 1:
        env = make_env(
            env_id=env_id,
            intrinsic=intrinsic,
            noise=noise,
            seed=chunk_seed,
            noise_prob=noise_prob,
            training=True,
            method=method,
            beta=beta,
        )
        env = Monitor(env, filename=str(log_dir / monitor_run_name),
                      info_keywords=("ep_extrinsic", "ep_intrinsic"))
    else:
        env = make_vector_env(
            env_id=env_id,
            intrinsic=intrinsic,
            noise=noise,
            seed=chunk_seed,
            training=True,
            n_envs=n_envs,
            log_dir=log_dir,
            run_name=monitor_run_name,
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

    # --- Load a committed complete-state checkpoint or build fresh ---
    model_class = algorithm_config["class"]
    device = algorithm_config["hyperparams"].get("device", "auto")

    if progress > 0:
        resume_model_path = _resume_path(model_dir, run_name, active_slot, "zip")
        resume_state_path = _resume_path(model_dir, run_name, active_slot, "state.pt")
        model = model_class.load(str(resume_model_path), env=env, device=device)
        _restore_resume_state(
            env,
            resume_state_path,
            progress=progress,
            device="cpu",
            expect_intrinsic=intrinsic,
        )
        # A restarted process cannot continue the exact in-flight environment
        # transition. Force SB3 to reset all envs while retaining PPO, curiosity,
        # optimizer, normalization, buffer, and RNG state.
        model._last_obs = None
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

    # One process handles one checkpoint interval so the external worker reaper
    # cannot destroy a long cell. The next process restores the committed PPO
    # and intrinsic learning state before advancing another interval.
    try:
        this_chunk = min(chunk_steps, total_timesteps - progress)
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

        completed_steps = progress + this_chunk
        _save_resume_checkpoint(
            model=model,
            env=env,
            run_name=run_name,
            model_dir=model_dir,
            completed_steps=completed_steps,
            active_slot=active_slot,
            expect_intrinsic=intrinsic,
        )
    finally:
        env.close()
        eval_env.close()

    return ckpt_path
