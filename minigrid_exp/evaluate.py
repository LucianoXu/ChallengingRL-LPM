from pathlib import Path
import re
import pandas as pd
import numpy as np

from algorithms import get_algorithm_class
from wrappers.env_factory import make_env
from config import (
    ALGORITHM_LABEL,
    ALGORITHM_NAME,
    DQN_EXPLORATION_STRATEGY,
    EVAL_RESULTS_PATH,
    EVAL_SUMMARY_PATH,
    MODELS_DIR,
    VARIANTS,
)


EVAL_EPISODES = 20
MAX_STEPS_PER_EPISODE = 500


def evaluate_model(
    model_path: Path,
    env_id: str,
    intrinsic: bool,
    noise: bool,
    seed: int,
    n_episodes: int = EVAL_EPISODES,
    max_steps: int = MAX_STEPS_PER_EPISODE,
):
    """
    Evaluate one trained model.

    Returns:
        dict containing success rate, average reward, and average episode length.
    """

    env = make_env(
        env_id=env_id,
        intrinsic=intrinsic,
        noise=noise,
        seed=seed,
    )

    model = get_algorithm_class().load(model_path, env=env)

    episode_rewards = []
    episode_lengths = []
    successes = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # MiniGrid task completion terminates; time limits truncate.
        success = terminated
        successes.append(success)

    env.close()

    return {
        "model_path": str(model_path),
        "algorithm": ALGORITHM_NAME,
        "exploration_strategy": (
            DQN_EXPLORATION_STRATEGY if ALGORITHM_NAME == "dqn" else ""
        ),
        "env_id": env_id,
        "intrinsic": intrinsic,
        "noise": noise,
        "seed": seed,
        "eval_episodes": n_episodes,
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
    }


def find_variant_by_name(variant_name: str):
    """
    Find the variant dictionary from config.py by its name.
    """

    for variant in VARIANTS:
        if variant["name"] == variant_name:
            return variant

    raise ValueError(f"Unknown variant name: {variant_name}")


def parse_model_filename(model_path: Path):
    """
    Parse filenames like:

    MiniGrid-Empty-8x8-v0__baseline_no_noise__seed_1.zip

    Returns:
        env_id, variant_name, seed
    """

    filename = model_path.stem

    pattern = r"(.+)__(.+)__seed_(\d+)"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Could not parse model filename: {filename}")

    env_id = match.group(1)
    variant_name = match.group(2)
    seed = int(match.group(3))

    return env_id, variant_name, seed


def evaluate_all_models():
    """
    Evaluate every trained model inside the configured model directory.
    """

    model_paths = sorted(MODELS_DIR.glob("*.zip"))

    if not model_paths:
        print(f"No trained models found in: {MODELS_DIR}")
        return

    all_results = []

    for model_path in model_paths:
        print("=" * 100)
        print(f"Evaluating model: {model_path.name}")

        try:
            env_id, variant_name, seed = parse_model_filename(model_path)
            variant = find_variant_by_name(variant_name)

            result = evaluate_model(
                model_path=model_path,
                env_id=env_id,
                intrinsic=variant["intrinsic"],
                noise=variant["noise"],
                seed=seed,
            )

            result["variant"] = variant_name
            all_results.append(result)

            print(f"Environment: {env_id}")
            print(f"Variant: {variant_name}")
            print(f"Seed: {seed}")
            print(f"Success rate: {result['success_rate']:.2f}")
            print(f"Average reward: {result['avg_reward']:.3f}")
            print(f"Average episode length: {result['avg_episode_length']:.2f}")

        except Exception as e:
            print(f"Failed to evaluate {model_path.name}")
            print(f"Reason: {e}")

    if all_results:
        df = pd.DataFrame(all_results)

        column_order = [
            "env_id",
            "algorithm",
            "exploration_strategy",
            "variant",
            "intrinsic",
            "noise",
            "seed",
            "eval_episodes",
            "success_rate",
            "avg_reward",
            "std_reward",
            "avg_episode_length",
            "std_episode_length",
            "model_path",
        ]

        df = df[column_order]

        df.to_csv(EVAL_RESULTS_PATH, index=False)

        print("=" * 100)
        print(f"Saved evaluation results to: {EVAL_RESULTS_PATH}")


def summarize_results():
    """
    Create a grouped summary over seeds.
    """

    if not EVAL_RESULTS_PATH.exists():
        print("No evaluation CSV found. Run evaluate_all_models() first.")
        return

    df = pd.read_csv(EVAL_RESULTS_PATH)

    summary = (
        df.groupby(
            [
                "algorithm",
                "exploration_strategy",
                "env_id",
                "variant",
                "intrinsic",
                "noise",
            ]
        )
        .agg(
            mean_success_rate=("success_rate", "mean"),
            std_success_rate=("success_rate", "std"),
            mean_reward=("avg_reward", "mean"),
            std_reward=("avg_reward", "std"),
            mean_episode_length=("avg_episode_length", "mean"),
            std_episode_length=("avg_episode_length", "std"),
        )
        .reset_index()
    )

    summary.to_csv(EVAL_SUMMARY_PATH, index=False)

    print("=" * 100)
    print(f"Saved {ALGORITHM_LABEL} summary results to: {EVAL_SUMMARY_PATH}")
    print(summary)


if __name__ == "__main__":
    evaluate_all_models()
    summarize_results()
