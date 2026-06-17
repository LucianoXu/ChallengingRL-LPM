from pathlib import Path
import re
import imageio.v2 as imageio

from algorithms import get_algorithm_class
from wrappers.env_factory import make_env
from config import MODELS_DIR, VARIANTS, VIDEOS_DIR


def find_variant_by_name(variant_name: str):
    for variant in VARIANTS:
        if variant["name"] == variant_name:
            return variant

    raise ValueError(f"Unknown variant name: {variant_name}")


def parse_model_filename(model_path: Path):
    """
    Example filename:

    MiniGrid-Empty-8x8-v0__baseline_no_noise__seed_1.zip
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


def record_model(
    model_path: Path,
    episodes: int = 1,
    max_steps: int = 300,
    fps: int = 5,
):
    env_id, variant_name, seed = parse_model_filename(model_path)
    variant = find_variant_by_name(variant_name)

    print("=" * 100)
    print(f"Recording model: {model_path.name}")
    print(f"Environment: {env_id}")
    print(f"Variant: {variant_name}")
    print(f"Seed: {seed}")
    print("=" * 100)

    env = make_env(
        env_id=env_id,
        intrinsic=variant["intrinsic"],
        noise=variant["noise"],
        seed=seed,
    )

    model = get_algorithm_class().load(model_path, env=env)

    frames = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False
        steps = 0

        # First frame
        frame = env.render()
        frames.append(frame)

        while not terminated and not truncated and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            frames.append(frame)

            steps += 1

        print(f"Episode {episode + 1} finished after {steps} steps.")

    env.close()

    output_name = f"{env_id}__{variant_name}__seed_{seed}.gif"
    output_name = output_name.replace("/", "_")

    output_path = VIDEOS_DIR / output_name

    imageio.mimsave(output_path, frames, fps=fps)

    print(f"Saved GIF to: {output_path}")


def record_first_available_model():
    model_paths = sorted(MODELS_DIR.glob("*.zip"))

    if not model_paths:
        print(f"No models found in {MODELS_DIR}")
        return

    record_model(model_paths[0])


def record_specific_model(model_name: str):
    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    record_model(model_path)


if __name__ == "__main__":
    # Option 1:
    # Record the first available trained model.
    record_first_available_model()

    # Option 2:
    # To record a specific model, comment the line above and use this:
    #
    # record_specific_model(
    #     r"MiniGrid-FourRooms-v0__baseline_no_noise__seed_1.zip"
    # )
