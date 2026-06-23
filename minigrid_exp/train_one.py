"""Train one MiniGrid cell from the CLI (one process per run).

Example:
  PYTHONPATH=. python train_one.py --env MiniGrid-FourRooms-v0 \
      --intrinsic --noise --method lpm --beta 0.05 --seed 1 --steps 1000000
"""
import argparse

from config import CHUNK_STEPS, LOGS_DIR, MODELS_DIR, TOTAL_TIMESTEPS
from train import train_agent
from method_utils import is_intrinsic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--intrinsic", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--method", default="rnd",
                    choices=["rnd", "lpm", "rnd_lstm", "lpm_lstm", "count", "entropy", "none"])
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--noise-prob", type=float, default=0.10)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    ap.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS,
                    help="max timesteps per chunk (default: config.CHUNK_STEPS)")
    a = ap.parse_args()

    intrinsic = a.intrinsic and is_intrinsic(a.method)
    variant = f"{'intrinsic' if intrinsic else 'baseline'}_{'noise' if a.noise else 'no_noise'}"
    parts = []
    if a.beta is not None:
        parts.append(f"beta{a.beta:g}")
    if a.noise:
        parts.append(f"np{a.noise_prob:g}")
    tag = "_".join(parts) or None

    train_agent(
        env_id=a.env, variant_name=variant, intrinsic=intrinsic, noise=a.noise,
        seed=a.seed, total_timesteps=a.steps, log_dir=LOGS_DIR, model_dir=MODELS_DIR,
        method=a.method, beta=a.beta, tag=tag, chunk_steps=a.chunk_steps,
        noise_prob=a.noise_prob,
    )


if __name__ == "__main__":
    main()
