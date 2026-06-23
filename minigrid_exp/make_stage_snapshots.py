"""Train the curated GIF configs, dumping per-stage checkpoint snapshots.

Each config is trained once (one representative seed) and saved at three
milestones -- untrained (step0) / mid / final -- into
`expr_data/minigrid/results/models/ppo_gif_snapshots/`. These feed `gif_gallery.py`.

Parallelism mirrors run_grid.py: one subprocess per config (threads block on
subprocess.run), which sidesteps multiprocessing fragility in background runs.

Usage:
  PYTHONPATH=. python make_stage_snapshots.py                 # all configs, parallel
  PYTHONPATH=. python make_stage_snapshots.py --slug doorkey-5x5_noisy_lpm   # one, in-process
  PYTHONPATH=. python make_stage_snapshots.py --jobs 6
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gif_config import GIF_CONFIGS, get_config, SNAPSHOTS_DIR, NOISE_PROB  # noqa: E402

EXP = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(EXP), "LPM_exploration", ".venv", "bin", "python")


def train_one(cfg) -> None:
    """In-process: build the training env/model and dump the three snapshots."""
    # Imported lazily so the parallel launcher doesn't pull in torch/SB3.
    from train import get_algorithm_config, make_vector_env

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.snapshot_path(cfg.final_steps).exists():
        print(f"[skip] {cfg.slug}: final snapshot exists", flush=True)
        return

    algo = get_algorithm_config()
    n_envs = algo["n_envs"]
    env = make_vector_env(
        env_id=cfg.env_id, intrinsic=cfg.intrinsic, noise=cfg.noise,
        seed=cfg.seed, training=True, n_envs=n_envs, log_dir=None,
        run_name=cfg.run_name, method=cfg.method, beta=None, noise_prob=NOISE_PROB,
    )
    model = algo["class"](
        policy=algo["policy"], env=env, verbose=0, seed=cfg.seed,
        policy_kwargs=algo["policy_kwargs"], **algo["hyperparams"],
    )

    # Stage 0: untrained policy.
    model.save(str(cfg.snapshot_path(0)))
    print(f"[{cfg.slug}] saved step0", flush=True)

    # Stage mid, then final (continue without resetting the step counter).
    model.learn(total_timesteps=cfg.mid_steps, reset_num_timesteps=True)
    model.save(str(cfg.snapshot_path(cfg.mid_steps)))
    print(f"[{cfg.slug}] saved step{cfg.mid_steps}", flush=True)

    model.learn(total_timesteps=cfg.final_steps - cfg.mid_steps,
                reset_num_timesteps=False)
    model.save(str(cfg.snapshot_path(cfg.final_steps)))
    print(f"[{cfg.slug}] saved step{cfg.final_steps}", flush=True)
    env.close()

    _report_outcome(cfg)


def _report_outcome(cfg) -> None:
    """Print deterministic eval returns so the human can sanity-check the
    documented outcome (and bump `seed` in gif_config.py if it disagrees)."""
    from gif_gallery import build_render_env

    final = cfg.snapshot_path(cfg.final_steps)
    from algorithms import get_algorithm_class
    model = get_algorithm_class().load(str(final))

    returns = []
    for s in (cfg.render_seed, cfg.render_seed + 1, cfg.render_seed + 2,
              cfg.render_seed + 3, cfg.render_seed + 4):
        env, _mg, _cap = build_render_env(cfg.env_id, cfg.noise, NOISE_PROB, s)
        obs, _ = env.reset(seed=s)
        done, ep_r, t = False, 0.0, 0
        while not done and t < 400:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            ep_r += float(r); done = term or trunc; t += 1
        env.close()
        returns.append(ep_r)
    gif_layout = returns[0]
    mean = sum(returns) / len(returns)
    print(f"[{cfg.slug}] OUTCOME: gif-layout return={gif_layout:.3f}, "
          f"mean(5 seeds)={mean:.3f}  | expected: {cfg.expected_outcome}", flush=True)


def _launch_parallel(configs, jobs: int) -> None:
    """One subprocess per config; pin OMP threads so jobs*n_envs fits the box."""
    os.makedirs("/tmp/minigrid_gif_logs", exist_ok=True)
    tvars = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}

    def run(cfg):
        import subprocess
        log = f"/tmp/minigrid_gif_logs/{cfg.slug}.out"
        with open(log, "w") as fh:
            rc = subprocess.run(
                [PY, os.path.join(EXP, "make_stage_snapshots.py"), "--slug", cfg.slug],
                cwd=EXP, stdout=fh, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": EXP, **tvars},
            ).returncode
        return cfg.slug, rc, log

    print(f"{len(configs)} configs, jobs={jobs}", flush=True)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(run, c) for c in configs]
        for fut in as_completed(futs):
            slug, rc, log = fut.result()
            ok = rc == 0
            done += ok; failed += (not ok)
            print(f"[{'done' if ok else 'FAIL'}] {slug}  (log: {log})", flush=True)
    print(f"=== snapshots finished: {done} done, {failed} failed ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=None, help="train ONE config in-process")
    ap.add_argument("--jobs", type=int, default=len(GIF_CONFIGS))
    a = ap.parse_args()

    if a.slug:
        train_one(get_config(a.slug))
    else:
        _launch_parallel(GIF_CONFIGS, a.jobs)


if __name__ == "__main__":
    main()
