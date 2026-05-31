"""Launch the Ms Pac-Man exploration-comparison grid sequentially (one run at a time on MPS).

Each run -> a CSV at results/<run_id>.csv. Use --dry-run to print the plan, --only SUBSTR
to filter run_ids, --num-env-steps to override (default 1,000,000).
"""
import argparse, itertools, os, subprocess, sys

ATARI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(os.path.dirname(ATARI), ".venv", "bin", "python")  # LPM_exploration/.venv

# method label -> (algo, extra args)
METHODS = {
    "lpm":     ("ppo-improvement", ["--beta", "1"]),
    "rnd":     ("rnd",             ["--beta", "1"]),
    "icm":     ("icm",             ["--beta", "1"]),
    "ama":     ("ama",             ["--beta", "1"]),
    "ppo":     ("ppo",             ["--epsilon", "0.0"]),     # softmax baseline
    "egreedy": ("ppo",             ["--epsilon", "0.1"]),     # epsilon-greedy
}
BASE = ["--env-name", "MsPacmanNoFrameskip-v4", "--use-gae", "--lr", "1e-4",
        "--clip-param", "0.1", "--value-loss-coef", "0.5", "--num-processes", "16",
        "--num-steps", "128", "--num-mini-batch", "8", "--ppo-epoch", "3",
        "--entropy-coef", "0.001", "--log-interval", "1", "--save-interval", "100000"]
SEEDS = [1, 2, 3]

def rq1_runs():
    for m, cond in itertools.product(METHODS, ["clean", "noisy"]):
        for s in SEEDS:
            yield (f"rq1-{m}-{cond}-s{s}", m, cond, None, s)

def rq2_runs():
    for m, sp in itertools.product(["lpm", "rnd", "icm", "ppo"], [0.25, 0.5]):
        for s in SEEDS:
            yield (f"rq2-{m}-sticky{sp}-s{s}", m, "clean", sp, s)

def build_cmd(run_id, method, cond, sticky, seed, steps, results_dir):
    algo, extra = METHODS[method]
    csv = os.path.join(results_dir, run_id + ".csv")
    cmd = [PY, os.path.join(ATARI, "main.py"), "--algo", algo, *BASE, *extra,
           "--num-env-steps", str(steps), "--seed", str(seed), "--csv-log", csv,
           "--log-dir", f"/tmp/exp_logs/{run_id}", "--save-dir", f"/tmp/exp_models/{run_id}"]
    if cond == "noisy":
        cmd += ["--noisy", "true", "--randop", "2"]
    if sticky is not None:
        cmd += ["--sticky-prob", str(sticky)]
    return run_id, csv, cmd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["rq1", "rq2", "all"], default="all")
    ap.add_argument("--num-env-steps", type=int, default=1_000_000)
    ap.add_argument("--results-dir", default=os.path.join(ATARI, "experiments", "results"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(a.results_dir, exist_ok=True)
    runs = []
    if a.which in ("rq1", "all"): runs += list(rq1_runs())
    if a.which in ("rq2", "all"): runs += list(rq2_runs())
    plan = [build_cmd(rid, m, c, sp, s, a.num_env_steps, a.results_dir)
            for (rid, m, c, sp, s) in runs if (a.only is None or a.only in rid)]
    print(f"{len(plan)} runs:")
    for rid, csv, _ in plan: print("  ", rid)
    if a.dry_run: return
    env = {**os.environ, "PYTHONPATH": ATARI, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    for i, (rid, csv, cmd) in enumerate(plan, 1):
        if os.path.exists(csv) and sum(1 for _ in open(csv)) > 2:
            print(f"[{i}/{len(plan)}] SKIP {rid} (csv exists)"); continue
        print(f"[{i}/{len(plan)}] RUN {rid}")
        r = subprocess.run(cmd, env=env, cwd=ATARI,
                           stdout=open(f"/tmp/exp_logs/{rid}.stdout", "w"),
                           stderr=subprocess.STDOUT)
        print(f"    exit {r.returncode}")

if __name__ == "__main__":
    main()
