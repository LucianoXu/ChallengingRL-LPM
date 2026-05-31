# Ms Pac-Man Exploration-Algorithm Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and analyze a controlled comparison of 6 exploration methods (LPM, RND, ICM, AMA, plain-PPO/softmax, ε-greedy-PPO) on Ms Pac-Man under clean/noisy conditions plus a sticky-action stochasticity sweep, producing a results table, 3 figures, and an intrinsic-reward attribution analysis.

**Architecture:** All methods share the existing PPO base in `LPM_exploration/Atari/main.py` (curiosity methods plug in via `--algo`; ε-greedy via a new `--epsilon` action layer; softmax = plain PPO). Per-run metrics are written to CSV (not parsed from stdout). A grid runner launches runs sequentially on MPS; an analysis script aggregates the CSVs. Sticky-action stochasticity is controlled via ALE's `repeat_action_probability`.

**Tech Stack:** Python 3.11 (uv venv at `LPM_exploration/.venv`), PyTorch 2.12 (MPS), gymnasium + ale-py, the embedded LPM/`pytorch-a2c-ppo-acktr`-style code, pandas + matplotlib for analysis.

**Spec:** `docs/superpowers/specs/2026-05-31-pacman-exploration-comparison-design.md`

**Conventions for every run command in this plan:**
```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PY=/Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python
RUN="PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 $PY"
```
Any change inside `LPM_exploration/` must be appended to `LPM_exploration/UPSTREAM.md` under "Local additions / deviations."

---

## Phase 0 — Infrastructure (deterministic code, testable)

### Task 1: CSV per-run logging in `main.py`

**Files:**
- Modify: `LPM_exploration/Atari/main.py` (arg parsing already in `exploration/arguments.py`; logging block ~line 597-617)
- Modify: `LPM_exploration/Atari/exploration/arguments.py` (add `--csv-log`)

- [ ] **Step 1: Add the `--csv-log` argument**

In `exploration/arguments.py`, immediately after the `--noisy` argument (the last `add_argument` before `args = parser.parse_args()`), add:
```python
    parser.add_argument('--csv-log', default=None, help='path to write a per-update metrics CSV')
    parser.add_argument('--epsilon', type=float, default=0.0, help='epsilon-greedy action probability over the PPO policy (0 = pure PPO/softmax)')
    parser.add_argument('--sticky-prob', type=float, default=None, help='ALE repeat_action_probability (sticky actions); None = env default')
```

- [ ] **Step 2: Open the CSV and write the header (once), near the start of `main()`**

In `main.py`, right after `device = torch.device(...)` block (the MPS block ends with `print(f"[device] using {device}")`), add:
```python
    csv_file = None
    if args.csv_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_log)), exist_ok=True)
        csv_file = open(args.csv_log, "w", buffering=1)
        csv_file.write("update,frames,fps,ep_score_mean,ep_score_std,ep_len_mean,"
                       "int_rew,ext_rew,pred_loss,unc_loss,dist_entropy,value_loss,action_loss\n")
```

- [ ] **Step 3: Write a CSV row inside the logging block**

In the `if j % args.log_interval == 0 and len(episode_rewards) > 0 ...` block in `main.py`, after the existing `print(...)` of `IntRew/ExtRew`, add (compute intrinsic/pred losses safely since plain PPO has none):
```python
            if csv_file is not None:
                _int = float(curiosity.mean()) if train_model else 0.0
                _ext = float(ext_reward.mean()) if train_model else float(rollouts.rewards.mean())
                _pl = float(model_pred_loss) if train_model else 0.0
                _ul = float(model_uncertainty_loss) if train_model else 0.0
                csv_file.write(f"{j},{int(total_num_steps)},"
                               f"{int(args.num_steps*args.num_processes/(end-start))},"
                               f"{np.mean(episode_rewards):.3f},{np.std(episode_rewards):.3f},"
                               f"{np.mean(episode_lengths):.1f},{_int:.4f},{_ext:.4f},"
                               f"{_pl:.4f},{_ul:.4f},{float(dist_entropy):.4f},"
                               f"{float(value_loss):.4f},{float(action_loss):.4f}\n")
```

- [ ] **Step 4: Smoke-test that the CSV is produced for LPM**

Run:
```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python main.py \
  --env-name MsPacmanNoFrameskip-v4 --algo ppo-improvement --beta 1 --use-gae \
  --lr 1e-4 --clip-param 0.1 --num-processes 8 --num-steps 128 --num-mini-batch 8 \
  --ppo-epoch 3 --entropy-coef 0.001 --num-env-steps 25000 --log-interval 1 \
  --save-interval 100000 --seed 1 --csv-log /tmp/exp/smoke_lpm.csv >/dev/null 2>&1
head -3 /tmp/exp/smoke_lpm.csv && wc -l /tmp/exp/smoke_lpm.csv
```
Expected: a header line + ≥5 data rows; `ep_score_mean` column populated with non-NaN numbers (~200-400).

- [ ] **Step 5: Commit**

```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add LPM_exploration/Atari/main.py LPM_exploration/Atari/exploration/arguments.py
git commit -m "Atari: add --csv-log/--epsilon/--sticky-prob args and CSV metrics logging"
```

---

### Task 2: Sticky-action (`repeat_action_probability`) passthrough

**Files:**
- Modify: `LPM_exploration/Atari/exploration/envs.py` (`make_env` ~line 134, `make_vec_envs` ~line 199)
- Modify: `LPM_exploration/Atari/main.py` (the two `make_vec_envs(...)` calls ~line 100-104)

- [ ] **Step 1: Thread `sticky_prob` into `make_env`**

In `envs.py`, change the `make_env(...)` signature to add `sticky_prob=None`, and change the Atari `gym.make` line (`env = gym.make(env_id, max_episode_steps=10000000)`) to:
```python
        if sticky_prob is not None and not env_id.startswith("dm"):
            env = gym.make(env_id, max_episode_steps=10000000, repeat_action_probability=sticky_prob)
        else:
            env = gym.make(env_id, max_episode_steps=10000000)
```

- [ ] **Step 2: Thread `sticky_prob` into `make_vec_envs`**

In `envs.py`, add `sticky_prob=None` to the `make_vec_envs(...)` signature and pass it into the `make_env(...)` call inside the list comprehension: add `sticky_prob=sticky_prob` to that call.

- [ ] **Step 3: Pass `args.sticky_prob` from `main.py`**

In `main.py`, add `sticky_prob=args.sticky_prob` to BOTH `make_vec_envs(...)` calls (the `noisy=True` and `noisy=False` branches near line 100-104). Leave the obs-normalization `init_env` call (line ~322) unchanged (it uses defaults).

- [ ] **Step 4: Test that sticky actions take effect**

Run:
```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python -c "
from exploration.envs import make_env
for sp in (0.0, 0.5):
    e = make_env('MsPacmanNoFrameskip-v4', 0, 0, None, False, sticky_prob=sp)()
    rap = e.unwrapped.ale.getFloat('repeat_action_probability') if hasattr(e.unwrapped,'ale') else None
    print('sticky', sp, '-> ale repeat_action_probability =', rap)
    e.close()
"
```
Expected: prints `sticky 0.0 -> ... = 0.0` and `sticky 0.5 -> ... = 0.5` (confirms the kwarg reaches ALE).

- [ ] **Step 5: Commit**

```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add LPM_exploration/Atari/exploration/envs.py LPM_exploration/Atari/main.py
git commit -m "Atari: pass sticky-action repeat_action_probability through to ALE"
```

---

### Task 3: ε-greedy action layer over PPO

**Files:**
- Modify: `LPM_exploration/Atari/main.py` (rollout loop, right after `actor_critic.act(...)` ~line 402-405)

- [ ] **Step 1: Add a pure helper near the top of `main.py` (after imports)**

```python
def epsilon_greedy(action, epsilon, n_actions):
    """With prob epsilon (per env) replace the policy action with a uniform-random action."""
    if epsilon <= 0.0:
        return action
    mask = torch.rand(action.shape[0], 1, device=action.device) < epsilon
    rand = torch.randint(0, n_actions, action.shape, device=action.device, dtype=action.dtype)
    return torch.where(mask, rand, action)
```

- [ ] **Step 2: Apply it in the rollout loop**

In `main.py`, immediately after the `value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(...)` call inside `for step in range(args.num_steps):`, add:
```python
                if args.epsilon > 0.0:
                    action = epsilon_greedy(action, args.epsilon, envs.action_space.n)
```

- [ ] **Step 3: Unit-test the helper**

Run:
```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python -c "
import torch, main
a = torch.zeros(1000,1, dtype=torch.long)
print('eps=0 unchanged:', bool((main.epsilon_greedy(a,0.0,9)==0).all()))
out = main.epsilon_greedy(a, 1.0, 9)
print('eps=1 all-random in [0,9):', int(out.min()), int(out.max()))
frac = float((main.epsilon_greedy(a,0.1,9)!=0).float().mean())
print('eps=0.1 random fraction ~0.1:', round(frac,3))
"
```
Expected: `eps=0 unchanged: True`; `eps=1` min≥0 max≤8; `eps=0.1` fraction roughly 0.08-0.12.

- [ ] **Step 4: Smoke-test ε-greedy-PPO end to end**

Run:
```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python main.py \
  --env-name MsPacmanNoFrameskip-v4 --algo ppo --use-gae --lr 1e-4 --clip-param 0.1 \
  --num-processes 8 --num-steps 128 --num-mini-batch 8 --ppo-epoch 3 --entropy-coef 0.001 \
  --num-env-steps 25000 --log-interval 1 --save-interval 100000 --seed 1 --epsilon 0.1 \
  --csv-log /tmp/exp/smoke_eps.csv >/dev/null 2>&1
wc -l /tmp/exp/smoke_eps.csv
```
Expected: exit 0, CSV with ≥5 data rows.

- [ ] **Step 5: Commit**

```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add LPM_exploration/Atari/main.py
git commit -m "Atari: add epsilon-greedy action layer over the PPO policy"
```

---

## Phase 1 — De-risk: every method runs (smoke)

> Each curiosity baseline likely has latent bugs like LPM did (missing files, channel order, hardcoded `device='cuda'`, Sigmoid-vs-normalized-target, eta sign). These are **diagnose-then-fix** tasks: run the smoke; if it fails, read the traceback and apply the analogous fix (see the LPM fixes in `UPSTREAM.md` / commit 264c65a as the playbook). A task is done when the smoke reaches the training loop and writes a CSV with sane `ep_score_mean`.

### Task 4: Smoke `--algo ppo` (plain PPO / softmax baseline)

**Files:** none expected (verification); fixes to `main.py`/`exploration/*` only if the smoke fails.

- [ ] **Step 1: Run the smoke**

```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python main.py \
  --env-name MsPacmanNoFrameskip-v4 --algo ppo --use-gae --lr 1e-4 --clip-param 0.1 \
  --num-processes 8 --num-steps 128 --num-mini-batch 8 --ppo-epoch 3 --entropy-coef 0.001 \
  --num-env-steps 25000 --log-interval 1 --save-interval 100000 --seed 1 \
  --epsilon 0.0 --csv-log /tmp/exp/smoke_ppo.csv 2>&1 | tail -5
```
Expected: reaches training loop; `/tmp/exp/smoke_ppo.csv` has ≥5 rows with `ep_score_mean` ~200-400. (Plain PPO has no curiosity model, so `int_rew=0`.)

- [ ] **Step 2: If it failed, diagnose & fix, then re-run Step 1 until it passes.** Likely issues: none new (PPO path is the simplest). If `int_coeff`/curiosity code runs for `ppo`, confirm `train_model` is False for `--algo ppo`.

- [ ] **Step 3: Commit any fixes (skip if none)**

```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add -A LPM_exploration/Atari && git commit -m "Atari: get plain PPO running for the comparison" || echo "no changes"
```

### Task 5: Smoke `--algo rnd`

- [ ] **Step 1: Run the smoke** (same command as Task 4 but `--algo rnd --beta 1`, `--csv-log /tmp/exp/smoke_rnd.csv`).
- [ ] **Step 2: Diagnose & fix** using the LPM playbook. Check specifically: RND model `__init__` device arg (`device=device` not `'cuda'`); obs channel order (expects `(4,84,84)`); any `/255` / Sigmoid mismatch in `exploration/models/RND.py`; intrinsic-reward sign/scale. Re-run until CSV has sane rows.
- [ ] **Step 3: Append the fix to `UPSTREAM.md` and commit:**
```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add -A LPM_exploration && git commit -m "Atari: get RND running (note fixes in UPSTREAM.md)" || echo "no changes"
```

### Task 6: Smoke `--algo icm`

- [ ] **Step 1: Run the smoke** (`--algo icm --beta 1`, `--csv-log /tmp/exp/smoke_icm.csv`). Note `icm` uses `eta=200` at the call site — watch intrinsic-reward magnitude.
- [ ] **Step 2: Diagnose & fix** (`exploration/models/icm.py`): device arg, channel order, feature/inverse-model shapes, intrinsic scale. Re-run until sane.
- [ ] **Step 3: Append to `UPSTREAM.md` and commit** (same pattern as Task 5).

### Task 7: Smoke `--algo ama`

- [ ] **Step 1: Run the smoke** (`--algo ama --beta 1`, `--csv-log /tmp/exp/smoke_ama.csv`). Note `main.py` sets AMA's coeffs from `--use-dones`; verify `int_coeff` ends up > 0 for AMA (the AMA branch sets its own ext/int coeffs ~line 165-170 — confirm the no-`--use-dones` path gives `int_coeff=1`).
- [ ] **Step 2: Diagnose & fix** (`exploration/models/ama.py`): device, channel order, the double-headed net shapes, Sigmoid/normalization. Re-run until sane.
- [ ] **Step 3: Append to `UPSTREAM.md` and commit** (same pattern).

### Task 8: Record which methods run

- [ ] **Step 1: Create a status note**

Create `LPM_exploration/Atari/experiments/METHODS_STATUS.md` listing each of the 6 methods and `RUNS / BROKEN(reason)` based on Tasks 4-7 (LPM already runs). Any BROKEN method is dropped from the grid and reported as "attempted, not working" per the spec.

- [ ] **Step 2: Commit**
```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add LPM_exploration/Atari/experiments/METHODS_STATUS.md
git commit -m "experiments: record which exploration methods run"
```

---

## Phase 2 — Grid runner + analysis

### Task 9: `run_grid.py`

**Files:**
- Create: `LPM_exploration/Atari/experiments/run_grid.py`

- [ ] **Step 1: Write the runner**

```python
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
```

- [ ] **Step 2: Dry-run test (no training)**

```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python experiments/run_grid.py --dry-run
```
Expected: prints `60 runs:` then the run_ids (36 `rq1-*` + 24 `rq2-*`). Verify counts: `... --dry-run --which rq1` → 36; `--which rq2` → 24.

- [ ] **Step 3: One-run smoke through the runner**

```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python experiments/run_grid.py \
  --which rq1 --only lpm-clean-s1 --num-env-steps 25000
wc -l experiments/results/rq1-lpm-clean-s1.csv
```
Expected: exit 0; CSV with ≥5 rows.

- [ ] **Step 4: Commit**
```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add LPM_exploration/Atari/experiments/run_grid.py
git commit -m "experiments: grid runner for the Ms Pac-Man comparison"
```

### Task 10: `analyze.py`

**Files:**
- Create: `LPM_exploration/Atari/experiments/analyze.py`

- [ ] **Step 1: Write the analyzer**

```python
"""Aggregate experiments/results/*.csv into a table + 3 figures + intrinsic-reward analysis.

Run ID grammar: rq1-<method>-<clean|noisy>-s<seed>  |  rq2-<method>-sticky<p>-s<seed>
Final score per run = mean ep_score_mean over the last 10% of updates.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)

def final_score(df, frac=0.1):
    k = max(1, int(len(df) * frac)); return df["ep_score_mean"].iloc[-k:].mean()
def final_int(df, frac=0.1):
    k = max(1, int(len(df) * frac)); return df["int_rew"].iloc[-k:].mean()

def load():
    rows = []
    for f in glob.glob(os.path.join(RES, "*.csv")):
        rid = os.path.basename(f)[:-4]
        df = pd.read_csv(f)
        if len(df) < 3: continue
        m1 = re.match(r"rq1-(\w+)-(clean|noisy)-s(\d+)", rid)
        m2 = re.match(r"rq2-(\w+)-sticky([\d.]+)-s(\d+)", rid)
        if m1:
            rows.append(dict(rq="rq1", method=m1[1], cond=m1[2], sticky=0.0, seed=int(m1[3]),
                             score=final_score(df), intr=final_int(df), df=f))
        elif m2:
            rows.append(dict(rq="rq2", method=m2[1], cond="clean", sticky=float(m2[2]), seed=int(m2[3]),
                             score=final_score(df), intr=final_int(df), df=f))
    return pd.DataFrame(rows)

def main():
    d = load()
    if d.empty: print("no results yet"); return
    # ---- RQ1 table: method x cond, mean+/-std over seeds, plus clean->noisy drop ----
    g = d[d.rq == "rq1"].groupby(["method", "cond"])["score"].agg(["mean", "std"]).reset_index()
    piv = g.pivot(index="method", columns="cond", values="mean")
    piv["drop_%"] = (piv.get("clean", np.nan) - piv.get("noisy", np.nan)) / piv.get("clean", np.nan) * 100
    piv.to_csv(os.path.join(OUT, "table_rq1.csv"))
    print("=== RQ1 final scores (mean over seeds) + clean->noisy drop ===")
    print(piv.round(1).to_string())

    # ---- Fig 1: learning curves (clean & noisy), score vs frames, mean over seeds ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ci, cond in enumerate(["clean", "noisy"]):
        for method in sorted(d.method.unique()):
            files = d[(d.rq == "rq1") & (d.method == method) & (d.cond == cond)].df.tolist()
            if not files: continue
            curves = [pd.read_csv(f)[["frames", "ep_score_mean"]] for f in files]
            base = curves[0]["frames"].values
            stacked = np.vstack([np.interp(base, c["frames"], c["ep_score_mean"]) for c in curves])
            ax[ci].plot(base/1e6, stacked.mean(0), label=method)
        ax[ci].set_title(f"{cond}"); ax[ci].set_xlabel("Frames (M)"); ax[ci].grid(alpha=.3)
    ax[0].set_ylabel("Episode score"); ax[0].legend(fontsize=8)
    fig.suptitle("Ms Pac-Man: learning curves by method"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_learning_curves.png"), dpi=130)

    # ---- Fig 2: noise-robustness drop bar chart ----
    fig, ax = plt.subplots(figsize=(7, 4))
    pp = piv.dropna(subset=["drop_%"]).sort_values("drop_%")
    ax.bar(pp.index, pp["drop_%"]); ax.axhline(0, color="k", lw=.5)
    ax.set_ylabel("clean→noisy drop (%)"); ax.set_title("Noise robustness (lower = more robust)")
    ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_noise_drop.png"), dpi=130)

    # ---- Fig 3: RQ2 stochasticity sweep — LPM advantage vs sticky prob ----
    fig, ax = plt.subplots(figsize=(7, 4))
    sticky_levels = [0.0, 0.25, 0.5]
    def score_at(method, sp):
        if sp == 0.0:
            sub = d[(d.rq == "rq1") & (d.method == method) & (d.cond == "clean")]
        else:
            sub = d[(d.rq == "rq2") & (d.method == method) & (np.isclose(d.sticky, sp))]
        return sub.score.mean() if len(sub) else np.nan
    for base in ["rnd", "icm"]:
        gaps = [score_at("lpm", sp) - score_at(base, sp) for sp in sticky_levels]
        ax.plot(sticky_levels, gaps, marker="o", label=f"LPM − {base.upper()}")
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel("sticky-action prob (stochasticity)")
    ax.set_ylabel("LPM score advantage"); ax.set_title("RQ2: does LPM's edge grow with stochasticity?")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_stochasticity_sweep.png"), dpi=130)
    print("saved figures + table_rq1.csv to", OUT)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the analyzer on synthetic CSVs**

```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python -c "
import os, numpy as np, pandas as pd
R='experiments/results'; os.makedirs(R, exist_ok=True)
def mk(rid, base): 
    f=np.linspace(0,1e6,50); s=base+np.linspace(0,100,50)+np.random.randn(50)
    pd.DataFrame(dict(update=range(50),frames=f,fps=700,ep_score_mean=s,ep_score_std=10,
      ep_len_mean=460,int_rew=np.linspace(0.3,0,50),ext_rew=s/470,pred_loss=1,unc_loss=.05,
      dist_entropy=2,value_loss=50,action_loss=0)).to_csv(f'{R}/{rid}.csv',index=False)
for m,b in [('lpm',420),('rnd',380),('icm',360),('ama',400),('ppo',300),('egreedy',310)]:
    for s in (1,2,3):
        mk(f'rq1-{m}-clean-s{s}', b); mk(f'rq1-{m}-noisy-s{s}', b*(0.6 if m in('rnd','icm') else 0.95))
for m in ('lpm','rnd','icm','ppo'):
    for sp in (0.25,0.5):
        for s in (1,2,3): mk(f'rq2-{m}-sticky{sp}-s{s}', 400-(0 if m=='lpm' else 60*sp*4))
print('synthetic CSVs written')
"
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python experiments/analyze.py
ls experiments/figures/
```
Expected: prints the RQ1 table (LPM smallest drop), saves `fig1/fig2/fig3.png` + `table_rq1.csv`. Then **delete the synthetic CSVs** before real runs: `rm experiments/results/*.csv`.

- [ ] **Step 3: Commit**
```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add LPM_exploration/Atari/experiments/analyze.py
git commit -m "experiments: analysis -> RQ1 table + 3 figures"
```

- [ ] **Step 4: Gitignore run artifacts**

Append to `.gitignore`:
```
LPM_exploration/Atari/experiments/results/
LPM_exploration/Atari/experiments/figures/
```
Commit: `git add .gitignore && git commit -m "gitignore experiment results/figures"`

---

## Phase 3 — Execute the grid (long-running; ~20 h compute total)

> These produce the actual data. Each is a long batch; run them as background jobs. The runner SKIPs already-completed CSVs, so batches are resumable.

### Task 11: Run RQ1 (clean/noisy × 6 methods × 3 seeds = 36 runs)

- [ ] **Step 1: Launch** (drop any BROKEN method from METHODS in `run_grid.py` first, per Task 8):
```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python experiments/run_grid.py --which rq1
```
Expected: ~12 h; 36 CSVs in `experiments/results/` each with ~488 rows.

- [ ] **Step 2: Sanity-check** `ls experiments/results/rq1-*.csv | wc -l` → 36 (or 6×2×3 minus dropped methods).

### Task 12: Run RQ2 sticky sweep (LPM/RND/ICM/PPO × {0.25,0.5} × 3 = 24 runs)

- [ ] **Step 1: Launch**
```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python experiments/run_grid.py --which rq2
```
Expected: ~8 h; 24 CSVs `rq2-*.csv`.

### Task 13: Analyze & write findings

- [ ] **Step 1: Generate table + figures**
```bash
cd /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/Atari
PYTHONPATH=. /Users/yingtexu/Codebase/ChallengingRL/LPM_exploration/.venv/bin/python experiments/analyze.py
```
Expected: `experiments/figures/{fig1,fig2,fig3}.png` + `table_rq1.csv`.

- [ ] **Step 2: Write `experiments/FINDINGS.md`** summarizing: the RQ1 table (who is noise-robust, LPM/AMA drop vs RND/ICM drop), the RQ2 sweep (does LPM−RND / LPM−ICM gap grow with sticky prob?), and the intrinsic-reward story (LPM decays; RND/ICM stay high under noise). State the caveats from spec §7 (sticky≠ghost, small sample).

- [ ] **Step 3: Commit findings (figures are gitignored; the writeup is tracked)**
```bash
cd /Users/yingtexu/Codebase/ChallengingRL
git add LPM_exploration/Atari/experiments/FINDINGS.md
git commit -m "experiments: findings for the Ms Pac-Man exploration comparison"
```

---

## Self-Review

**Spec coverage:** RQ1 → Tasks 11+13; RQ2 sticky sweep → Tasks 2,12,13; 6 methods → Tasks 4-7 (+ LPM done); classical (softmax=ppo, ε-greedy) → Tasks 3,4; CSV logging → Task 1; metrics (drop, sweep, intrinsic) → Task 10; deliverables (table+3 figs+findings) → Tasks 10,13; risks (drop broken methods) → Task 8. UCB/Thompson are discussion-only (no task — correct). All spec sections covered.

**Placeholders:** none — every code/test/command step is concrete. Baseline-debug tasks (5-7) are intentionally diagnose-then-fix (the bug is unknown until run) with explicit smoke commands and the known failure-class checklist; this is the correct shape for ML-infra debugging, not a placeholder.

**Type/name consistency:** `--csv-log`/`--epsilon`/`--sticky-prob` defined in Task 1, used consistently in Tasks 2,3,9. `epsilon_greedy(action, epsilon, n_actions)` defined in Task 3, used in Task 3 Step 2. `run_grid.py` METHODS keys (lpm/rnd/icm/ama/ppo/egreedy) match `analyze.py` regex method group and Fig-3 baselines. CSV columns written in Task 1 match those read in Task 10 (`ep_score_mean`, `int_rew`, `frames`). Consistent.
