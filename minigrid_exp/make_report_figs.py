"""Generate clean slide/report figures + trace filmstrips from the results table.

Writes to expr_data/minigrid/figures/report/. Run: PYTHONPATH=. python make_report_figs.py
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import config
from method_utils import is_intrinsic

FIGDIR = os.path.join(str(config.EXPR_DATA), "figures")
OUT = os.path.join(FIGDIR, "report")
os.makedirs(OUT, exist_ok=True)
TAB = pd.read_csv(os.path.join(FIGDIR, "table_final_success.csv"))
LOGS_EVAL = os.path.join(str(config.EXPR_DATA), "results", "logs", "ppo", "eval")


def per_seed_finals(env, variant, method, npv=np.nan, frac=0.1):
    """Per-seed final eval return, using the SAME definition as analyze.py's bar
    means: the mean over the last `frac` (10%) of the seed's eval curve
    (chunks concatenated, sorted by global timestep). Using the same definition
    for the dots and the bars keeps them consistent (otherwise a single-last-point
    dot can sit above a windowed-mean bar even with no collapse)."""
    base = f"{env}__{variant}__{method}__seed_".replace("/", "_")
    vals = []
    for d in sorted(glob.glob(os.path.join(LOGS_EVAL, base + "*"))):
        run = os.path.basename(d)
        if pd.isna(npv):
            if "__np" in run:            # clean: reject noise runs
                continue
        elif not run.endswith(f"__np{npv:g}"):
            continue
        pts = []
        for p in (sorted(glob.glob(os.path.join(d, "*", "evaluations.npz")))
                  + glob.glob(os.path.join(d, "evaluations.npz"))):
            z = np.load(p)
            pts.extend((int(t), float(np.mean(r))) for t, r in zip(z["timesteps"], z["results"]))
        if not pts:
            continue
        pts.sort(key=lambda tr: tr[0])
        returns = [r for _, r in pts]
        k = max(1, int(frac * len(returns)))
        vals.append(float(np.mean(returns[-k:])))
    return vals

METHODS = ["none", "entropy", "rnd", "lpm"]
COLORS = {"none": "#888888", "entropy": "#1f77b4", "rnd": "#2ca02c", "lpm": "#d62728",
          "rnd_lstm": "#98df8a", "lpm_lstm": "#ff9896"}
# Display labels: uppercase acronyms so "rnd" isn't misread as "md" at small sizes.
LBL = {"none": "none", "entropy": "entropy", "rnd": "RND", "lpm": "LPM",
       "rnd_lstm": "RND+LSTM", "lpm_lstm": "LPM+LSTM"}

# Theoretical-max eval return = 1 - 0.9*E[optimal_steps]/max_steps, from an exact
# BFS solver over 300 random layouts (see optimal_reward.py). This is the ceiling a
# perfect (optimal-play) agent could reach on each env's eval-layout distribution;
# ~1.0 is unreachable because solving costs a fraction of the step budget.
THEORETICAL_MAX = {
    "MiniGrid-DoorKey-5x5-v0": 0.965,
    "MiniGrid-FourRooms-v0": 0.856,
    "MiniGrid-MultiRoom-N6-v0": 0.652,
}


def cell(env, variant, method, beta=np.nan, npv=np.nan):
    d = TAB[(TAB.env == env) & (TAB.variant == variant) & (TAB.method == method)]
    d = d[(d.beta.isna() if pd.isna(beta) else d.beta == beta)]
    d = d[(d["np"].isna() if pd.isna(npv) else d["np"] == npv)]
    return (float(d["mean"].iloc[0]), float(d["std"].iloc[0])) if len(d) else (np.nan, np.nan)


# --- Fig 1: difficulty ladder (clean) ---
def fig_ladder():
    envs = [("MiniGrid-DoorKey-5x5-v0", "easy\nDoorKey-5x5"),
            ("MiniGrid-FourRooms-v0", "medium\nFourRooms"),
            ("MiniGrid-MultiRoom-N6-v0", "hard\nMultiRoom-N6")]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x = np.arange(len(envs)); w = 0.2
    for i, m in enumerate(METHODS):
        var = "intrinsic_no_noise" if is_intrinsic(m) else "baseline_no_noise"
        means, stds = [], []
        for env, _ in envs:
            mu, sd = cell(env, var, m)
            means.append(mu); stds.append(sd)
        xpos = x + (i - 1.5) * w
        ax.bar(xpos, means, w, yerr=stds, capsize=2, label=LBL[m],
               color=COLORS[m], alpha=0.85, zorder=1)
        # Overlay per-seed final returns (same windowed definition as the bars).
        for j, (env, _) in enumerate(envs):
            vals = per_seed_finals(env, var, m)
            if not vals:
                continue
            jit = np.linspace(-1, 1, len(vals)) * w * 0.3 if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), xpos[j]) + jit, vals, s=13, color="black",
                       zorder=3, edgecolors="white", linewidths=0.4)
    # Per-env theoretical-max (optimal play) reference line over each env group.
    for j, (env, _) in enumerate(envs):
        ax.hlines(THEORETICAL_MAX[env], x[j] - 0.42, x[j] + 0.42, colors="black",
                  linestyles="--", lw=1.3, zorder=4,
                  label=("theoretical max (optimal play)" if j == 0 else None))
        ax.text(x[j] + 0.43, THEORETICAL_MAX[env], f"{THEORETICAL_MAX[env]:.2f}",
                va="center", ha="left", fontsize=7, color="black")
    ax.set_xticks(x); ax.set_xticklabels([e[1] for e in envs])
    ax.set_ylabel("final eval return")
    ax.set_title("Difficulty ladder (clean): intrinsic motivation is difficulty-gated\n"
                 "(bars = mean$\\pm$std; dots = per-seed finals; DoorKey 8 seeds, others 3)",
                 fontsize=10)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.18)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_difficulty_ladder.png"), dpi=140)
    plt.close(fig)


# --- Fig 2: beta sweep on FourRooms (from FINDINGS sec.1, 500k) ---
def fig_beta():
    betas = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05]
    rnd = [0.25, 0.20, 0.30, 0.30, 0.25, 0.00]
    lpm = [0.25, 0.20, 0.16, 0.12, 0.09, 0.01]
    base = 0.30
    x = [b if b > 0 else 1e-4 for b in betas]   # plot 0 at a small x on log axis
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, rnd, "o-", color=COLORS["rnd"], label="RND")
    ax.plot(x, lpm, "s-", color=COLORS["lpm"], label="LPM")
    ax.axhline(base, ls="--", color=COLORS["none"], label="baseline (none)")
    tmax = THEORETICAL_MAX["MiniGrid-FourRooms-v0"]
    ax.axhline(tmax, ls=":", color="black", lw=1.4, label=f"theoretical max ({tmax:.2f})")
    ax.set_xscale("log"); ax.set_xlabel(r"intrinsic coefficient $\beta$ (0 shown at left)")
    ax.set_ylabel("final eval return")
    ax.set_title(r"$\beta$ sweep, FourRooms (500k): too large $\Rightarrow$ drowns the sparse signal")
    ax.set_xticks(x); ax.set_xticklabels(["0", "5e-4", "1e-3", "5e-3", "1e-2", "5e-2"])
    ax.legend(fontsize=9); ax.set_ylim(0, 0.95)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_beta_sweep.png"), dpi=140)
    plt.close(fig)


# --- Fig 3: DoorKey clean vs noisy (the clean RQ4 demo) ---
def fig_doorkey_noise():
    env = "MiniGrid-DoorKey-5x5-v0"
    cvar = lambda m: "intrinsic_no_noise" if is_intrinsic(m) else "baseline_no_noise"
    nvar = lambda m: "intrinsic_noise" if is_intrinsic(m) else "baseline_noise"
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(METHODS)); w = 0.38
    clean = [cell(env, cvar(m), m) for m in METHODS]
    noisy = [cell(env, nvar(m), m, npv=0.1) for m in METHODS]
    ax.bar(x - w / 2, [c[0] for c in clean], w, yerr=[c[1] for c in clean], capsize=3,
           label="clean (mean$\\pm$std)", color="#4c78a8", alpha=0.85, zorder=1)
    ax.bar(x + w / 2, [c[0] for c in noisy], w, yerr=[c[1] for c in noisy], capsize=3,
           label="noisy 10% (mean$\\pm$std)", color="#e45756", alpha=0.85, zorder=1)
    # Overlay per-seed final returns so bimodality (the LPM-clean 0/0.96 split) is visible.
    for i, m in enumerate(METHODS):
        for xc, vals in [(x[i] - w / 2, per_seed_finals(env, cvar(m), m)),
                         (x[i] + w / 2, per_seed_finals(env, nvar(m), m, npv=0.1))]:
            if not vals:
                continue
            jit = np.linspace(-1, 1, len(vals)) * w * 0.28 if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), xc) + jit, vals, s=20, color="black",
                       zorder=3, edgecolors="white", linewidths=0.5)
    # Annotate the LPM-clean solve rate (the point of the whole figure).
    li = METHODS.index("lpm")
    cv = per_seed_finals(env, cvar("lpm"), "lpm")
    n_solve = sum(v > 0.5 for v in cv)
    ax.annotate(f"{n_solve}/{len(cv)} seeds solve;\n{len(cv) - n_solve}/{len(cv)} collapse to 0",
                xy=(x[li] - w / 2, 0.55), fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9), zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in METHODS])
    ax.set_ylabel("final eval return"); ax.set_ylim(0, 1.18)
    ax.set_title("DoorKey-5x5: under noise, RND collapses while LPM stays robust\n"
                 "(dots = per-seed finals; LPM-clean is bimodal, not just high-variance)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_doorkey_noise.png"), dpi=140)
    plt.close(fig)


# --- Fig 4: FourRooms fine noise sweep (0..0.1, 11 points) ---
def fig_fourrooms_noise():
    """Final eval return vs observation-noise probability on FourRooms, swept
    0..0.1 in steps of 0.01 (3 seeds, bands = +/-std). The honest read: only RND
    separates (it degrades much faster and pulls below the pack from ~0.04 on),
    while none/entropy/LPM stay clustered and statistically tied across the whole
    range -- LPM tracks the baseline, it does not beat it."""
    env = "MiniGrid-FourRooms-v0"
    nps = [round(0.01 * i, 2) for i in range(11)]  # 0.00 .. 0.10
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for m in METHODS:
        base = "intrinsic" if is_intrinsic(m) else "baseline"
        means, stds = [], []
        for p in nps:
            if p == 0.0:
                mu, sd = cell(env, f"{base}_no_noise", m)
            else:
                mu, sd = cell(env, f"{base}_noise", m, npv=p)
            means.append(mu); stds.append(sd)
        means, stds = np.array(means), np.array(stds)
        ax.plot(nps, means, "o-", ms=4, color=COLORS[m], label=LBL[m])
        ax.fill_between(nps, means - stds, means + stds, alpha=0.15, color=COLORS[m])
    ax.set_xlabel("observation-noise probability"); ax.set_ylabel("final eval return")
    ax.set_ylim(0, 0.40)
    ax.set_title("FourRooms noise sweep (3 seeds): only RND separates;\n"
                 "none/entropy/LPM stay clustered (LPM tracks the baseline, not above it)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_fourrooms_noise.png"), dpi=140)
    plt.close(fig)


# --- Fig 6 left: FourRooms clean vs noisy@0.1 bars (DoorKey-style snapshot) ---
def fig_fourrooms_noise_bars():
    """The DoorKey-style clean-vs-noisy bar comparison, recomputed on FourRooms:
    4 methods, clean vs noisy@0.1, mean+/-std with per-seed dots. Honest snapshot
    (no retained-% annotation): every method loses ground, none/entropy/LPM land
    together while RND lands lowest. Companion to the full sweep."""
    env = "MiniGrid-FourRooms-v0"
    cvar = lambda m: "intrinsic_no_noise" if is_intrinsic(m) else "baseline_no_noise"
    nvar = lambda m: "intrinsic_noise" if is_intrinsic(m) else "baseline_noise"
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    x = np.arange(len(METHODS)); w = 0.38
    clean = [cell(env, cvar(m), m) for m in METHODS]
    noisy = [cell(env, nvar(m), m, npv=0.1) for m in METHODS]
    ax.bar(x - w / 2, [c[0] for c in clean], w, yerr=[c[1] for c in clean], capsize=3,
           label="clean (mean$\\pm$std)", color="#4c78a8", alpha=0.85, zorder=1)
    ax.bar(x + w / 2, [c[0] for c in noisy], w, yerr=[c[1] for c in noisy], capsize=3,
           label="noisy 10% (mean$\\pm$std)", color="#e45756", alpha=0.85, zorder=1)
    for i, m in enumerate(METHODS):
        for xc, vals in [(x[i] - w / 2, per_seed_finals(env, cvar(m), m)),
                         (x[i] + w / 2, per_seed_finals(env, nvar(m), m, npv=0.1))]:
            if not vals:
                continue
            jit = np.linspace(-1, 1, len(vals)) * w * 0.28 if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), xc) + jit, vals, s=20, color="black",
                       zorder=3, edgecolors="white", linewidths=0.5)
    tmax = THEORETICAL_MAX["MiniGrid-FourRooms-v0"]
    ax.axhline(tmax, ls=":", color="black", lw=1.4, label=f"theoretical max ({tmax:.2f})")
    ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in METHODS])
    ax.set_ylabel("final eval return"); ax.set_ylim(0, 0.95)
    ax.set_title("FourRooms: clean vs noisy 10%\n(dots = per-seed finals, 3 seeds)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_fourrooms_bars.png"), dpi=140)
    plt.close(fig)


# --- Fig 5: memory ablation — MLP vs LSTM policy for RND and LPM ---
def fig_memory_ablation():
    """Per env, MLP vs LSTM policy for RND and LPM, clean and noisy@0.1."""
    envs = [("MiniGrid-DoorKey-5x5-v0", "DoorKey-5x5"),
            ("MiniGrid-FourRooms-v0", "FourRooms"),
            ("MiniGrid-MultiRoom-N6-v0", "MultiRoom-N6")]
    bars = ["rnd", "rnd_lstm", "lpm", "lpm_lstm"]  # intrinsic-only; none/entropy excluded by design (MLP-vs-LSTM ablation)
    panels = [("clean", "intrinsic_no_noise", np.nan),
              ("noisy 10%", "intrinsic_noise", 0.1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, (cond, var, npv) in zip(axes, panels):
        x = np.arange(len(envs)); w = 0.2
        for i, m in enumerate(bars):
            means = [cell(env, var, m, npv=npv)[0] for env, _ in envs]
            stds = [cell(env, var, m, npv=npv)[1] for env, _ in envs]
            ax.bar(x + (i - 1.5) * w, means, w, yerr=stds, capsize=2,
                   label=LBL[m], color=COLORS[m], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([e[1] for e in envs], fontsize=8)
        ax.set_title(cond); ax.set_ylim(0, 1.18)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("final eval return")
    fig.suptitle("Memory ablation: MLP vs LSTM policy (RND, LPM)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_memory_ablation.png"), dpi=140)
    plt.close(fig)


# --- trace filmstrips (evenly-spaced frames from a GIF, side by side) ---
def filmstrip(gif_name, out_name, n=4):
    path = os.path.join(FIGDIR, "traces", gif_name)
    if not os.path.exists(path):
        print("missing", path); return
    frames = imageio.mimread(path)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    strip = np.hstack([frames[i] for i in idx])
    imageio.imwrite(os.path.join(OUT, out_name), strip)
    print("wrote", out_name, f"({len(frames)} frames -> {n})")


if __name__ == "__main__":
    fig_ladder(); fig_beta(); fig_doorkey_noise(); fig_fourrooms_noise()
    fig_fourrooms_noise_bars()
    fig_memory_ablation()
    filmstrip("MiniGrid-MultiRoom-N6-v0__intrinsic_no_noise__rnd__seed_1.gif",
              "strip_multiroom_rnd_solves.png")
    filmstrip("MiniGrid-MultiRoom-N6-v0__baseline_no_noise__none__seed_1.gif",
              "strip_multiroom_none_fails.png")
    filmstrip("MiniGrid-DoorKey-5x5-v0__intrinsic_noise__lpm__seed_1__np0.1.gif",
              "strip_doorkey_lpm_noisy_solves.png")
    filmstrip("MiniGrid-DoorKey-5x5-v0__intrinsic_noise__rnd__seed_1__np0.1.gif",
              "strip_doorkey_rnd_noisy_fails.png")
    print("figures ->", OUT)
