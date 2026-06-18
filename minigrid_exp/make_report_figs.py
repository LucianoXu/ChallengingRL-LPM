"""Generate clean slide/report figures + trace filmstrips from the results table.

Writes to expr_data/minigrid/figures/report/. Run: PYTHONPATH=. python make_report_figs.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import config

FIGDIR = os.path.join(str(config.EXPR_DATA), "figures")
OUT = os.path.join(FIGDIR, "report")
os.makedirs(OUT, exist_ok=True)
TAB = pd.read_csv(os.path.join(FIGDIR, "table_final_success.csv"))

METHODS = ["none", "entropy", "rnd", "lpm"]
COLORS = {"none": "#888888", "entropy": "#1f77b4", "rnd": "#2ca02c", "lpm": "#d62728"}
# Display labels: uppercase acronyms so "rnd" isn't misread as "md" at small sizes.
LBL = {"none": "none", "entropy": "entropy", "rnd": "RND", "lpm": "LPM"}


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
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(envs)); w = 0.2
    for i, m in enumerate(METHODS):
        means, stds = [], []
        for env, _ in envs:
            var = "intrinsic_no_noise" if m in ("rnd", "lpm") else "baseline_no_noise"
            mu, sd = cell(env, var, m)
            means.append(mu); stds.append(sd)
        ax.bar(x + (i - 1.5) * w, means, w, yerr=stds, capsize=2,
               label=LBL[m], color=COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels([e[1] for e in envs])
    ax.set_ylabel("final eval return (3 seeds)")
    ax.set_title("Difficulty ladder (clean): intrinsic motivation is difficulty-gated")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.05)
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
    ax.set_xscale("log"); ax.set_xlabel(r"intrinsic coefficient $\beta$ (0 shown at left)")
    ax.set_ylabel("final eval return")
    ax.set_title(r"$\beta$ sweep, FourRooms (500k): too large $\Rightarrow$ drowns the sparse signal")
    ax.set_xticks(x); ax.set_xticklabels(["0", "5e-4", "1e-3", "5e-3", "1e-2", "5e-2"])
    ax.legend(fontsize=9); ax.set_ylim(0, 0.4)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_beta_sweep.png"), dpi=140)
    plt.close(fig)


# --- Fig 3: DoorKey clean vs noisy (the clean RQ4 demo) ---
def fig_doorkey_noise():
    env = "MiniGrid-DoorKey-5x5-v0"
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(METHODS)); w = 0.38
    clean = [cell(env, "intrinsic_no_noise" if m in ("rnd", "lpm") else "baseline_no_noise", m)
             for m in METHODS]
    noisy = [cell(env, "intrinsic_noise" if m in ("rnd", "lpm") else "baseline_noise", m, npv=0.1)
             for m in METHODS]
    ax.bar(x - w / 2, [c[0] for c in clean], w, yerr=[c[1] for c in clean], capsize=3,
           label="clean", color="#4c78a8")
    ax.bar(x + w / 2, [c[0] for c in noisy], w, yerr=[c[1] for c in noisy], capsize=3,
           label="noisy (10%)", color="#e45756")
    ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in METHODS])
    ax.set_ylabel("final eval return"); ax.set_ylim(0, 1.05)
    ax.set_title("DoorKey-5x5: under noise, RND collapses while LPM stays robust")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_doorkey_noise.png"), dpi=140)
    plt.close(fig)


# --- Fig 4: FourRooms noise degradation curves ---
def fig_fourrooms_noise():
    env = "MiniGrid-FourRooms-v0"; nps = [0.0, 0.1, 0.2, 0.3]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ["none", "rnd", "lpm"]:
        var = "intrinsic_noise" if m in ("rnd", "lpm") else "baseline_noise"
        ys = [cell(env, var, m, npv=p)[0] for p in nps]
        ax.plot(nps, ys, "o-", color=COLORS[m], label=LBL[m])
    ax.set_xlabel("observation-noise probability"); ax.set_ylabel("final eval return")
    ax.set_title("FourRooms: RND degrades most under noise; LPM least")
    ax.legend(fontsize=9); ax.set_ylim(0, 0.4)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_fourrooms_noise.png"), dpi=140)
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
    filmstrip("MiniGrid-MultiRoom-N6-v0__intrinsic_no_noise__rnd__seed_1.gif",
              "strip_multiroom_rnd_solves.png")
    filmstrip("MiniGrid-MultiRoom-N6-v0__baseline_no_noise__none__seed_1.gif",
              "strip_multiroom_none_fails.png")
    filmstrip("MiniGrid-DoorKey-5x5-v0__intrinsic_noise__lpm__seed_1__np0.1.gif",
              "strip_doorkey_lpm_noisy_solves.png")
    filmstrip("MiniGrid-DoorKey-5x5-v0__intrinsic_noise__rnd__seed_1__np0.1.gif",
              "strip_doorkey_rnd_noisy_fails.png")
    print("figures ->", OUT)
