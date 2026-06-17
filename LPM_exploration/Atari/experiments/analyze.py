"""Aggregate experiments/results/*.csv into a table + 3 figures + intrinsic-reward analysis.

Run ID grammar: rq1-<method>-<clean|noisy>-s<seed>  |  rq2-<method>-sticky<p>-s<seed>
Final score per run = mean ep_score_mean over the last 10% of updates.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Artifacts live under <repo>/expr_data/atari/ (gitignored). HERE -> Atari -> LPM_exploration -> repo root.
EXPR_DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "expr_data", "atari"))
RES = os.path.join(EXPR_DATA, "results"); OUT = os.path.join(EXPR_DATA, "figures"); os.makedirs(OUT, exist_ok=True)

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
