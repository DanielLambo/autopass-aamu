"""Figure for Report 3 eval: time_s and scans per setting, rule vs PPO-Lag + shield (simulated).

Run (from phase2/autopass_sim, after autopass_rl.evaluate):
    python tools/fig_eval.py
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results_rl/eval")
ap.add_argument("--ppo", default=r"ppolag_shield_s\d\|eval_shield", help="regex for the PPO-Lag + shield policies")
ap.add_argument("--ppo-label", default="PPO-Lag + shield\n(trained & evaluated with shield, 3 seeds)")
ap.add_argument("--title", default="Report 3 deterministic evaluation, simulated (600 fixed seeds per policy and setting)")
args = ap.parse_args()
D = args.dir
SETTINGS = ["main", "uniform", "budget20", "budget12"]
LABELS = {"main": "main", "uniform": "uniform w_A\n(66-92 cm)", "budget20": "scan budget\n20", "budget12": "scan budget\n12"}
POLS = {"rule": ("Rule (z-test)", "#4C72B0"),
        "ppo_shield": (args.ppo_label, "#DD8452")}


def ci(x, rng, n=2000):
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


rng = np.random.default_rng(0)
stats = {}
for s in SETTINGS:
    ep = pd.read_csv(os.path.join(D, f"episodes_{s}.csv"))
    sel = {"rule": ep[ep.policy == "rule"],
           "ppo_shield": ep[ep.policy.str.match(args.ppo)]}
    for p, g in sel.items():
        for m in ("time_s", "scans"):
            stats[s, p, m] = ci(g[m].values.astype(float), rng)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(len(SETTINGS)); w = 0.38
for ax, m, yl in zip(axes, ("time_s", "scans"), ("mission time (s)", "sweeps per episode")):
    for j, (p, (lab, col)) in enumerate(POLS.items()):
        v = np.array([stats[s, p, m] for s in SETTINGS])
        ax.bar(x + (j - 0.5) * w, v[:, 0], w, color=col, label=lab,
               yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=3, error_kw=dict(lw=1))
    ax.set_xticks(x, [LABELS[s] for s in SETTINGS], fontsize=9)
    ax.set_ylabel(yl); ax.set_title(f"{m} (mean, 95% bootstrap CI)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, [t.replace("\n", " ") for t in l], loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2,
           fontsize=8.5, frameon=False)
fig.suptitle(args.title, fontsize=11)
fig.text(0.5, 0.005, "Simulated. Rule: 600 episodes. PPO-Lag + shield: 3 training seeds x 600 episodes pooled. "
         "Same seeds 5000-5599 for every bar.", ha="center", fontsize=8, color="0.3")
fig.tight_layout(rect=(0, 0.04, 1, 0.88))
fig.savefig(os.path.join(D, "fig_main.png"), dpi=150)
print("wrote", os.path.join(D, "fig_main.png"))
