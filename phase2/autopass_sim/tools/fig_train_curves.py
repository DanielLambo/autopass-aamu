"""Training curves (episode return, cost, lambda) for v2-A and v2-B, 3 seeds each (simulated).

Per iteration: mean over the episodes that finished in that iteration's batch (train.csv).
Lines: mean over the 3 training seeds after a 10-iteration moving average; band: min-max over seeds.

    python tools/fig_train_curves.py --out results_rl/v2/addendum/fig_train_curves.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = [("v2-A: mask shield + budget-aware", "results_rl/v2/A_mask_budget_s{}/train.csv", "#DD8452"),
          ("v2-B: no shield + budget-aware (Lagrangian only)", "results_rl/v2/B_noshield_budget_s{}/train.csv", "#55A868")]
PANELS = [("ret", "episode return"), ("cost", "cost = unsafe commits / episode"), ("lambda", "Lagrange multiplier λ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_rl/v2/addendum/fig_train_curves.png")
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--cost-limit", type=float, default=0.01)
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for label, pat, col in GROUPS:
        runs = [pd.read_csv(pat.format(s)) for s in range(3)]
        n = min(len(r) for r in runs)
        x = runs[0].steps.values[:n] / 1e3
        for ax, (k, title) in zip(axes, PANELS):
            Y = np.stack([r[k].rolling(a.smooth, min_periods=1).mean().values[:n] for r in runs])
            ax.plot(x, Y.mean(0), color=col, lw=1.8, label=label)
            ax.fill_between(x, Y.min(0), Y.max(0), color=col, alpha=0.2, lw=0)
    axes[1].axhline(a.cost_limit, color="0.3", ls="--", lw=1)
    axes[1].text(axes[1].get_xlim()[1], a.cost_limit, f" limit d = {a.cost_limit}", va="bottom", ha="right",
                 fontsize=8, color="0.3")
    for ax, (k, title) in zip(axes, PANELS):
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("environment steps (thousands)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Report 3 v2 training curves (simulated; 500k steps, 18 envs, budget mix {none .4, 30/20/12 .2 each})",
                 fontsize=11.5)
    fig.text(0.5, 0.01, f"Simulated. Line: mean of 3 training seeds ({a.smooth}-iteration moving average); "
             "band: min-max over seeds. Return does not see safety (an unsafe commit earns the success reward),\n"
             "so v2-B's higher return includes unsafe commits. With the mask shield no unsafe commit is possible, so v2-A's cost and λ stay 0.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
