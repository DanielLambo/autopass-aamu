"""Figure: share of missions that committed to a too-narrow door, per method, with 95% Wilson CIs."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analyze import load, wilson

ORDER = [("B1", "B1  shortest path, naive rule"), ("P-A*(4)n", "P-A*(4)n  obs.-aware path, naive rule"),
         ("B2", "B2  shortest path, trusted rule"), ("P-A*(4)", "P-A*(4)  obs.-aware path, trusted rule"),
         ("P-RRT*(4)", "P-RRT*(4)  obs.-aware RRT*, trusted rule")]
COL = {"B1": "#C9A227", "P-A*(4)n": "#D9822B", "B2": "#5A5A5A", "P-A*(4)": "#7A1E2C", "P-RRT*(4)": "#2F6F73"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_tb")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(f"{a.results}/missions_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0), dpi=200, sharey=True)
    for ax, lay in zip(axes, ["corridor", "room"]):
        for i, (m, lab) in enumerate(ORDER):
            R = [r for r in rows if r["layout"] == lay and r["method"] == m]
            k = sum(r["outcome"] == "unsafe" for r in R)
            p, lo, hi = wilson(k, len(R))
            y = len(ORDER) - 1 - i
            ax.barh(y, 100 * p, color=COL[m], height=0.6)
            ax.errorbar(100 * p, y, xerr=[[100 * (p - lo)], [100 * (hi - p)]], color="#222222", capsize=3, lw=1)
            ax.text(100 * hi + 0.4, y, f"{k}/{len(R)}", va="center", fontsize=8)
        ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([l for _, l in ORDER][::-1], fontsize=8.5)
        ax.set_xlim(0, 14); ax.set_xlabel("missions that went through a too-narrow door (%)")
        ax.set_title(f"{lay} layout", fontsize=10)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("Simulated, 300 worlds per layout (95% Wilson CI)", fontsize=10, y=1.02)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
