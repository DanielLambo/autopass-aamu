"""Top-down plot: v2-A's planned path vs the Go2 trajectories in Isaac Lab, with non-foot contacts (simulated).

    python gui/plot_walk.py --dir results_rl/gui --seed 5081
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rl/gui")
    ap.add_argument("--seed", type=int, default=5081)
    a = ap.parse_args()
    P = json.load(open(os.path.join(a.dir, f"plan_{a.seed}.json")))
    w = P["world"]
    runs = [(t, os.path.join(a.dir, f"trajectory_{t}.csv"), os.path.join(a.dir, f"walkability_{t}.json"))
            for t in ("faithful", "straight")]
    runs = [r for r in runs if os.path.exists(r[1])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for ax, zoom in zip(axes, (False, True)):
        ax.add_patch(Rectangle((0, 0), w["W"], w["H"], fill=False, lw=1.5, ec="0.3"))
        edges = [0.0] + [e for d in sorted(w["doors"], key=lambda d: d["xc"]) for e in (d["xc"] - d["hf"], d["xc"] + d["hf"])] + [w["W"]]
        for x0, x1 in zip(edges[0::2], edges[1::2]):
            ax.add_patch(Rectangle((x0, w["wall_y"]), x1 - x0, w["wall_depth"], color="0.45"))
        for d in w["doors"]:
            ws = d["D"] - 2 * d["r"]
            for x in (d["xc"] - d["hf"], d["xc"] + d["hf"] - d["s"]):
                ax.add_patch(Rectangle((x, d["yw"] + d["r"]), d["s"], ws, color="k"))
            col = "#2a9d4b" if d["margin_cm"] >= 3 else "#e69f00" if d["margin_cm"] >= 0 else "#c0392b"
            ax.text(d["xc"], d["yw"] - 0.35, f"door {d['id']}\n{d['margin_cm']:+.2f} cm", ha="center", va="top",
                    fontsize=7.5, color=col, clip_on=True)
        for x0, y0, x1, y1 in w["boxes"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#c8a97e"))
        for k, st in enumerate(P["plan"]):
            if "pts" in st:
                q = np.array(st["pts"])
                ax.plot(q[:, 0], q[:, 1], "--", color="0.2", lw=1.2, label="v2-A plan (RL env path)" if k == 0 else None)
        cols = {"faithful": "#d95f02", "straight": "#1b9e77"}
        names = {"faithful": "Go2, plan followed as given", "straight": "Go2, straight pass through door centre"}
        for t, f, js in runs:
            d = pd.read_csv(f)
            ax.plot(d.x, d.y, color=cols[t], lw=1.6, label=names[t])
            c = d[d.nonfoot_contact_N > 1.0]
            if len(c):
                ax.plot(c.x, c.y, "x", color=cols[t], ms=6, mew=1.5, label=f"non-foot contact ({t})")
        ax.plot(*w["start"], "o", color="0.2"); ax.plot(*w["goal"], "*", color="#2a9d4b", ms=13)
        ax.set_aspect("equal")
        if zoom:
            d1 = next(d for d in w["doors"] if d["id"] == P["plan"][-1]["door"])
            ax.set_xlim(d1["xc"] - 1.6, d1["xc"] + 0.9); ax.set_ylim(d1["yw"] - 1.0, d1["yw"] + 1.2)
            ax.set_title(f"zoom: commit door {d1['id']}", fontsize=10)
        else:
            ax.set_xlim(-0.3, w["W"] + 0.3); ax.set_ylim(-0.3, w["H"] + 0.3)
            ax.legend(fontsize=7.5, loc="upper right", frameon=True)
            ax.set_title(f"Seed {a.seed}: v2-A plan executed by the frozen Go2 policy in Isaac Lab (simulated)", fontsize=10.5)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.text(0.5, 0.01, "Simulated. Root (base) trajectory of the Go2; x = steps with contact force > 1 N on a non-foot link. "
             "Dashed: path from the RL env (point robot, 0.2 m grid inflation).", ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(a.dir, f"walk_paths_{a.seed}.png")
    fig.savefig(out, dpi=150); print("wrote", out)


if __name__ == "__main__":
    main()
