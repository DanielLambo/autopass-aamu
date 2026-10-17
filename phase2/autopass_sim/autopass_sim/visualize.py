"""Draw one world with the paths two methods took.

Run:  python -m autopass_sim.visualize --layout corridor --seed 1003 --out fig_example.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .world import make_world, FRAME_DEPTH
from .observability import ObsModel
from .mission import Config, run_mission
from .experiments import METHODS
from .sensor import view_geometry

MAROON, GOLD, GRAY, TEAL = "#7A1E2C", "#C9A227", "#555555", "#2F6F73"


def draw(world, runs, model, path):
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * world.H / world.W * 0.9))
    ax.set_aspect("equal")
    # trusted viewing zone per door
    xs = np.linspace(0, world.W, 400); ys = np.linspace(0, world.wall_y, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X, bool)
    for d in world.doors:
        dx, dy = X - d.xc, d.yw - Y
        dist = np.hypot(dx, dy)
        psi = np.degrees(np.arctan(np.maximum(0, np.abs(dx) - d.w / 2) / (d.r + np.maximum(dy, 1e-6))))
        Z |= (psi <= model.psi_safe) & (dist <= model.d_max)
    ax.contourf(X, Y, Z.astype(float), levels=[0.5, 1.5], colors=["#E3F0EF"])
    # walls and doors
    edges = [0.0] + sum([[d.xc - d.hf, d.xc + d.hf] for d in sorted(world.doors, key=lambda d: d.xc)], []) + [world.W]
    for a, b in zip(edges[0::2], edges[1::2]):
        ax.add_patch(Rectangle((a, world.wall_y), b - a, FRAME_DEPTH, color="#9E9E9E"))
    for bx in world.boxes:
        ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], color="#D6D6D6"))
    ax.add_patch(Rectangle((0, 0), world.W, world.H, fill=False, ec=GRAY, lw=1.5))
    for d in world.doors:
        m = d.w * 100 - 81.3
        ax.text(d.xc, world.wall_y + 0.35, f"D{d.id}\n{d.w*100:.1f} cm ({m:+.1f})", ha="center", va="bottom",
                fontsize=8, color=TEAL if m >= 0 else MAROON)
    styles = [(GOLD, "-"), (MAROON, "-")]
    for (name, res), (col, ls) in zip(runs, styles):
        for k, p in enumerate(res.paths):
            ax.plot(p[:, 0], p[:, 1], ls, color=col, lw=2, label=name if k == 0 else None, alpha=0.9)
        dec = ", ".join(f"D{x[0]} {x[2]}" for x in res.decisions)
        ax.plot([], [], " ", label=f"   {dec}; {res.time:.0f} s")
    ax.plot(*world.start, "o", color="k", ms=6); ax.text(world.start[0] + 0.15, world.start[1] - 0.35, "start", fontsize=8)
    ax.plot(*world.goal, "*", color="k", ms=10); ax.text(world.goal[0] - 0.2, world.goal[1] + 0.25, "goal", fontsize=8, ha="right")
    ax.set_xlim(-0.2, world.W + 0.2); ax.set_ylim(-0.2, world.H + 0.2)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.set_title("shaded: trusted viewing zone (ψ ≤ 12.5°, d ≤ 4 m); door labels: clear width (margin vs 81.3 cm)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="corridor")
    ap.add_argument("--seed", type=int, default=1003)
    ap.add_argument("--methods", default="B2,P-A*(4)")
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--out", default="fig_example.png")
    a = ap.parse_args()
    model = ObsModel(a.calib, rho_calib="calib_rho.json")
    world = make_world(np.random.default_rng(a.seed), a.layout)
    runs = []
    for m in a.methods.split(","):
        r = run_mission(world, Config(**METHODS[m]), model, np.random.default_rng(a.seed * 101 + 7))
        runs.append((m, r))
        print(m, r.outcome, round(r.time, 1), r.decisions)
    draw(world, runs, model, a.out)


if __name__ == "__main__":
    main()
