"""Figure: the observability cost field c(x) = 1 + lambda * g(x) and the A* paths it produces.

Run: python -m autopass_sim.fig_costfield --out results_tb/fig_costfield.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .world import make_world, FRAME_DEPTH
from .observability import ObsModel, cost_field
from .planners import astar

INK, GRAY, MAROON, GOLD = "#222222", "#5A5A5A", "#7A1E2C", "#C9A227"


def panel(ax, world, model, lam, title):
    cost = cost_field(world, model, lam)
    ext = [0, world.free.shape[1] * world.res, 0, world.free.shape[0] * world.res]
    im = ax.imshow(np.where(world.free, cost, np.nan), origin="lower", extent=ext, cmap="YlOrRd", vmin=1, vmax=1 + lam, alpha=0.85)
    edges = [0.0] + sum([[d.xc - d.hf, d.xc + d.hf] for d in sorted(world.doors, key=lambda d: d.xc)], []) + [world.W]
    for a, b in zip(edges[0::2], edges[1::2]):
        ax.add_patch(Rectangle((a, world.wall_y), b - a, FRAME_DEPTH, color=GRAY))
    for bx in world.boxes:
        ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], color="#BDB8B1"))
    ones = np.ones_like(world.free, float)
    p0 = astar(world, world.start, world.goal, ones)
    p1 = astar(world, world.start, world.goal, cost)
    ax.plot(p0[:, 0], p0[:, 1], "--", color="#1F4E79", lw=2, label="A*, λ = 0 (shortest)")
    ax.plot(p1[:, 0], p1[:, 1], "-", color="#111111", lw=2.4, label=f"observability-aware A*, λ = {lam:g}")
    ax.plot(*world.start, "o", color=INK); ax.plot(*world.goal, "*", color=INK, ms=11)
    ax.set_xlim(0, world.W); ax.set_ylim(0, world.H); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10, loc="left")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = ObsModel(a.calib, rho_calib="calib_rho.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)
    w1 = make_world(np.random.default_rng(1016), "corridor")
    w2 = make_world(np.random.default_rng(1022), "room")
    panel(axes[0], w1, model, a.lam, "(a) corridor layout")
    im = panel(axes[1], w2, model, a.lam, "(b) room layout")
    axes[0].legend(fontsize=8, loc="upper left", frameon=True)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("cost per metre c(x)   (1 = trusted view or outside approach zones)")
    fig.suptitle("Simulated: observability cost field and the paths A* finds with and without it", fontsize=11)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
