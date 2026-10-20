"""Side-by-side 2D replay of two methods on the same world (MP4 + still frames).

Run:
  python -m autopass_sim.render2d --layout corridor --seed 1016 --methods "B1,P-A*(4)" \
      --out results/replay_corridor_1016.mp4 --stills results/replay_corridor_1016_stills.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.animation import FuncAnimation, FFMpegWriter
from .world import make_world, FRAME_DEPTH
from .observability import ObsModel
from .mission import Config, run_mission, Tracer
from .experiments import METHODS
from .planners import resample

INK, GRAY, LIGHT = "#222222", "#5A5A5A", "#D9D6D2"
MAROON, GOLD, TEAL, ORANGE = "#7A1E2C", "#C9A227", "#2F6F73", "#D9822B"
LABEL = {"B1": "Baseline: shortest path, decide while walking",
         "B2": "Shortest path + trusted decision rule",
         "P-A*(4)": "Proposed: observability-aware A* + trusted rule",
         "P-A*(4)n": "Observability-aware path, naive rule",
         "P-RRT*(4)": "Proposed: observability-aware RRT*"}


def simulate(world, method, model, seed):
    tr = Tracer()
    res = run_mission(world, Config(tracer=tr, **METHODS[method]), model, np.random.default_rng(seed * 101 + 7))
    frames = list(tr.frames)
    if res.outcome in ("success", "unsafe") and res.paths:
        pts, _ = resample(res.paths[-1], 0.25)              # final leg to the goal (no scanning needed)
        for p in pts[1:]:
            frames.append(dict(pos=p, pts=np.zeros((0, 3)), est=frames[-1]["est"] if frames else {}))
    return res, frames, tr.events


def trusted_mask(world, model, n=360):
    xs = np.linspace(0, world.W, n); ys = np.linspace(0, world.wall_y, max(60, int(n * world.wall_y / world.W)))
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X, bool)
    for d in world.doors:
        dx, dy = X - d.xc, d.yw - Y
        psi = np.degrees(np.arctan(np.maximum(0, np.abs(dx) - d.w / 2) / (d.r + np.maximum(dy, 1e-6))))
        Z |= (psi <= model.psi_safe) & (np.hypot(dx, dy) <= model.d_max)
    return X, Y, Z


def draw_static(ax, world, model, title):
    X, Y, Z = trusted_mask(world, model)
    ax.contourf(X, Y, Z.astype(float), levels=[0.5, 1.5], colors=["#E1EFEE"], zorder=0)
    edges = [0.0] + sum([[d.xc - d.hf, d.xc + d.hf] for d in sorted(world.doors, key=lambda d: d.xc)], []) + [world.W]
    for a, b in zip(edges[0::2], edges[1::2]):
        ax.add_patch(Rectangle((a, world.wall_y), b - a, FRAME_DEPTH, color=GRAY, zorder=2))
    for bx in world.boxes:
        ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], color=LIGHT, zorder=2))
    ax.add_patch(Rectangle((0, 0), world.W, world.H, fill=False, ec=GRAY, lw=2, zorder=2))
    ax.plot(*world.goal, "*", color=INK, ms=14, zorder=5)
    ax.text(world.goal[0], world.goal[1] + 0.35, "goal", ha="center", fontsize=9, color=INK)
    ax.set_xlim(-0.3, world.W + 0.3); ax.set_ylim(-0.3, world.H + 0.3); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, color=INK, loc="left")


class Panel:
    def __init__(self, ax, world, model, method, res, frames, events, agent_w):
        self.ax, self.world, self.res, self.frames, self.events = ax, world, res, frames, events
        self.agent_w = agent_w
        draw_static(ax, world, model, LABEL.get(method, method))
        self.trail, = ax.plot([], [], "-", color=MAROON if method.startswith("P") else GOLD, lw=2.2, zorder=4)
        self.rays = ax.plot([], [], "-", color=GOLD, lw=0.4, alpha=0.35, zorder=3)[0]
        self.hits = ax.scatter([], [], s=6, zorder=6)
        self.robot = Circle((0, 0), 0.2, fc="white", ec=INK, lw=1.6, zorder=7)
        ax.add_patch(self.robot)
        self.arrow = None
        self.chips = {}
        for d in world.doors:
            self.chips[d.id] = ax.text(d.xc, world.wall_y + 0.45, "", ha="center", va="bottom", fontsize=8.5,
                                       zorder=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRAY, lw=0.8))
            m = d.w * 100 - agent_w
            ax.text(d.xc, world.wall_y - 0.42, f"true {d.w*100:.1f} cm", ha="center", va="top", fontsize=7,
                    color=TEAL if m >= 0 else MAROON, zorder=8)
        self.hud = ax.text(0.01, -0.02, "", transform=ax.transAxes, va="top", fontsize=9, color=INK)
        self.banner = ax.text(0.5, 0.6, "", transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                              color="white", fontweight="bold", zorder=10,
                              bbox=dict(boxstyle="round,pad=0.4", fc=GRAY, ec="none", alpha=0.0))

    def update(self, i):
        n = len(self.frames)
        k = min(i, n - 1)
        f = self.frames[k]
        path = np.array([g["pos"] for g in self.frames[:k + 1]])
        self.trail.set_data(path[:, 0], path[:, 1])
        p = f["pos"]
        self.robot.center = (p[0], p[1])
        if self.arrow is not None:
            self.arrow.remove()
        j = min(k + 1, n - 1)
        h = self.frames[j]["pos"] - p if j != k else (p - self.frames[max(k - 1, 0)]["pos"])
        h = h / (np.linalg.norm(h) + 1e-9) * 0.35
        self.arrow = self.ax.add_patch(FancyArrow(p[0], p[1], h[0], h[1], width=0.05, head_width=0.18,
                                                  head_length=0.15, color=INK, zorder=8))
        Q = f["pts"] if i < n else np.zeros((0, 3))
        if len(Q):
            sel = Q[:: max(1, len(Q) // 60)]
            xs, ys = [], []
            for q in sel:
                xs += [p[0], q[0], np.nan]; ys += [p[1], q[1], np.nan]
            self.rays.set_data(xs, ys)
            self.hits.set_offsets(Q[:, :2])
            trusted_now = any(v[3] for v in f["est"].values()) if f["est"] else False
            self.hits.set_color(TEAL if trusted_now else ORANGE)
        else:
            self.rays.set_data([], []); self.hits.set_offsets(np.zeros((0, 2)))
        decided = {}
        for (fi, did, dec) in self.events:
            if fi <= k:
                decided[did] = dec
        for d in self.world.doors:
            e = f["est"].get(d.id)
            if d.id in decided:
                dec = decided[d.id]
                col = {"PASS": TEAL, "FAIL": MAROON, "FLAG": GRAY}[dec]
                txt = f"D{d.id}  {dec}"
                if e is not None and np.isfinite(e[0]):
                    txt += f"\nest {e[0] + self.agent_w:.1f} cm"
                self.chips[d.id].set_text(txt)
                self.chips[d.id].set_color("white")
                self.chips[d.id].get_bbox_patch().set(fc=col, ec=col)
            else:
                txt = f"D{d.id}  ?"
                if e is not None and np.isfinite(e[0]):
                    txt += f"\nest {e[0] + self.agent_w:.1f} ± {2 * e[1]:.1f}"
                self.chips[d.id].set_text(txt)
                self.chips[d.id].set_color(INK)
                self.chips[d.id].get_bbox_patch().set(fc="white", ec=GRAY)
        t = k * 0.5
        dec_txt = "   ".join(f"D{did}:{dec}" for (fi, did, dec) in self.events if fi <= k)
        self.hud.set_text(f"t = {t:4.1f} s     decisions: {dec_txt or '—'}")
        if k == n - 1:
            if self.res.outcome == "unsafe":
                d = [x for x in self.res.decisions if x[2] == "PASS"][-1]
                self.banner.set_text(f"UNSAFE: committed to D{d[0]}, {abs(d[1]):.1f} cm too narrow for a wheelchair")
                self.banner.get_bbox_patch().set(fc=MAROON, alpha=0.95)
            elif self.res.outcome == "success":
                self.banner.set_text(f"Reached goal through a passable door  ({self.res.time:.0f} s)")
                self.banner.get_bbox_patch().set(fc=TEAL, alpha=0.95)
            else:
                self.banner.set_text(f"No passable route confirmed ({self.res.outcome})")
                self.banner.get_bbox_patch().set(fc=GRAY, alpha=0.95)
        else:
            self.banner.get_bbox_patch().set(alpha=0.0); self.banner.set_text("")


def build(world, model, methods, seed, agent_w=81.3):
    h = 12.8 / len(methods) * world.H / world.W + 1.3
    fig, axes = plt.subplots(1, len(methods), figsize=(12.8, h), dpi=150)
    fig.patch.set_facecolor("#FBFBF8")
    panels = []
    for ax, m in zip(np.atleast_1d(axes), methods):
        res, frames, events = simulate(world, m, model, seed)
        panels.append(Panel(ax, world, model, m, res, frames, events, agent_w))
    fig.text(0.01, 1 - 0.12 / fig.get_figheight(), "AutoPass Phase 2 — simulated mission replay (not robot data)", fontsize=12,
             fontweight="bold", color=INK, va="top")
    fig.text(0.01, 1 - 0.38 / fig.get_figheight(), "Shaded: trusted viewing zone (near-jamb angle ψ ≤ 12.5°, ≤ 4 m). Dots: LiDAR hits on door "
             "frames (teal = trusted view, orange = oblique/occluded). Agent: wheelchair, needs 81.3 cm.",
             fontsize=8.5, color=GRAY, va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.95 / h, bottom=0.45 / h, wspace=0.04)
    return fig, panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="corridor")
    ap.add_argument("--seed", type=int, default=1016)
    ap.add_argument("--methods", default="B1,P-A*(4)")
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="replay.mp4")
    ap.add_argument("--stills", default=None, help="PNG with 3 key moments per method")
    a = ap.parse_args()
    model = ObsModel(a.calib, rho_calib="calib_rho.json")
    world = make_world(np.random.default_rng(a.seed), a.layout)
    methods = a.methods.split(",")
    fig, panels = build(world, model, methods, a.seed)
    n = max(len(p.frames) for p in panels) + a.fps * 2      # hold the last frame 2 s
    def step(i):
        for p in panels:
            p.update(i)
        return []
    if a.out:
        anim = FuncAnimation(fig, step, frames=n, blit=False)
        anim.save(a.out, writer=FFMpegWriter(fps=a.fps, bitrate=4000))
        print("wrote", a.out, n, "frames")
    plt.close(fig)
    if a.stills:
        fracs = [0.3, 1.0]
        import textwrap
        sf, axs = plt.subplots(len(methods), len(fracs), figsize=(7.0, 2.75 * len(methods) * world.H / world.W * 1.6), dpi=220)
        axs = np.atleast_2d(axs)
        for r, m in enumerate(methods):
            res, frames, events = simulate(world, m, model, a.seed)
            for c, fr in enumerate(fracs):
                p = Panel(axs[r, c], world, model, m, res, frames, events, 81.3)
                k = int(round(fr * (len(frames) - 1)))
                p.update(k)
                p.banner.set_fontsize(8.5); p.banner.set_y(0.24)
                p.banner.set_text(textwrap.fill(p.banner.get_text(), 30))
                for ch in p.chips.values():
                    ch.set_fontsize(8)
                for t in axs[r, c].texts:
                    if t.get_text().startswith("true"):
                        t.set_fontsize(8)
                    if t.get_text() == "goal":
                        t.set_fontsize(9)
                axs[r, c].set_title(("" if c else LABEL.get(m, m) + "\n") + f"t = {k*0.5:.1f} s", fontsize=9.5, loc="left")
                p.hud.set_text("")
        sf.tight_layout(); sf.savefig(a.stills); plt.close(sf)
        print("wrote", a.stills)


if __name__ == "__main__":
    main()
