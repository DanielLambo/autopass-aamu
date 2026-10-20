"""Angle-sweep animation: the Go2 + L1 walks an arc around one steel door frame at a fixed distance.
Left: 3D view of the frame, the LiDAR hits (teal = on the stop, orange = frame corner/face) and the rays.
Right: width error vs viewing angle, filled in as the robot moves, with the trusted zone and psi*.

Run:  python -m autopass_sim.render_sweep --out results/sweep_door.mp4
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from .world import World, Door
from .sensor import scan, estimate_width, view_geometry, SENSOR_Z
from .render3d import frame_boxes, add_box, on_stop, INK, GRAY, MAROON, GOLD, TEAL, ORANGE

BG = "#FBFBF8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sweep_door.mp4")
    ap.add_argument("--dist", type=float, default=1.6)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    door = Door(0, 4.0, 5.0, 0.837, 0.0159, 0.0492)
    world = World("door", 8, 8, 5.0, [door], [], np.array([4, 1.0]), np.array([4, 7.0])).build()
    thetas = np.radians(np.linspace(0, 42, a.frames))
    poses = [(door.xc - a.dist * np.sin(t), door.yw - a.dist * np.cos(t)) for t in thetas]
    # precompute errors (median of 12 scans per pose) so the curve is stable
    psis, errs = [], []
    for pos in poses:
        _, psi = view_geometry(door, pos)
        w = np.nanmedian([estimate_width(scan(world, pos, rng, doors=[door]), door) for _ in range(12)])
        psis.append(psi); errs.append((w - door.w) * 100)
    psis, errs = np.array(psis), np.array(errs)
    psi_star = np.degrees(np.arctan(door.s / door.r))

    fig = plt.figure(figsize=(12.8, 5.4), dpi=150)
    fig.patch.set_facecolor(BG)
    ax3 = fig.add_axes([0.0, 0.02, 0.56, 0.9], projection="3d")
    ax2 = fig.add_axes([0.63, 0.14, 0.34, 0.7])
    writer = FFMpegWriter(fps=a.fps, bitrate=4000)
    walls, stops = frame_boxes(door, 3.0, 5.0)
    with writer.saving(fig, a.out, dpi=150):
        for i, pos in enumerate(poses):
            ax3.cla(); ax3.set_facecolor(BG)
            for b in walls:
                add_box(ax3, b, "#BDB8B1", 0.35, ec="#8A857E")
            for b in stops:
                add_box(ax3, b, TEAL, 0.55, ec=TEAL)
            P = np.vstack([scan(world, pos, rng, doors=[door]) for _ in range(3)])
            m = (P[:, 1] > door.yw - 0.03) & (P[:, 1] < door.yw + door.D + 0.03) & (P[:, 2] > 0.3) & (P[:, 2] < 1.9) \
                & (np.abs(P[:, 0] - door.xc) < door.hf + 0.6)
            P = P[m]; st = on_stop(P, door)
            ax3.scatter(P[~st, 0], P[~st, 1], P[~st, 2], s=2, c=ORANGE, depthshade=False, label="hits on frame corner / face")
            ax3.scatter(P[st, 0], P[st, 1], P[st, 2], s=3, c=TEAL, depthshade=False, label="hits on the stop (true edge)")
            sel = P[np.random.default_rng(i).choice(len(P), min(50, len(P)), replace=False)] if len(P) else P
            for q in sel:
                ax3.plot([pos[0], q[0]], [pos[1], q[1]], [SENSOR_Z, q[2]], color=GOLD, lw=0.3, alpha=0.35)
            arc = np.array(poses[: i + 1])
            ax3.plot(arc[:, 0], arc[:, 1], np.zeros(len(arc)), color=MAROON, lw=1.5)
            ax3.scatter([pos[0]], [pos[1]], [SENSOR_Z], s=50, c=INK)
            ax3.text(pos[0], pos[1] - 0.15, SENSOR_Z + 0.18, "Go2 + L1", fontsize=8, color=INK)
            ax3.set_xlim(2.3, 5.0); ax3.set_ylim(3.3, 5.4); ax3.set_zlim(0, 2.4)
            ax3.set_box_aspect((2.7, 2.1, 2.4))
            ax3.view_init(elev=26, azim=-118 + 0.25 * i)
            ax3.set_xlabel("x (m)", fontsize=8); ax3.set_ylabel("y (m)", fontsize=8); ax3.set_zlabel("z (m)", fontsize=8)
            ax3.tick_params(labelsize=7)
            ax3.legend(loc="upper left", fontsize=8, frameon=False)
            ax3.set_title(f"viewing angle ψ = {psis[i]:4.1f}°    measured width error {errs[i]:+.1f} cm",
                          fontsize=11, color=MAROON if psis[i] > 12.5 else TEAL, pad=2)

            ax2.cla(); ax2.set_facecolor(BG)
            ax2.axvspan(0, 12.5, color=TEAL, alpha=0.08)
            ax2.text(6.2, 2.25, "trusted\nψ ≤ 12.5°", ha="center", fontsize=8, color=TEAL)
            ax2.axvline(psi_star, color=GRAY, ls="--", lw=1)
            ax2.text(psi_star + 0.6, -0.85, f"stop fully hidden\nψ* = {psi_star:.0f}°", fontsize=8, color=GRAY)
            ax2.axhline(0, color=INK, lw=0.6)
            ax2.plot(psis[: i + 1], errs[: i + 1], color=MAROON, lw=2)
            ax2.scatter([psis[i]], [errs[i]], color=MAROON, s=30, zorder=3)
            ax2.set_xlim(0, max(psis) + 1); ax2.set_ylim(-1.2, 2.6)
            ax2.set_xlabel("viewing angle ψ (deg)", fontsize=9)
            ax2.set_ylabel("width error (cm)  [+ = reads wider]", fontsize=9)
            ax2.tick_params(labelsize=8)
            for s in ("top", "right"):
                ax2.spines[s].set_visible(False)
            ax2.set_title("Error grows with angle and does not average out", fontsize=10, color=INK)
            fig.text(0.985, 0.015, "SIMULATED", ha="right", fontsize=8, color=GRAY) if i == 0 else None
            writer.grab_frame()
    # last frame still
    fig.savefig(a.out.replace(".mp4", "_last.png"), facecolor=BG)
    print("wrote", a.out, "psi range", psis.min().round(1), psis.max().round(1), "err range", errs.min().round(2), errs.max().round(2))


if __name__ == "__main__":
    main()
