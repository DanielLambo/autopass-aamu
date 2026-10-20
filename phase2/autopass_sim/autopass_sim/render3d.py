"""3D renders from the simulator.

  door   close-up of one door frame scanned from a square-on and an oblique pose: LiDAR rays,
         hits on the stop (teal) vs the frame corner/face (orange), and the edge GAIT picks
  world  perspective view of a floor plan with extruded walls, both methods' paths and the
         door-frame LiDAR hits collected along the proposed path

Run:  python -m autopass_sim.render3d door  --out results/render3d_door.png
      python -m autopass_sim.render3d world --layout corridor --seed 1016 --out results/render3d_world.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .world import World, Door, make_world, FRAME_DEPTH, DOOR_HEIGHT
from .sensor import scan, estimate_width, view_geometry, SENSOR_Z
from .observability import ObsModel
from .render2d import simulate, LABEL

INK, GRAY, MAROON, GOLD, TEAL, ORANGE = "#222222", "#5A5A5A", "#7A1E2C", "#C9A227", "#2F6F73", "#D9822B"


def box_faces(x0, y0, z0, x1, y1, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [v[i] for i in idx]


def add_box(ax, b, color, alpha, ec=None, lw=0.3):
    pc = Poly3DCollection(box_faces(*b), facecolor=color, alpha=alpha, edgecolor=ec or color, linewidths=lw)
    ax.add_collection3d(pc)


def frame_boxes(door, x_lo, x_hi, H=2.4):
    """Wall slabs, header, and both stops of one door (3D boxes)."""
    xc, yw, hf, s, r, D = door.xc, door.yw, door.hf, door.s, door.r, door.D
    ws = D - 2 * r
    walls = [(x_lo, yw, 0, xc - hf, yw + D, H), (xc + hf, yw, 0, x_hi, yw + D, H),
             (xc - hf, yw, DOOR_HEIGHT, xc + hf, yw + D, H)]
    stops = [(xc - hf, yw + r, 0, xc - hf + s, yw + r + ws, DOOR_HEIGHT),
             (xc + hf - s, yw + r, 0, xc + hf, yw + r + ws, DOOR_HEIGHT)]
    return walls, stops


def on_stop(P, door, eps=0.004):
    xc, yw, hf, s, r, D = door.xc, door.yw, door.hf, door.s, door.r, door.D
    ws = D - 2 * r
    iny = (P[:, 1] >= yw + r - eps) & (P[:, 1] <= yw + r + ws + eps)
    inl = (P[:, 0] >= xc - hf - eps) & (P[:, 0] <= xc - hf + s + eps)
    inr = (P[:, 0] >= xc + hf - s - eps) & (P[:, 0] <= xc + hf + eps)
    return iny & (inl | inr)


def render_door(out, seed=3):
    rng = np.random.default_rng(seed)
    door = Door(0, 4.0, 5.0, 0.837, 0.0159, 0.0492)
    world = World("door", 8, 8, 5.0, [door], [], np.array([4, 1.0]), np.array([4, 7.0])).build()
    poses = [("Square-on view (ψ = 0°)", (4.0, 3.6)), ("Oblique view (ψ ≈ 26°)", (2.95, 3.3))]
    fig = plt.figure(figsize=(12.8, 5.6), dpi=200)
    fig.patch.set_facecolor("#FBFBF8")
    for k, (title, pos) in enumerate(poses):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.set_facecolor("#FBFBF8")
        walls, stops = frame_boxes(door, 3.0, 5.0)
        for b in walls:
            add_box(ax, b, "#BDB8B1", 0.35, ec="#8A857E")
        for b in stops:
            add_box(ax, b, TEAL, 0.55, ec=TEAL)
        P = np.vstack([scan(world, pos, rng, doors=[door]) for _ in range(4)])
        m = (P[:, 1] > door.yw - 0.03) & (P[:, 1] < door.yw + door.D + 0.03) & (P[:, 2] > 0.3) & (P[:, 2] < 1.9) \
            & (np.abs(P[:, 0] - door.xc) < door.hf + 0.6)
        P = P[m]
        st = on_stop(P, door)
        ax.scatter(P[~st, 0], P[~st, 1], P[~st, 2], s=2.5, c=ORANGE, depthshade=False, label="hits on frame face / corner")
        ax.scatter(P[st, 0], P[st, 1], P[st, 2], s=4, c=TEAL, depthshade=False, label="hits on the stop (true edge)")
        sel = P[np.random.default_rng(1).choice(len(P), min(70, len(P)), replace=False)]
        for q in sel:
            ax.plot([pos[0], q[0]], [pos[1], q[1]], [SENSOR_Z, q[2]], color=GOLD, lw=0.3, alpha=0.35)
        ax.scatter([pos[0]], [pos[1]], [SENSOR_Z], s=60, c=INK, marker="o")
        ax.text(pos[0], pos[1] - 0.15, SENSOR_Z + 0.15, "Go2 + L1", fontsize=8, color=INK)
        w_est = np.nanmedian([estimate_width(scan(world, pos, rng, doors=[door]), door) for _ in range(30)])
        d, psi = view_geometry(door, pos)
        ax.set_title(f"{title}\nmeasured {w_est*100:.1f} cm vs true {door.w*100:.1f} cm "
                     f"({(w_est - door.w)*100:+.1f} cm)", fontsize=10, color=INK)
        ax.set_xlim(3.0, 5.0); ax.set_ylim(min(pos[1], 3.2), 5.4); ax.set_zlim(0, 2.4)
        ax.set_box_aspect((2.0, 5.4 - min(pos[1], 3.2), 2.4))
        ax.view_init(elev=28, azim=-115 if k == 0 else -128)
        ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8); ax.set_zlabel("z (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False)
    # title omitted: the report caption carries it
    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.02, wspace=0.0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
    print("wrote", out)


def render_world(out, layout, seed, methods, calib):
    model = ObsModel(calib)
    world = make_world(np.random.default_rng(seed), layout)
    runs = [(m, *simulate(world, m, model, seed)) for m in methods]
    fig = plt.figure(figsize=(12.8, 6.4), dpi=200)
    fig.patch.set_facecolor("#FBFBF8")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#FBFBF8")
    floor = [[[0, 0, 0], [world.W, 0, 0], [world.W, world.H, 0], [0, world.H, 0]]]
    ax.add_collection3d(Poly3DCollection(floor, facecolor="#ECE8E2", alpha=1.0, edgecolor="#CFCAC2", zorder=0))
    for bx in world.boxes:
        add_box(ax, (bx[0], bx[1], 0, bx[2], bx[3], 0.8), "#D6D2CC", 0.9, ec="#A8A39C")
    for (m, res, frames, events), col in zip(runs, [GOLD, MAROON]):
        p = np.array([f["pos"] for f in frames])
        ax.plot(p[:, 0], p[:, 1], np.full(len(p), 0.03), color=col, lw=2.6, zorder=3,
                label=f"{LABEL.get(m, m)}: {res.outcome}, {res.time:.0f} s")
        ax.scatter(*p[-1], 0.05, s=40, c=col, zorder=4)
    _, res, frames, _ = runs[-1]
    pts = np.vstack([f["pts"] for f in frames if len(f["pts"])])
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c=pts[:, 2], cmap="viridis", alpha=0.8,
               depthshade=False, zorder=5)
    edges = [0.0] + sum([[d.xc - d.hf, d.xc + d.hf] for d in sorted(world.doors, key=lambda d: d.xc)], []) + [world.W]
    for a, b in zip(edges[0::2], edges[1::2]):
        add_box(ax, (a, world.wall_y, 0, b, world.wall_y + FRAME_DEPTH, 2.4), "#BDB8B1", 0.35, ec="#8A857E")
    for d in world.doors:
        add_box(ax, (d.xc - d.hf, d.yw, DOOR_HEIGHT, d.xc + d.hf, d.yw + d.D, 2.4), "#BDB8B1", 0.35, ec="#8A857E")
        m = d.w * 100 - 81.3
        ax.text(d.xc, d.yw, 2.75, f"D{d.id}\n{d.w*100:.1f} cm", fontsize=8.5, ha="center", zorder=10,
                color=TEAL if m >= 0 else MAROON, fontweight="bold")
    ax.scatter(*world.goal, 0.05, marker="*", s=200, c=INK, zorder=6)
    ax.text(world.goal[0], world.goal[1], 0.35, "goal", fontsize=8, ha="center", zorder=10)
    ax.text(world.start[0], world.start[1], 0.35, "start", fontsize=8, ha="center", zorder=10)
    ax.set_xlim(0, world.W); ax.set_ylim(0, world.H); ax.set_zlim(0, 2.6)
    ax.set_box_aspect((world.W, world.H, 2.6 * 1.4), zoom=1.05)
    ax.view_init(elev=30, azim=-72)
    ax.set_position([-0.08, -0.12, 1.16, 1.12])
    ax.set_axis_off()
    ax.legend(loc="upper left", fontsize=9, frameon=False, bbox_to_anchor=(0.02, 0.98))
    fig.text(0.5, 0.97, "Simulated corridor mission in 3D (not robot data): extruded walls and frames, both robot paths, "
             "and door-frame LiDAR hits (colored by height) from the proposed path", ha="center", fontsize=10, color=INK)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["door", "world"])
    ap.add_argument("--layout", default="corridor")
    ap.add_argument("--seed", type=int, default=1016)
    ap.add_argument("--methods", default="B1,P-A*(4)")
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.what == "door":
        render_door(a.out)
    else:
        render_world(a.out, a.layout, a.seed, a.methods.split(","), a.calib)


if __name__ == "__main__":
    main()
