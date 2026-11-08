#!/usr/bin/env python3
"""Sprint 6: Doorway detection from segmented wall planes.

Algorithm:
    1. Load all wall PCDs from data/planes/wall_*.pcd.
    2. For each wall, project points to the floor plane (set Z=0),
       then fit a 2D line via PCA on the XY coordinates. Store the
       line direction (unit vector), midpoint, endpoint extremes along
       the line direction, and point count.
    3. Detect doorway candidates over all wall pairs:
       - Case A (collinear gap): two walls whose projected lines are
         nearly collinear (parallel directions |dot| > 0.95, perp
         distance < 0.15 m) with a gap between nearest endpoints in
         [0.6, 2.0] m. Width = gap distance.
       - Case B (parallel-frame doorway): two walls with parallel
         directions (|dot| > 0.95), perpendicular distance in
         [0.6, 2.0] m, and projected extents overlapping by at least
         0.3 m. Width = perpendicular distance.
    4. Write results to data/doorways.json.
    5. Generate figures/sprint6_doorways.png — top-down view.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
PLANES_DIR = ROOT / "data" / "planes"
OUTPUT_JSON = ROOT / "data" / "doorways.json"
OUTPUT_FIG = ROOT / "figures" / "sprint6_doorways.png"

# Detection thresholds
PARALLEL_DOT = 0.95
COLLINEAR_PERP_MAX = 0.15   # m
GAP_RANGE = (0.6, 2.0)      # m
FRAME_PERP_RANGE = (0.6, 2.0)  # m
FRAME_OVERLAP_MIN = 0.3     # m


@dataclass
class WallLine:
    name: str
    direction: np.ndarray    # 2D unit vector along the line
    midpoint: np.ndarray     # 2D centroid
    t_min: float             # min projection along direction
    t_max: float             # max projection along direction
    n_pts: int
    xy: np.ndarray           # projected XY points
    pts3d: np.ndarray        # original 3D points


def load_walls() -> list[WallLine]:
    walls = []
    for path in sorted(PLANES_DIR.glob("wall_*.pcd")):
        pcd = o3d.io.read_point_cloud(str(path))
        pts3d = np.asarray(pcd.points)
        xy = pts3d[:, :2].copy()

        centroid = xy.mean(axis=0)
        centered = xy - centroid
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        direction /= np.linalg.norm(direction)

        proj = centered @ direction
        walls.append(WallLine(
            name=path.stem, direction=direction, midpoint=centroid,
            t_min=float(proj.min()), t_max=float(proj.max()),
            n_pts=len(pts3d), xy=xy, pts3d=pts3d,
        ))
        print(f"  {path.stem}: {len(pts3d)} pts, "
              f"dir=({direction[0]:.3f}, {direction[1]:.3f}), "
              f"span={proj.max() - proj.min():.2f} m")
    return walls


def _ep_a(w: WallLine) -> np.ndarray:
    return w.midpoint + w.t_min * w.direction


def _ep_b(w: WallLine) -> np.ndarray:
    return w.midpoint + w.t_max * w.direction


def _perp_dist(w1: WallLine, w2: WallLine) -> float:
    avg = w1.direction + w2.direction
    avg /= np.linalg.norm(avg)
    perp = np.array([-avg[1], avg[0]])
    return abs(np.dot(w2.midpoint - w1.midpoint, perp))


def _common_dir(w1: WallLine, w2: WallLine) -> np.ndarray:
    avg = w1.direction + w2.direction
    return avg / np.linalg.norm(avg)


def _project_extents(w: WallLine, axis: np.ndarray):
    ea = np.dot(_ep_a(w), axis)
    eb = np.dot(_ep_b(w), axis)
    return min(ea, eb), max(ea, eb)


def detect_doorways(walls: list[WallLine]) -> list[dict]:
    doorways = []
    for w1, w2 in combinations(walls, 2):
        dot = abs(np.dot(w1.direction, w2.direction))
        if dot < PARALLEL_DOT:
            continue

        pd = _perp_dist(w1, w2)
        cd = _common_dir(w1, w2)
        perp = np.array([-cd[1], cd[0]])

        # Case A — collinear gap
        if pd < COLLINEAR_PERP_MAX:
            lo1, hi1 = _project_extents(w1, cd)
            lo2, hi2 = _project_extents(w2, cd)
            gap = max(lo2 - hi1, lo1 - hi2)
            if GAP_RANGE[0] <= gap <= GAP_RANGE[1]:
                if lo2 > hi1:
                    ct = (hi1 + lo2) / 2
                else:
                    ct = (hi2 + lo1) / 2
                avg_perp = (np.dot(w1.midpoint, perp)
                            + np.dot(w2.midpoint, perp)) / 2
                center = ct * cd + avg_perp * perp
                conf = round(min(1.0, max(0.5,
                    1.0 - abs(gap - 0.9) / 1.0 - pd / COLLINEAR_PERP_MAX * 0.2
                )), 2)
                doorways.append(dict(
                    wall_a=w1.name, wall_b=w2.name, case="A",
                    gap_center_xy=center.round(3).tolist(),
                    width_m=round(gap, 3), confidence=conf))

        # Case B — parallel-frame doorway
        elif FRAME_PERP_RANGE[0] <= pd <= FRAME_PERP_RANGE[1]:
            lo1, hi1 = _project_extents(w1, cd)
            lo2, hi2 = _project_extents(w2, cd)
            overlap = min(hi1, hi2) - max(lo1, lo2)
            if overlap >= FRAME_OVERLAP_MIN:
                ct = (max(lo1, lo2) + min(hi1, hi2)) / 2
                avg_perp = (np.dot(w1.midpoint, perp)
                            + np.dot(w2.midpoint, perp)) / 2
                center = ct * cd + avg_perp * perp
                span = (hi1 - lo1 + hi2 - lo2)
                conf = round(min(1.0, max(0.5,
                    0.7 + overlap / span * 0.3 if span > 0 else 0.7
                )), 2)
                doorways.append(dict(
                    wall_a=w1.name, wall_b=w2.name, case="B",
                    gap_center_xy=center.round(3).tolist(),
                    width_m=round(pd, 3), confidence=conf))
    return doorways


def plot_doorways(walls: list[WallLine], doorways: list[dict]) -> None:
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Floor planes as light gray
    for path in sorted(PLANES_DIR.glob("floor_*.pcd")):
        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        ax.scatter(pts[:, 0], pts[:, 1], c="lightgray", s=0.3, alpha=0.3,
                   zorder=1)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
              "#d62728", "#8c564b", "#e377c2", "#7f7f7f"]
    wall_map = {}
    for i, w in enumerate(walls):
        c = colors[i % len(colors)]
        ea, eb = _ep_a(w), _ep_b(w)
        ax.plot([ea[0], eb[0]], [ea[1], eb[1]], color=c, linewidth=4,
                solid_capstyle="round", zorder=3,
                label=f"{w.name} ({w.n_pts} pts)")
        wall_map[w.name] = w

    for d in doorways:
        cx, cy = d["gap_center_xy"]
        w1, w2 = wall_map[d["wall_a"]], wall_map[d["wall_b"]]
        cd = _common_dir(w1, w2)
        perp = np.array([-cd[1], cd[0]])
        if d["case"] == "A":
            hw = 0.15
        else:
            hw = d["width_m"] / 2
        p1 = np.array([cx, cy]) + hw * perp
        p2 = np.array([cx, cy]) - hw * perp
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r--", linewidth=2.5, zorder=4)
        ax.annotate(f"{d['width_m'] * 100:.0f} cm", (cx, cy),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=10, fontweight="bold", color="red", zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Sprint 6: Doorway Detection — {len(doorways)} doorway(s)\n"
                 f"Source: data/planes/wall_*.pcd | {np.datetime64('today')}",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_FIG), dpi=150, facecolor="white")
    plt.close()
    print(f"\nVisualization saved: {OUTPUT_FIG}")


def main():
    print("=" * 60)
    print("Sprint 6: Doorway Detection")
    print("=" * 60)

    print("\n--- Loading wall planes ---")
    walls = load_walls()
    if not walls:
        print("ERROR: No wall PCDs found in data/planes/")
        return

    n_pairs = len(walls) * (len(walls) - 1) // 2
    print(f"\n--- Detecting doorways ({len(walls)} walls, {n_pairs} pairs) ---")
    doorways = detect_doorways(walls)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(doorways, f, indent=2)
    print(f"Doorways written to: {OUTPUT_JSON}")

    print(f"\n{'#':>3}  {'Case':>4}  {'Width':>8}  {'Conf':>5}  Walls")
    print("-" * 50)
    for i, d in enumerate(doorways, 1):
        print(f"{i:3d}  {d['case']:>4}  {d['width_m']*100:6.1f} cm  "
              f"{d['confidence']:5.2f}  {d['wall_a']} + {d['wall_b']}")
    if not doorways:
        print("  (no doorways detected)")

    print("\n--- Generating visualization ---")
    plot_doorways(walls, doorways)


if __name__ == "__main__":
    main()
