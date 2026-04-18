#!/usr/bin/env python3
"""Improved segmentation visualization — loads planes from PCD files, no re-segmentation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

PLANES_DIR = Path.home() / "autopass" / "data" / "planes"
VIS_PATH = Path.home() / "autopass" / "data" / "segmentation_result_v2.png"

# Per-label rendering config: (color, alpha, size)
STYLE = {
    "floor_main":           ((0.90, 0.10, 0.10), 0.5,  3),   # red
    "floor_secondary":      ((1.00, 0.55, 0.00), 0.5,  3),   # orange
    "horizontal_other_01":  ((0.75, 0.75, 0.15), 0.6,  5),   # yellow-olive
    "horizontal_other_02":  ((0.60, 0.60, 0.20), 0.6,  5),   # darker olive
    "ceiling":              ((0.60, 0.20, 0.80), 0.5,  3),   # purple
    "remaining":            ((0.78, 0.78, 0.78), 0.08, 1),   # light gray, faded
}

# Walls get tab10 colors
TAB10 = plt.cm.tab10.colors
WALL_STYLE_BASE = (1.0, 15)  # alpha, size


def load_planes():
    """Load all plane PCDs and return list of (label, points, style)."""
    planes = []
    for pcd_path in sorted(PLANES_DIR.glob("*.pcd")):
        label = pcd_path.stem
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            continue

        if label in STYLE:
            color, alpha, size = STYLE[label]
        elif label.startswith("horizontal_other"):
            color, alpha, size = (0.65, 0.65, 0.18), 0.6, 5
        elif label.startswith("wall_"):
            idx = int(label.split("_")[1]) - 1
            color = TAB10[idx % len(TAB10)]
            alpha, size = WALL_STYLE_BASE
        else:
            color, alpha, size = (0.5, 0.5, 0.5), 0.3, 2

        planes.append((label, pts, color, alpha, size))
        print(f"  Loaded {label}: {len(pts)} pts")
    return planes


def draw_scatter(ax, planes, x_col, y_col, highlight_category=None):
    """Draw a scatter panel. If highlight_category is set, only that category
    gets its full style; everything else is drawn as faded gray."""
    # Draw remaining/background first, then foreground
    for label, pts, color, alpha, size in planes:
        if highlight_category:
            if _matches_category(label, highlight_category):
                ax.scatter(pts[:, x_col], pts[:, y_col],
                           c=[color], s=size, alpha=alpha, linewidths=0, zorder=3)
            else:
                ax.scatter(pts[:, x_col], pts[:, y_col],
                           c=[(0.82, 0.82, 0.82)], s=0.5, alpha=0.06,
                           linewidths=0, zorder=1)
        else:
            z = 1 if label == "remaining" else 2
            ax.scatter(pts[:, x_col], pts[:, y_col],
                       c=[color], s=size, alpha=alpha, linewidths=0, zorder=z)


def _matches_category(label, category):
    if category == "wall":
        return label.startswith("wall_")
    if category == "floor":
        return label.startswith("floor_")
    return label == category


def main():
    print("Loading planes...")
    planes = load_planes()

    # Sort so remaining draws first (background), walls last (foreground)
    order = {"remaining": 0, "horizontal_other": 1, "floor": 2, "wall": 3}

    def sort_key(item):
        label = item[0]
        for prefix, rank in order.items():
            if label.startswith(prefix):
                return rank
        return 1
    planes.sort(key=sort_key)

    # Collect legend entries (skip remaining from legend)
    legend_entries = [(l, c, len(p)) for l, p, c, a, s in planes if l != "remaining"]

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.patch.set_facecolor("white")

    # ── Top row ──

    # Panel (0,0): 3D perspective
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    axes[0, 0].set_visible(False)  # hide the 2D axis created by subplots
    rng = np.random.default_rng(42)
    for label, pts, color, alpha, size in planes:
        n = len(pts)
        if label == "remaining":
            idx = rng.choice(n, min(1500, n), replace=False)
            ax3d.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                         c=[(0.82, 0.82, 0.82)], s=0.3, alpha=0.05, linewidths=0, zorder=1)
        else:
            sub = min(800, n)
            idx = rng.choice(n, sub, replace=False) if n > sub else np.arange(n)
            ax3d.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                         c=[color], s=size * 0.5, alpha=min(alpha, 0.8),
                         linewidths=0, zorder=2 if not label.startswith("wall") else 3)
    ax3d.set_xlabel("X", fontsize=8)
    ax3d.set_ylabel("Y", fontsize=8)
    ax3d.set_zlabel("Z", fontsize=8)
    ax3d.set_title("3D Perspective", fontsize=11, fontweight="bold")
    ax3d.view_init(elev=25, azim=-55)

    # Panel (0,1): Top-down X-Y (all planes)
    ax = axes[0, 1]
    draw_scatter(ax, planes, 0, 1)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_title("Top-Down (X-Y) — All Planes", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")

    # Panel (0,2): Top-down X-Y walls only highlighted
    ax = axes[0, 2]
    draw_scatter(ax, planes, 0, 1, highlight_category="wall")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_title("Top-Down (X-Y) — Walls Highlighted", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")

    # ── Bottom row ──

    # Panel (1,0): Side X-Z
    ax = axes[1, 0]
    draw_scatter(ax, planes, 0, 2)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Z (m)", fontsize=9)
    ax.set_title("Side View (X-Z)", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")

    # Panel (1,1): Side Y-Z
    ax = axes[1, 1]
    draw_scatter(ax, planes, 1, 2)
    ax.set_xlabel("Y (m)", fontsize=9)
    ax.set_ylabel("Z (m)", fontsize=9)
    ax.set_title("Side View (Y-Z)", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")

    # Panel (1,2): Side X-Z floors only highlighted + annotations
    ax = axes[1, 2]
    draw_scatter(ax, planes, 0, 2, highlight_category="floor")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Z (m)", fontsize=9)
    ax.set_title("Side View (X-Z) — Floors Highlighted", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")

    # Annotate floor heights
    x_bounds = ax.get_xlim()
    x_mid = (x_bounds[0] + x_bounds[1]) / 2

    ax.axhline(y=-0.100, color=(0.9, 0.1, 0.1), linewidth=1.2, linestyle="--", alpha=0.7, zorder=4)
    ax.annotate("floor_main  Z = -0.100 m",
                xy=(x_mid, -0.100), xytext=(x_mid + 0.8, -0.100 - 0.15),
                fontsize=9, fontweight="bold", color=(0.9, 0.1, 0.1),
                arrowprops=dict(arrowstyle="->", color=(0.9, 0.1, 0.1), lw=1.2),
                zorder=5)

    ax.axhline(y=-0.003, color=(1.0, 0.55, 0.0), linewidth=1.2, linestyle="--", alpha=0.7, zorder=4)
    ax.annotate("floor_secondary  Z = -0.003 m",
                xy=(x_mid, -0.003), xytext=(x_mid + 0.8, -0.003 + 0.18),
                fontsize=9, fontweight="bold", color=(0.85, 0.45, 0.0),
                arrowprops=dict(arrowstyle="->", color=(0.85, 0.45, 0.0), lw=1.2),
                zorder=5)

    # ── Shared legend ──
    legend_ax = axes[0, 1]
    for label, color, count in legend_entries:
        legend_ax.scatter([], [], c=[color], s=40, label=f"{label} ({count:,})")
    rem_count = sum(len(p) for l, p, c, a, s in planes if l == "remaining")
    legend_ax.scatter([], [], c=[(0.78, 0.78, 0.78)], s=20, alpha=0.4,
                      label=f"remaining ({rem_count:,})")
    legend_ax.legend(loc="upper right", fontsize=8,
                     markerscale=1.5, framealpha=0.9, edgecolor="gray")

    plt.suptitle("Sprint 4: Plane Segmentation — Doorway Transit Data",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(VIS_PATH), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {VIS_PATH}")


if __name__ == "__main__":
    main()
