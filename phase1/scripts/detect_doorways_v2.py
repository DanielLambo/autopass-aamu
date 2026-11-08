#!/usr/bin/env python3
"""Sprint 6.5: Density-based doorway detection on fitted wall planes.

Why density, not segment topology?
    RANSAC plane segmentation fits a single continuous plane through all
    coplanar points — including those on both sides of a doorway opening.
    The doorway itself has no LiDAR returns (it is air), but the plane
    model spans the gap.  As a result, Sprint 6's inter-segment gap
    detection found 0 doorways: there are no separate wall segments to
    measure a gap between.

    This script instead projects each wall's points onto the wall's
    principal axis (via PCA) and builds a 1D density histogram.  A
    doorway appears as a run of low-density bins flanked by high-density
    wall regions.  This approach detects gaps *within* a single RANSAC
    plane, which is exactly where indoor doorways hide.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
PLANES_DIR = ROOT / "data" / "planes"
OUTPUT_JSON = ROOT / "data" / "doorways_v2.json"
OUTPUT_FIG = ROOT / "figures" / "sprint6_5_doorways.png"

BIN_SIZE = 0.05        # 5 cm bins
MIN_DENSITY = 5        # points per bin to count as "occupied"
GAP_WIDTH_RANGE = (0.6, 2.0)  # meters


def load_wall(path: Path):
    """Return name, 3D points, 2D XY projection, PCA direction, and t array."""
    pcd = o3d.io.read_point_cloud(str(path))
    pts3d = np.asarray(pcd.points)
    xy = pts3d[:, :2].copy()
    centroid = xy.mean(axis=0)
    cov = np.cov((xy - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    direction /= np.linalg.norm(direction)
    t = (xy - centroid) @ direction
    return path.stem, pts3d, xy, centroid, direction, t


def find_gaps(t: np.ndarray):
    """Histogram t values, return list of (gap_start_t, gap_end_t, left_density, right_density, median_density)."""
    t_min, t_max = t.min(), t.max()
    edges = np.arange(t_min, t_max + BIN_SIZE, BIN_SIZE)
    counts, edges = np.histogram(t, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    occupied = counts >= MIN_DENSITY
    median_dens = float(np.median(counts[occupied])) if occupied.any() else 1.0

    gaps = []
    n = len(occupied)
    i = 0
    while i < n:
        if not occupied[i]:
            j = i
            while j < n and not occupied[j]:
                j += 1
            # run of empty bins from i to j-1, need flanking occupied on both sides
            if i > 0 and j < n:
                width = (j - i) * BIN_SIZE
                if GAP_WIDTH_RANGE[0] <= width <= GAP_WIDTH_RANGE[1]:
                    left_dens = float(counts[i - 1])
                    right_dens = float(counts[j])
                    gap_start_t = centers[i] - BIN_SIZE / 2
                    gap_end_t = centers[j - 1] + BIN_SIZE / 2
                    gaps.append((gap_start_t, gap_end_t, left_dens, right_dens, median_dens))
            i = j
        else:
            i += 1
    return gaps, edges, counts


def main():
    print("=" * 60)
    print("Sprint 6.5: Density-Based Doorway Detection")
    print("=" * 60)

    wall_paths = sorted(PLANES_DIR.glob("wall_*.pcd"))
    if not wall_paths:
        print("ERROR: No wall PCDs found")
        return

    all_walls = []
    doorways = []
    wall_histograms = []  # (name, edges, counts, gap_regions)

    for path in wall_paths:
        name, pts3d, xy, centroid, direction, t = load_wall(path)
        all_walls.append((name, pts3d, xy, centroid, direction, t))
        gaps, edges, counts = find_gaps(t)
        gap_regions = []

        print(f"\n  {name}: {len(pts3d)} pts, span={t.ptp():.2f} m, "
              f"dir=({direction[0]:.3f}, {direction[1]:.3f})")

        for gs, ge, ld, rd, md in gaps:
            width = ge - gs
            conf = round(min(1.0, min(ld, rd) / md), 2)
            start_xy = (centroid + gs * direction).round(3).tolist()
            end_xy = (centroid + ge * direction).round(3).tolist()
            center_xy = (centroid + (gs + ge) / 2 * direction).round(3).tolist()
            doorways.append(dict(
                source_wall=name, gap_start_xy=start_xy, gap_end_xy=end_xy,
                gap_center_xy=center_xy, width_m=round(width, 3),
                flanking_density_left=ld, flanking_density_right=rd,
                confidence=conf,
            ))
            gap_regions.append((gs, ge))
            print(f"    GAP: {width*100:.0f} cm, confidence={conf:.2f}, "
                  f"flanking L={ld:.0f} R={rd:.0f}")

        wall_histograms.append((name, edges, counts, gap_regions, len(pts3d)))

    # Save JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(doorways, f, indent=2)
    print(f"\nDoorways written to: {OUTPUT_JSON}")

    # Console summary
    print(f"\n{'#':>3}  {'Wall':>10}  {'Width':>8}  {'Conf':>5}")
    print("-" * 35)
    for i, d in enumerate(doorways, 1):
        print(f"{i:3d}  {d['source_wall']:>10}  {d['width_m']*100:6.1f} cm  "
              f"{d['confidence']:5.2f}")
    if not doorways:
        print("  (no doorways detected)")

    # --- Plot ---
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Top: top-down view
    ax = axes[0]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
              "#d62728", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, (name, _, xy, centroid, direction, t) in enumerate(all_walls):
        c = colors[i % len(colors)]
        ax.scatter(xy[:, 0], xy[:, 1], c=c, s=1, alpha=0.6,
                   label=f"{name} ({len(xy)} pts)")
    for d in doorways:
        sx, sy = d["gap_start_xy"]
        ex, ey = d["gap_end_xy"]
        cx, cy = d["gap_center_xy"]
        ax.plot([sx, ex], [sy, ey], "r--", linewidth=3, zorder=5)
        ax.annotate(f"{d['width_m']*100:.0f} cm", (cx, cy),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold", color="red", zorder=6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Top-Down View — {len(doorways)} doorway(s) detected")
    ax.grid(True, alpha=0.3)

    # Sort walls by point count descending for histogram panels
    ranked = sorted(wall_histograms, key=lambda x: x[4], reverse=True)

    for panel_idx in range(2):
        ax = axes[1 + panel_idx]
        if panel_idx >= len(ranked):
            ax.set_visible(False)
            continue
        wname, edges, counts, gap_regions, npts = ranked[panel_idx]
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=BIN_SIZE * 0.9, color="#1f77b4",
               edgecolor="none", alpha=0.7)
        ax.axhline(MIN_DENSITY, color="gray", linestyle=":", linewidth=1,
                   label=f"min density = {MIN_DENSITY}")
        for gs, ge in gap_regions:
            ax.axvspan(gs, ge, color="red", alpha=0.2, label="gap")
        ax.set_xlabel("Position along wall (m)")
        ax.set_ylabel("Points per bin")
        ax.set_title(f"Density histogram — {wname} ({npts} pts, "
                     f"bin = {BIN_SIZE*100:.0f} cm)")
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels)
                  if l not in seen and not seen.add(l)]
        ax.legend(*zip(*unique), fontsize=8) if unique else None
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sprint 6.5: Density-Based Doorway Detection | "
                 f"{np.datetime64('today')}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_FIG), dpi=150, facecolor="white")
    plt.close()
    print(f"\nVisualization saved: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
