#!/usr/bin/env python3
"""Two-phase RANSAC plane segmentation with orientation-based classification.

Phase 1: Standard iterative RANSAC to extract horizontal planes (floor, ceiling).
Phase 2: Constrained RANSAC on remaining points — only accepts planes with |nz| < 0.2
         to detect walls that would otherwise be starved by floor fragmentation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

INPUT_PATH = Path.home() / "autopass" / "data" / "frame_clean.pcd"
PLANES_DIR = Path.home() / "autopass" / "data" / "planes"
VIS_PATH = Path.home() / "autopass" / "data" / "segmentation_result.png"

# RANSAC parameters
MAX_HORIZ_PLANES = 4
MAX_WALL_PLANES = 4
DISTANCE_THRESHOLD = 0.05
RANSAC_N = 3
NUM_ITERATIONS = 1000
MIN_PLANE_POINTS = 150

# Constrained wall RANSAC
WALL_DIST_THRESHOLD = 0.03
WALL_MAX_NZ = 0.2
WALL_RANSAC_ITERS = 5000
WALL_MIN_POINTS = 100

# Classification thresholds
HORIZONTAL_NZ = 0.9
FLOOR_SECONDARY_DZ = 0.1
CEILING_DZ = 1.0

# Colors
LABEL_COLORS = {
    "floor_main": (0.9, 0.1, 0.1),
    "floor_secondary": (1.0, 0.5, 0.0),
    "ceiling": (0.6, 0.2, 0.8),
    "horizontal_other": (0.7, 0.7, 0.3),
    "remaining": (0.4, 0.4, 0.4),
}
WALL_COLORS = [
    (0.1, 0.4, 0.9),
    (0.1, 0.8, 0.4),
    (0.0, 0.7, 0.7),
    (0.3, 0.5, 0.9),
]
OTHER_COLOR = (0.5, 0.5, 0.5)


def extract_horizontal_planes(pcd):
    """Iterative RANSAC for horizontal planes. Accepts any plane found by standard RANSAC,
    then classifies by normal afterward."""
    planes = []
    remaining = o3d.geometry.PointCloud(pcd)

    for i in range(MAX_HORIZ_PLANES):
        n_pts = len(remaining.points)
        if n_pts < MIN_PLANE_POINTS:
            break

        model, inliers = remaining.segment_plane(
            distance_threshold=DISTANCE_THRESHOLD,
            ransac_n=RANSAC_N,
            num_iterations=NUM_ITERATIONS,
        )

        if len(inliers) < MIN_PLANE_POINTS:
            break

        a, b, c, d = model
        # Only keep if actually horizontal
        if abs(c) < HORIZONTAL_NZ:
            print(f"  H-Plane {i}: SKIP (|nz|={abs(c):.3f}, not horizontal)")
            # Remove these points — they'll be reconsidered in wall phase
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        plane_pcd = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)
        print(f"  H-Plane {i}: {len(inliers)} pts, normal=({a:.3f}, {b:.3f}, {c:.3f}), d={d:.3f}")
        planes.append((plane_pcd, model))

    return planes, remaining


def constrained_wall_ransac(pts_array, max_nz=WALL_MAX_NZ, dist_thresh=WALL_DIST_THRESHOLD,
                            n_iter=WALL_RANSAC_ITERS, min_inliers=WALL_MIN_POINTS):
    """RANSAC that only accepts planes with |nz| < max_nz (vertical walls)."""
    best_inliers = np.array([], dtype=int)
    best_model = None
    n = len(pts_array)
    if n < 3:
        return None, best_inliers
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts_array[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        if abs(normal[2]) > max_nz:
            continue

        d = -np.dot(normal, p0)
        dists = np.abs(pts_array @ normal + d)
        inlier_idx = np.where(dists < dist_thresh)[0]

        if len(inlier_idx) > len(best_inliers):
            best_inliers = inlier_idx
            best_model = (normal[0], normal[1], normal[2], d)

    if best_model is not None and len(best_inliers) < min_inliers:
        return None, np.array([], dtype=int)
    return best_model, best_inliers


def extract_wall_planes(remaining_pcd):
    """Run constrained RANSAC to find vertical wall planes."""
    walls = []
    pts = np.asarray(remaining_pcd.points).copy()

    for i in range(MAX_WALL_PLANES):
        if len(pts) < WALL_MIN_POINTS:
            break

        model, inliers = constrained_wall_ransac(pts)
        if model is None or len(inliers) < WALL_MIN_POINTS:
            print(f"  W-Plane {i}: no more walls found")
            break

        a, b, c, d = model
        wall_pts = pts[inliers]
        z_span = wall_pts[:, 2].max() - wall_pts[:, 2].min()
        print(f"  W-Plane {i}: {len(inliers)} pts, normal=({a:.3f}, {b:.3f}, {c:.3f}), "
              f"d={d:.3f}, z_span={z_span:.2f}m")

        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(wall_pts)
        walls.append((wall_pcd, model))

        # Remove inliers
        mask = np.ones(len(pts), dtype=bool)
        mask[inliers] = False
        pts = pts[mask]

    # Build remaining pcd from leftover points
    rem_pcd = o3d.geometry.PointCloud()
    rem_pcd.points = o3d.utility.Vector3dVector(pts)
    return walls, rem_pcd


def classify_horizontal(planes):
    """Classify horizontal planes by height."""
    classified = []
    items = []
    for plane_pcd, model in planes:
        mean_z = np.asarray(plane_pcd.points)[:, 2].mean()
        items.append((plane_pcd, model, mean_z))
    items.sort(key=lambda x: x[2])

    horiz_other_idx = 0
    if items:
        floor_z = items[0][2]
        for plane_pcd, model, mean_z in items:
            dz = mean_z - floor_z
            if dz < 1e-6:
                label = "floor_main"
            elif dz < FLOOR_SECONDARY_DZ:
                label = "floor_secondary"
            elif dz > CEILING_DZ:
                label = "ceiling"
            else:
                horiz_other_idx += 1
                label = f"horizontal_other_{horiz_other_idx:02d}"
            classified.append((plane_pcd, model, label))
    return classified


def get_color(label):
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    if label.startswith("wall_"):
        idx = int(label.split("_")[1]) - 1
        return WALL_COLORS[idx % len(WALL_COLORS)]
    if label.startswith("horizontal_other"):
        return LABEL_COLORS["horizontal_other"]
    return OTHER_COLOR


def save_planes(classified, remaining):
    PLANES_DIR.mkdir(parents=True, exist_ok=True)
    for f in PLANES_DIR.glob("*.pcd"):
        f.unlink()

    for plane_pcd, model, label in classified:
        path = PLANES_DIR / f"{label}.pcd"
        o3d.io.write_point_cloud(str(path), plane_pcd)
        print(f"  Saved {path.name} ({len(plane_pcd.points)} pts)")

    if len(remaining.points) > 0:
        path = PLANES_DIR / "remaining.pcd"
        o3d.io.write_point_cloud(str(path), remaining)
        print(f"  Saved remaining.pcd ({len(remaining.points)} pts)")


def plot_segmentation(classified, remaining, output_path):
    """4-panel figure: 3D perspective, top-down X-Y, side X-Z, side Y-Z."""
    all_pts = []
    all_colors = []
    legend_entries = []

    for plane_pcd, model, label in classified:
        pts = np.asarray(plane_pcd.points)
        color = get_color(label)
        all_pts.append(pts)
        all_colors.append(np.tile(color, (len(pts), 1)))
        legend_entries.append((label, color, len(pts)))

    if len(remaining.points) > 0:
        pts = np.asarray(remaining.points)
        color = LABEL_COLORS["remaining"]
        all_pts.append(pts)
        all_colors.append(np.tile(color, (len(pts), 1)))
        legend_entries.append(("remaining", color, len(pts)))

    all_pts = np.vstack(all_pts)
    all_colors = np.vstack(all_colors)

    fig = plt.figure(figsize=(18, 14))

    # Panel 1: 3D perspective (subsampled)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    n_total = len(all_pts)
    if n_total > 5000:
        idx = np.random.default_rng(42).choice(n_total, 5000, replace=False)
        ax1.scatter(all_pts[idx, 0], all_pts[idx, 1], all_pts[idx, 2],
                    c=all_colors[idx], s=0.5, alpha=0.7)
    else:
        ax1.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
                    c=all_colors, s=0.5, alpha=0.7)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Perspective (5k subsample)")

    # Panel 2: Top-down X-Y with legend
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(all_pts[:, 0], all_pts[:, 1], c=all_colors, s=0.3, alpha=0.6)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top-Down (X-Y)")
    ax2.set_aspect("equal")
    for label, color, count in legend_entries:
        ax2.scatter([], [], c=[color], s=30, label=f"{label} ({count})")
    ax2.legend(loc="upper right", fontsize=7, markerscale=2)

    # Panel 3: Side X-Z
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(all_pts[:, 0], all_pts[:, 2], c=all_colors, s=0.3, alpha=0.6)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("Side View (X-Z)")
    ax3.set_aspect("equal")

    # Panel 4: Side Y-Z
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(all_pts[:, 1], all_pts[:, 2], c=all_colors, s=0.3, alpha=0.6)
    ax4.set_xlabel("Y (m)")
    ax4.set_ylabel("Z (m)")
    ax4.set_title("Side View (Y-Z)")
    ax4.set_aspect("equal")

    plt.suptitle("Sprint 4: Plane Segmentation — Doorway Transit Data", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, facecolor="white")
    plt.close()
    print(f"Visualization saved to: {output_path}")


def main():
    print(f"Loading: {INPUT_PATH}")
    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    print(f"  Points: {len(pcd.points)}")

    # Phase 1: Extract horizontal planes (floor/ceiling)
    print("\n--- Phase 1: Horizontal plane extraction ---")
    horiz_planes, after_horiz = extract_horizontal_planes(pcd)
    print(f"  Found {len(horiz_planes)} horizontal planes, {len(after_horiz.points)} remaining")

    # Phase 2: Extract wall planes from remainder using constrained RANSAC
    print("\n--- Phase 2: Wall detection (constrained RANSAC, |nz| < 0.2) ---")
    wall_planes, remaining = extract_wall_planes(after_horiz)
    print(f"  Found {len(wall_planes)} walls, {len(remaining.points)} remaining")

    # Classify horizontal planes
    print("\n--- Classification ---")
    classified = classify_horizontal(horiz_planes)

    # Add walls
    for i, (wall_pcd, model) in enumerate(wall_planes):
        classified.append((wall_pcd, model, f"wall_{i+1:02d}"))

    for plane_pcd, model, label in classified:
        a, b, c, d = model
        mean_z = np.asarray(plane_pcd.points)[:, 2].mean()
        print(f"  {label:20s} | {len(plane_pcd.points):6d} pts | "
              f"normal=({a:+.3f}, {b:+.3f}, {c:+.3f}) | mean_z={mean_z:.3f} | d={d:+.3f}")
    print(f"  {'remaining':20s} | {len(remaining.points):6d} pts")

    print("\n--- Saving planes ---")
    save_planes(classified, remaining)

    print("\n--- Generating visualization ---")
    plot_segmentation(classified, remaining, VIS_PATH)

    # Final summary table
    total = sum(len(p.points) for p, _, _ in classified) + len(remaining.points)
    print("\n" + "=" * 80)
    print(f"{'Plane':20s} | {'Points':>8s} | {'Normal (a, b, c)':>22s} | {'Mean Z':>8s} | {'d':>8s}")
    print("-" * 80)
    for plane_pcd, model, label in classified:
        a, b, c, d = model
        mean_z = np.asarray(plane_pcd.points)[:, 2].mean()
        print(f"{label:20s} | {len(plane_pcd.points):8d} | "
              f"({a:+.3f}, {b:+.3f}, {c:+.3f}) | {mean_z:8.3f} | {d:+8.3f}")
    print(f"{'remaining':20s} | {len(remaining.points):8d} |")
    print("-" * 80)
    print(f"{'TOTAL':20s} | {total:8d} |")
    print("=" * 80)


if __name__ == "__main__":
    main()
