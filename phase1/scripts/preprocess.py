#!/usr/bin/env python3
"""Preprocess a raw point cloud: crop, downsample, remove outliers, and visualize."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

INPUT_PATH = Path.home() / "autopass" / "data" / "frame_raw.pcd"
OUTPUT_PATH = Path.home() / "autopass" / "data" / "frame_clean.pcd"
VIS_PATH = Path.home() / "autopass" / "data" / "preprocess_comparison.png"

# ROI bounding box (meters)
ROI_MIN = np.array([-5.0, -5.0, -0.5])
ROI_MAX = np.array([5.0, 5.0, 2.5])

VOXEL_SIZE = 0.03
SOR_NEIGHBORS = 20
SOR_STD_RATIO = 2.0


def crop_to_roi(pcd, roi_min, roi_max):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=roi_min, max_bound=roi_max)
    return pcd.crop(bbox)


def colorize_by_height(pcd):
    """Color points by Z height: red (low/floor) -> blue (high/ceiling)."""
    points = np.asarray(pcd.points)
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-6:
        norm = np.zeros_like(z)
    else:
        norm = (z - z_min) / (z_max - z_min)

    colors = np.zeros((len(points), 3))
    colors[:, 0] = 1.0 - norm  # Red channel: high at floor
    colors[:, 2] = norm         # Blue channel: high at ceiling
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def plot_comparison(raw_pcd, clean_pcd, output_path):
    """Create side-by-side matplotlib scatter plot."""
    raw_pts = np.asarray(raw_pcd.points)
    clean_pts = np.asarray(clean_pcd.points)
    clean_colors = np.asarray(clean_pcd.colors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Raw: top-down view (X vs Y), colored by Z
    ax = axes[0]
    sc = ax.scatter(raw_pts[:, 0], raw_pts[:, 1], c=raw_pts[:, 2], cmap="coolwarm",
                    s=0.3, alpha=0.6)
    ax.set_title(f"Raw ({len(raw_pts)} pts)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Z (m)")

    # Clean: top-down view with height coloring
    ax = axes[1]
    ax.scatter(clean_pts[:, 0], clean_pts[:, 1], c=clean_colors, s=0.5, alpha=0.7)
    ax.set_title(f"Cleaned ({len(clean_pts)} pts)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    plt.suptitle("Sprint 3: Preprocessing Pipeline — Raw vs Cleaned", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Visualization saved to: {output_path}")


def main():
    print(f"Loading: {INPUT_PATH}")
    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    raw_pcd = o3d.geometry.PointCloud(pcd)  # copy for visualization
    count_raw = len(pcd.points)
    print(f"  Loaded points: {count_raw}")

    # Stage 1: Crop to ROI
    pcd = crop_to_roi(pcd, ROI_MIN, ROI_MAX)
    count_crop = len(pcd.points)
    print(f"  After ROI crop: {count_crop} ({count_raw - count_crop} removed)")

    # Stage 2: Voxel downsample
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    count_voxel = len(pcd.points)
    print(f"  After voxel downsample (size={VOXEL_SIZE}): {count_voxel}")

    # Stage 3: Statistical outlier removal
    pcd, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NEIGHBORS, std_ratio=SOR_STD_RATIO
    )
    count_sor = len(pcd.points)
    print(f"  After outlier removal: {count_sor} ({count_voxel - count_sor} outliers removed)")

    # Colorize by height
    pcd = colorize_by_height(pcd)

    # Save
    o3d.io.write_point_cloud(str(OUTPUT_PATH), pcd)
    print(f"Saved cleaned cloud to: {OUTPUT_PATH}")

    # Visualize
    plot_comparison(raw_pcd, pcd, VIS_PATH)

    print("\n--- Summary ---")
    print(f"  Raw:              {count_raw}")
    print(f"  After crop:       {count_crop}")
    print(f"  After downsample: {count_voxel}")
    print(f"  After SOR:        {count_sor}")


if __name__ == "__main__":
    main()
