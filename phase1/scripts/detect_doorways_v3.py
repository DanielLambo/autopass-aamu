#!/usr/bin/env python3
"""Sprint 6.6: Wall deduplication and noise-tolerant gap detection.

Improvements over v2 (Sprint 6.5):
    1. Wall deduplication — RANSAC often produces duplicate planes for the
       same physical wall (perp distances ~10-28 cm).  Before gap detection
       we merge walls whose PCA lines are parallel, close, and overlapping.
       This halves 4 raw walls down to 2 physical walls.
    2. Noise-tolerant gap walk — a single occupied bin inside a true gap
       (a density spike from a door frame edge or noise) caused v2 to
       split one doorway into two narrow false positives.  We now require
       MIN_REOCCUPANCY_BINS consecutive occupied bins to close a gap.
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
OUTPUT_JSON = ROOT / "data" / "doorways_v3.json"
OUTPUT_FIG = ROOT / "figures" / "sprint6_6_doorways.png"

BIN_SIZE = 0.05
MIN_DENSITY = 5
MIN_REOCCUPANCY_BINS = 2   # consecutive occupied bins to end a gap
GAP_WIDTH_RANGE = (0.6, 2.0)
MERGE_DOT = 0.95
MERGE_PERP = 0.30
MERGE_OVERLAP_FRAC = 0.50


# ── helpers ──────────────────────────────────────────────────────────

def _pca_fit(xy):
    centroid = xy.mean(axis=0)
    cov = np.cov((xy - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    d = eigvecs[:, np.argmax(eigvals)]
    d /= np.linalg.norm(d)
    t = (xy - centroid) @ d
    return centroid, d, t


def _perp_dist(c1, d1, c2, d2):
    avg = d1 + d2
    avg /= np.linalg.norm(avg)
    perp = np.array([-avg[1], avg[0]])
    return abs((c2 - c1) @ perp)


def _overlap_frac(t1, t2):
    """Fraction of the shorter wall's extent that overlaps the longer."""
    lo1, hi1 = t1.min(), t1.max()
    lo2, hi2 = t2.min(), t2.max()
    overlap = max(0, min(hi1, hi2) - max(lo1, lo2))
    shorter = min(hi1 - lo1, hi2 - lo2)
    return overlap / shorter if shorter > 0 else 0


# ── Step 0: load and merge ───────────────────────────────────────────

def load_and_merge():
    raw = []
    for p in sorted(PLANES_DIR.glob("wall_*.pcd")):
        pts3d = np.asarray(o3d.io.read_point_cloud(str(p)).points)
        xy = pts3d[:, :2].copy()
        centroid, direction, t = _pca_fit(xy)
        raw.append(dict(name=p.stem, pts3d=pts3d, xy=xy,
                        centroid=centroid, direction=direction, t=t))

    n = len(raw)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = raw[i], raw[j]
            dot = abs(wi["direction"] @ wj["direction"])
            if dot < MERGE_DOT:
                continue
            pd = _perp_dist(wi["centroid"], wi["direction"],
                            wj["centroid"], wj["direction"])
            if pd > MERGE_PERP:
                continue
            avg_d = wi["direction"] + wj["direction"]
            avg_d /= np.linalg.norm(avg_d)
            t_i = (wi["xy"] - wi["centroid"]) @ avg_d
            t_j = (wj["xy"] - wj["centroid"]) @ avg_d
            if _overlap_frac(t_i, t_j) >= MERGE_OVERLAP_FRAC:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        names = [raw[m]["name"] for m in members]
        pts3d = np.vstack([raw[m]["pts3d"] for m in members])
        xy = pts3d[:, :2].copy()
        centroid, direction, t = _pca_fit(xy)
        label = "+".join(names)
        merged.append(dict(name=label, merged_from=names, pts3d=pts3d,
                           xy=xy, centroid=centroid, direction=direction, t=t))
    return raw, merged


# ── gap detection with reoccupancy filter ────────────────────────────

def find_gaps(t):
    edges = np.arange(t.min(), t.max() + BIN_SIZE, BIN_SIZE)
    counts, edges = np.histogram(t, bins=edges)
    occupied = counts >= MIN_DENSITY
    median_dens = float(np.median(counts[occupied])) if occupied.any() else 1.0

    gaps = []
    n = len(occupied)
    i = 0
    while i < n:
        if occupied[i]:
            i += 1
            continue
        # start of a potential gap
        gap_start = i
        i += 1
        while i < n:
            if not occupied[i]:
                i += 1
                continue
            # hit an occupied bin — check for sustained reoccupancy
            run = 0
            k = i
            while k < n and occupied[k]:
                run += 1
                k += 1
            if run >= MIN_REOCCUPANCY_BINS:
                break  # gap truly ends at i
            i = k  # skip isolated spike, continue gap
        gap_end = i  # first occupied bin after gap (or n)
        if gap_start > 0 and gap_end < n:
            centers = (edges[:-1] + edges[1:]) / 2
            gs_t = centers[gap_start] - BIN_SIZE / 2
            ge_t = centers[gap_end - 1] + BIN_SIZE / 2
            width = ge_t - gs_t
            if GAP_WIDTH_RANGE[0] <= width <= GAP_WIDTH_RANGE[1]:
                left_d = float(counts[gap_start - 1])
                right_d = float(counts[gap_end])
                gaps.append((gs_t, ge_t, left_d, right_d, median_dens))
    return gaps, edges, counts


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sprint 6.6: Wall Dedup + Noise-Tolerant Gap Detection")
    print("=" * 60)

    raw, walls = load_and_merge()
    print(f"\n  {len(raw)} walls -> {len(walls)} merged walls")
    for w in walls:
        print(f"    {w['name']}: {len(w['pts3d'])} pts, "
              f"merged from {w['merged_from']}")

    doorways = []
    wall_hists = []

    for w in walls:
        gaps, edges, counts = find_gaps(w["t"])
        regions = []
        print(f"\n  {w['name']}: {len(w['pts3d'])} pts, "
              f"span={w['t'].ptp():.2f} m")
        for gs, ge, ld, rd, md in gaps:
            width = ge - gs
            conf = round(min(1.0, min(ld, rd) / md), 2)
            c, d = w["centroid"], w["direction"]
            doorways.append(dict(
                source_wall=w["name"], merged_from=w["merged_from"],
                gap_start_xy=(c + gs * d).round(3).tolist(),
                gap_end_xy=(c + ge * d).round(3).tolist(),
                gap_center_xy=(c + (gs + ge) / 2 * d).round(3).tolist(),
                width_m=round(width, 3),
                flanking_density_left=ld, flanking_density_right=rd,
                confidence=conf))
            regions.append((gs, ge))
            print(f"    GAP: {width*100:.0f} cm, conf={conf:.2f}, "
                  f"flanking L={ld:.0f} R={rd:.0f}")
        wall_hists.append((w["name"], edges, counts, regions, len(w["pts3d"])))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(doorways, f, indent=2)
    print(f"\nDoorways written to: {OUTPUT_JSON}")

    print(f"\n{'#':>3}  {'Wall':>20}  {'Width':>8}  {'Conf':>5}")
    print("-" * 45)
    for i, d in enumerate(doorways, 1):
        print(f"{i:3d}  {d['source_wall']:>20}  "
              f"{d['width_m']*100:6.1f} cm  {d['confidence']:5.2f}")
    if not doorways:
        print("  (no doorways detected)")

    # ── plot ──
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    clrs = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
            "#d62728", "#8c564b", "#e377c2", "#7f7f7f"]

    ax = axes[0]
    for i, w in enumerate(walls):
        ax.scatter(w["xy"][:, 0], w["xy"][:, 1], c=clrs[i % len(clrs)],
                   s=1, alpha=0.6, label=f"{w['name']} ({len(w['pts3d'])} pts)")
    for d in doorways:
        sx, sy = d["gap_start_xy"]
        ex, ey = d["gap_end_xy"]
        cx, cy = d["gap_center_xy"]
        ax.plot([sx, ex], [sy, ey], "r--", linewidth=3, zorder=5)
        ax.annotate(f"{d['width_m']*100:.0f} cm", (cx, cy),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold", color="red", zorder=6)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Top-Down — {len(doorways)} doorway(s) | "
                 f"{len(raw)} raw walls -> {len(walls)} merged")

    ranked = sorted(wall_hists, key=lambda x: x[4], reverse=True)
    for p in range(2):
        ax = axes[1 + p]
        if p >= len(ranked):
            ax.set_visible(False); continue
        wn, edg, cnt, regs, npts = ranked[p]
        ctrs = (edg[:-1] + edg[1:]) / 2
        ax.bar(ctrs, cnt, width=BIN_SIZE * 0.9, color="#1f77b4",
               edgecolor="none", alpha=0.7)
        ax.axhline(MIN_DENSITY, color="gray", ls=":", lw=1,
                   label=f"min density = {MIN_DENSITY}")
        for gs, ge in regs:
            ax.axvspan(gs, ge, color="red", alpha=0.2, label="gap")
        ax.set_xlabel("Position along wall (m)"); ax.set_ylabel("Pts / bin")
        ax.set_title(f"{wn} ({npts} pts, bin={BIN_SIZE*100:.0f} cm)")
        h, l = ax.get_legend_handles_labels()
        seen = set()
        u = [(a, b) for a, b in zip(h, l) if b not in seen and not seen.add(b)]
        if u: ax.legend(*zip(*u), fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sprint 6.6: Dedup + Noise-Tolerant Gap Detection | "
                 f"{np.datetime64('today')}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_FIG), dpi=150, facecolor="white")
    plt.close()
    print(f"\nVisualization saved: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
