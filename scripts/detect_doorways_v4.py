#!/usr/bin/env python3
"""Sprint 8.5: Sub-bin edge refinement to correct quantization bias.

Builds on Sprint 6.6 (wall dedup + noise-tolerant gap detection).  After
the histogram-based gap detector finds a gap, we refine both edges by
locating the actual outermost wall points near each bin boundary.  The
5 cm bin grid systematically eats into gap edges; refinement recovers
the true gap width from the raw point coordinates.
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
OUTPUT_JSON = ROOT / "data" / "doorways_v4.json"
OUTPUT_FIG = ROOT / "figures" / "sprint8_5_doorways.png"

BIN_SIZE = 0.05
MIN_DENSITY = 5
MIN_REOCCUPANCY_BINS = 2
GAP_WIDTH_RANGE = (0.6, 2.0)
MERGE_DOT = 0.95
MERGE_PERP = 0.30
MERGE_OVERLAP_FRAC = 0.50


def _pca_fit(xy):
    centroid = xy.mean(axis=0)
    cov = np.cov((xy - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    d = eigvecs[:, np.argmax(eigvals)]
    d /= np.linalg.norm(d)
    t = (xy - centroid) @ d
    return centroid, d, t


def _perp_dist(c1, d1, c2, d2):
    avg = d1 + d2; avg /= np.linalg.norm(avg)
    perp = np.array([-avg[1], avg[0]])
    return abs((c2 - c1) @ perp)


def _overlap_frac(t1, t2):
    lo1, hi1 = t1.min(), t1.max()
    lo2, hi2 = t2.min(), t2.max()
    overlap = max(0, min(hi1, hi2) - max(lo1, lo2))
    shorter = min(hi1 - lo1, hi2 - lo2)
    return overlap / shorter if shorter > 0 else 0


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
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = raw[i], raw[j]
            if abs(wi["direction"] @ wj["direction"]) < MERGE_DOT:
                continue
            if _perp_dist(wi["centroid"], wi["direction"],
                          wj["centroid"], wj["direction"]) > MERGE_PERP:
                continue
            avg_d = wi["direction"] + wj["direction"]
            avg_d /= np.linalg.norm(avg_d)
            if _overlap_frac((wi["xy"] - wi["centroid"]) @ avg_d,
                             (wj["xy"] - wj["centroid"]) @ avg_d) >= MERGE_OVERLAP_FRAC:
                ri, rj = find(i), find(j)
                if ri != rj: parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        names = [raw[m]["name"] for m in members]
        pts3d = np.vstack([raw[m]["pts3d"] for m in members])
        xy = pts3d[:, :2].copy()
        centroid, direction, t = _pca_fit(xy)
        merged.append(dict(name="+".join(names), merged_from=names,
                           pts3d=pts3d, xy=xy, centroid=centroid,
                           direction=direction, t=t))
    return raw, merged


def find_gaps(t):
    edges = np.arange(t.min(), t.max() + BIN_SIZE, BIN_SIZE)
    counts, edges = np.histogram(t, bins=edges)
    occupied = counts >= MIN_DENSITY
    med = float(np.median(counts[occupied])) if occupied.any() else 1.0
    centers = (edges[:-1] + edges[1:]) / 2

    gaps = []
    n = len(occupied); i = 0
    while i < n:
        if occupied[i]: i += 1; continue
        gs = i; i += 1
        while i < n:
            if not occupied[i]: i += 1; continue
            run, k = 0, i
            while k < n and occupied[k]: run += 1; k += 1
            if run >= MIN_REOCCUPANCY_BINS: break
            i = k
        ge = i
        if gs > 0 and ge < n:
            gs_t = centers[gs] - BIN_SIZE / 2
            ge_t = centers[ge - 1] + BIN_SIZE / 2
            w = ge_t - gs_t
            if GAP_WIDTH_RANGE[0] <= w <= GAP_WIDTH_RANGE[1]:
                gaps.append((gs_t, ge_t, float(counts[gs-1]),
                             float(counts[ge]), med))
    return gaps, edges, counts


def refine_edge(t_arr, boundary_t, side):
    """Find the actual wall-point edge within +-1 bin of the boundary."""
    lo = boundary_t - BIN_SIZE
    hi = boundary_t + BIN_SIZE
    mask = (t_arr >= lo) & (t_arr <= hi)
    pts = t_arr[mask]
    if len(pts) == 0:
        return boundary_t
    return float(pts.max()) if side == "left" else float(pts.min())


def main():
    print("=" * 60)
    print("Sprint 8.5: Sub-Bin Edge Refinement")
    print("=" * 60)

    raw, walls = load_and_merge()
    print(f"\n  {len(raw)} walls -> {len(walls)} merged")

    doorways = []
    wall_hists = []

    for w in walls:
        gaps, edges, counts = find_gaps(w["t"])
        regions_binned, regions_refined = [], []
        c, d = w["centroid"], w["direction"]

        for gs, ge, ld, rd, md in gaps:
            ref_left = refine_edge(w["t"], gs, "left")
            ref_right = refine_edge(w["t"], ge, "right")
            w_binned = ge - gs
            w_refined = ref_right - ref_left
            conf = round(min(1.0, min(ld, rd) / md), 2)
            delta = (w_refined - w_binned) * 100

            doorways.append(dict(
                source_wall=w["name"], merged_from=w["merged_from"],
                gap_start_xy=(c + ref_left * d).round(3).tolist(),
                gap_end_xy=(c + ref_right * d).round(3).tolist(),
                gap_center_xy=(c + (ref_left + ref_right) / 2 * d).round(3).tolist(),
                width_m=round(w_refined, 4),
                width_m_binned=round(w_binned, 4),
                width_m_refined=round(w_refined, 4),
                refinement_delta_cm=round(delta, 2),
                flanking_density_left=ld, flanking_density_right=rd,
                confidence=conf))

            regions_binned.append((gs, ge))
            regions_refined.append((ref_left, ref_right))
            print(f"  {w['name']}: binned={w_binned*100:.0f} cm -> "
                  f"refined={w_refined*100:.1f} cm (delta={delta:+.1f} cm)")

        wall_hists.append((w["name"], edges, counts,
                           regions_binned, regions_refined, len(w["pts3d"])))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(doorways, f, indent=2)
    print(f"\nDoorways written to: {OUTPUT_JSON}")

    # ── plot ──
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    clrs = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    ax = axes[0]
    for i, w in enumerate(walls):
        ax.scatter(w["xy"][:, 0], w["xy"][:, 1], c=clrs[i % len(clrs)],
                   s=1, alpha=0.6, label=f"{w['name']} ({len(w['pts3d'])} pts)")
    for d in doorways:
        sx, sy = d["gap_start_xy"]; ex, ey = d["gap_end_xy"]
        cx, cy = d["gap_center_xy"]
        ax.plot([sx, ex], [sy, ey], "r-", linewidth=3, zorder=5)
        ax.annotate(f"binned: {d['width_m_binned']*100:.0f} cm | "
                    f"refined: {d['width_m_refined']*100:.1f} cm",
                    (cx, cy), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold", color="red", zorder=6)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Sprint 8.5: {len(doorways)} doorway(s) — sub-bin refined")

    ranked = sorted(wall_hists, key=lambda x: x[5], reverse=True)
    for p in range(2):
        ax = axes[1 + p]
        if p >= len(ranked): ax.set_visible(False); continue
        wn, edg, cnt, rb, rr, npts = ranked[p]
        ctrs = (edg[:-1] + edg[1:]) / 2
        ax.bar(ctrs, cnt, width=BIN_SIZE * 0.9, color="#1f77b4",
               edgecolor="none", alpha=0.7)
        ax.axhline(MIN_DENSITY, color="gray", ls=":", lw=1,
                   label=f"min density = {MIN_DENSITY}")
        for gs, ge in rb:
            ax.axvspan(gs, ge, color="red", alpha=0.15, label="binned gap")
        for rs, re in rr:
            ax.axvline(rs, color="darkred", ls="--", lw=1.2, label="refined edge")
            ax.axvline(re, color="darkred", ls="--", lw=1.2)
        ax.set_xlabel("Position along wall (m)"); ax.set_ylabel("Pts / bin")
        ax.set_title(f"{wn} ({npts} pts)")
        h, l = ax.get_legend_handles_labels()
        seen = set()
        u = [(a, b) for a, b in zip(h, l) if b not in seen and not seen.add(b)]
        if u: ax.legend(*zip(*u), fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sprint 8.5: Sub-Bin Edge Refinement | "
                 f"{np.datetime64('today')}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_FIG), dpi=150, facecolor="white")
    plt.close()
    print(f"Visualization saved: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
