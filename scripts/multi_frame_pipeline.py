#!/usr/bin/env python3
"""Sprint 8.7: Multi-frame averaging across doorway transit.

Runs the full pipeline (extract → preprocess → segment → dedup → gap
detect → refine) on N consecutive rosbag frames and aggregates detected
doorway widths.  Averaging over the transit reduces angle-dependent
measurement noise from any single viewpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# ── paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BAG_PATH = Path.home() / "autopass" / "bags" / "run_01"
GT_PATH = ROOT / "ground_truth.txt"
RESULTS_DIR = ROOT / "results"
FIG_PATH = ROOT / "figures" / "sprint8_7_multiframe.png"

# ── pipeline parameters (match existing scripts) ────────────────────
ROI_MIN, ROI_MAX = np.array([-5, -5, -0.5]), np.array([5, 5, 2.5])
VOXEL_SIZE = 0.03
SOR_NB, SOR_STD = 20, 2.0

HORIZ_NZ = 0.9; MAX_HORIZ = 4; PLANE_DIST = 0.05; MIN_PLANE_PTS = 150
MAX_WALLS = 4; WALL_DIST = 0.03; WALL_NZ = 0.2; WALL_ITERS = 5000; WALL_MIN = 100

BIN_SIZE = 0.05; MIN_DENS = 5; REOCC = 2; GAP_RANGE = (0.6, 2.0)
MERGE_DOT = 0.95; MERGE_PERP = 0.30; MERGE_OVERLAP = 0.50

POINTFIELD_DTYPES = {1:("B",1),2:("b",1),3:("H",2),4:("h",2),
                     5:("I",4),6:("i",4),7:("f",4),8:("d",8)}


# ── frame extraction ────────────────────────────────────────────────
def parse_pc2(msg):
    fields = {}
    for f in msg.fields:
        fmt, sz = POINTFIELD_DTYPES[f.datatype]
        fields[f.name] = (f.offset, fmt)
    ps = msg.point_step
    data = bytes(msg.data)
    n = msg.width * msg.height
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        b = i * ps
        for j, ax in enumerate(("x", "y", "z")):
            o, fmt = fields[ax]
            pts[i, j] = struct.unpack_from(fmt, data, b + o)[0]
    return pts[np.isfinite(pts).all(axis=1)]


def read_frames(indices):
    """Yield (frame_index, points_array) for each requested index."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    want = set(indices)
    with AnyReader([BAG_PATH], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic == "/point_cloud2"]
        for idx, (conn, _, raw) in enumerate(reader.messages(connections=conns)):
            if idx in want:
                msg = reader.deserialize(raw, conn.msgtype)
                yield idx, parse_pc2(msg)
                want.discard(idx)
                if not want:
                    break


# ── preprocessing ────────────────────────────────────────────────────
def preprocess(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(ROI_MIN, ROI_MAX))
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(SOR_NB, SOR_STD)
    return pcd


# ── segmentation (inlined from segment.py) ───────────────────────────
def _wall_ransac(pts, rng_seed):
    best_inl, best_mod = np.array([], dtype=int), None
    n = len(pts)
    if n < 3:
        return None, best_inl
    rng = np.random.default_rng(rng_seed)
    for _ in range(WALL_ITERS):
        ix = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[ix]
        nrm = np.cross(p1 - p0, p2 - p0)
        nl = np.linalg.norm(nrm)
        if nl < 1e-10: continue
        nrm /= nl
        if abs(nrm[2]) > WALL_NZ: continue
        d = -nrm @ p0
        inl = np.where(np.abs(pts @ nrm + d) < WALL_DIST)[0]
        if len(inl) > len(best_inl):
            best_inl, best_mod = inl, (nrm[0], nrm[1], nrm[2], d)
    if best_mod and len(best_inl) < WALL_MIN:
        return None, np.array([], dtype=int)
    return best_mod, best_inl


def segment_walls(pcd):
    """Return list of wall point arrays (Nx3)."""
    rem = o3d.geometry.PointCloud(pcd)
    # strip horizontal planes first
    for _ in range(MAX_HORIZ):
        if len(rem.points) < MIN_PLANE_PTS: break
        mod, inl = rem.segment_plane(PLANE_DIST, 3, 1000)
        if len(inl) < MIN_PLANE_PTS: break
        if abs(mod[2]) < HORIZ_NZ:
            rem = rem.select_by_index(inl, invert=True); continue
        rem = rem.select_by_index(inl, invert=True)

    pts = np.asarray(rem.points).copy()
    walls = []
    for i in range(MAX_WALLS):
        if len(pts) < WALL_MIN: break
        mod, inl = _wall_ransac(pts, 42 + i)
        if mod is None or len(inl) < WALL_MIN: break
        walls.append(pts[inl])
        mask = np.ones(len(pts), dtype=bool); mask[inl] = False
        pts = pts[mask]
    return walls


# ── dedup + gap detect + refine (from v4) ────────────────────────────
def _pca(xy):
    c = xy.mean(0)
    ev, evec = np.linalg.eigh(np.cov((xy - c).T))
    d = evec[:, np.argmax(ev)]; d /= np.linalg.norm(d)
    return c, d, (xy - c) @ d

def _perp(c1, d1, c2, d2):
    a = d1 + d2; a /= np.linalg.norm(a)
    p = np.array([-a[1], a[0]])
    return abs((c2 - c1) @ p)

def _ovlp(t1, t2):
    o = max(0, min(t1.max(),t2.max()) - max(t1.min(),t2.min()))
    s = min(np.ptp(t1), np.ptp(t2))
    return o / s if s > 0 else 0

def detect_width(wall_arrays):
    """Run dedup → gap detect → refine, return list of refined widths (m)."""
    raws = []
    for pts3d in wall_arrays:
        xy = pts3d[:, :2].copy()
        c, d, t = _pca(xy)
        raws.append(dict(xy=xy, c=c, d=d, t=t))

    n = len(raws)
    par = list(range(n))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i+1, n):
            if abs(raws[i]["d"]@raws[j]["d"]) < MERGE_DOT: continue
            if _perp(raws[i]["c"],raws[i]["d"],raws[j]["c"],raws[j]["d"]) > MERGE_PERP: continue
            ad = raws[i]["d"]+raws[j]["d"]; ad /= np.linalg.norm(ad)
            if _ovlp((raws[i]["xy"]-raws[i]["c"])@ad,
                     (raws[j]["xy"]-raws[j]["c"])@ad) >= MERGE_OVERLAP:
                ri, rj = find(i), find(j)
                if ri != rj: par[rj] = ri

    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)

    widths = []
    for members in groups.values():
        xy = np.vstack([raws[m]["xy"] for m in members])
        c, d, t = _pca(xy)
        # gap detection
        edges = np.arange(t.min(), t.max() + BIN_SIZE, BIN_SIZE)
        counts, edges = np.histogram(t, bins=edges)
        occ = counts >= MIN_DENS
        nn = len(occ); i = 0
        while i < nn:
            if occ[i]: i += 1; continue
            gs = i; i += 1
            while i < nn:
                if not occ[i]: i += 1; continue
                run, k = 0, i
                while k < nn and occ[k]: run += 1; k += 1
                if run >= REOCC: break
                i = k
            ge = i
            if gs > 0 and ge < nn:
                ctrs = (edges[:-1]+edges[1:])/2
                gs_t = ctrs[gs] - BIN_SIZE/2
                ge_t = ctrs[ge-1] + BIN_SIZE/2
                w_bin = ge_t - gs_t
                if GAP_RANGE[0] <= w_bin <= GAP_RANGE[1]:
                    # refine
                    mask_l = (t >= gs_t - BIN_SIZE) & (t <= gs_t + BIN_SIZE)
                    mask_r = (t >= ge_t - BIN_SIZE) & (t <= ge_t + BIN_SIZE)
                    rl = float(t[mask_l].max()) if mask_l.any() else gs_t
                    rr = float(t[mask_r].min()) if mask_r.any() else ge_t
                    widths.append(rr - rl)
    return widths


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-start", type=int, default=20)
    ap.add_argument("--frame-end", type=int, default=50)
    ap.add_argument("--frame-step", type=int, default=1)
    args = ap.parse_args()

    indices = list(range(args.frame_start, args.frame_end + 1, args.frame_step))
    print(f"Sprint 8.7: multi-frame pipeline — {len(indices)} frames "
          f"[{args.frame_start}..{args.frame_end} step {args.frame_step}]")

    per_frame = []
    t_start = time.time()
    for fidx, pts in read_frames(indices):
        pcd = preprocess(pts)
        walls = segment_walls(pcd)
        ws = detect_width(walls)
        width_cm = round(ws[0] * 100, 2) if ws else None
        status = f"{width_cm:.1f} cm" if width_cm else "no detection"
        print(f"  frame {fidx:3d}: {len(pts):5d} pts, {len(walls)} walls → {status}")
        per_frame.append({"frame": fidx, "width_cm": width_cm})

    elapsed = time.time() - t_start
    detected = [r["width_cm"] for r in per_frame if r["width_cm"] is not None]
    n_det, n_proc = len(detected), len(per_frame)
    arr = np.array(detected) if detected else np.array([])

    agg = dict(
        mean_width_cm=round(float(arr.mean()), 2) if len(arr) else None,
        median_width_cm=round(float(np.median(arr)), 2) if len(arr) else None,
        std_width_cm=round(float(arr.std()), 2) if len(arr) else None,
        min_cm=round(float(arr.min()), 2) if len(arr) else None,
        max_cm=round(float(arr.max()), 2) if len(arr) else None,
        iqr_cm=round(float(np.percentile(arr,75)-np.percentile(arr,25)), 2) if len(arr) else None,
        frames_with_detection=n_det, frames_processed=n_proc,
        per_frame_detection_rate=round(n_det/n_proc, 2) if n_proc else 0,
        elapsed_s=round(elapsed, 1),
    )

    # Load GT clear width for comparison
    gt_clear = None
    if GT_PATH.exists():
        for line in GT_PATH.read_text().splitlines():
            if line.startswith("#") or not line.strip(): continue
            parts = [p.strip() for p in line.split(",", 3)]
            if parts[0] == "door_01" and parts[1]:
                gt_clear = float(parts[1]); break

    # Console summary
    print(f"\n{'='*50}")
    print(f"  Frames processed:   {n_proc}")
    print(f"  Frames w/ detection:{n_det} ({agg['per_frame_detection_rate']:.0%})")
    print(f"  Mean width:         {agg['mean_width_cm']} cm")
    print(f"  Median width:       {agg['median_width_cm']} cm")
    print(f"  Std:                {agg['std_width_cm']} cm")
    print(f"  Range:              [{agg['min_cm']}, {agg['max_cm']}] cm")
    print(f"  IQR:                {agg['iqr_cm']} cm")
    if gt_clear and agg["mean_width_cm"]:
        err_mean = round(agg["mean_width_cm"] - gt_clear, 2)
        err_med = round(agg["median_width_cm"] - gt_clear, 2)
        err_single = round(74.2 - gt_clear, 2)  # Sprint 8.5 single-frame
        print(f"\n  GT clear width:     {gt_clear} cm")
        print(f"  Error (mean):       {err_mean:+.2f} cm  (MAE {abs(err_mean):.2f})")
        print(f"  Error (median):     {err_med:+.2f} cm  (MAE {abs(err_med):.2f})")
        print(f"  Single-frame error: {err_single:+.2f} cm  (MAE {abs(err_single):.2f})")
    print(f"  Runtime:            {elapsed:.1f}s")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "multi_frame_run01.json", "w") as f:
        json.dump({"per_frame": per_frame, "aggregate": agg}, f, indent=2)
    with open(RESULTS_DIR / "multi_frame_run01.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "width_cm"])
        w.writeheader(); w.writerows(per_frame)

    # Also produce evaluation JSON for the mean width
    if gt_clear and agg["mean_width_cm"]:
        import math
        det_cm = agg["mean_width_cm"]
        se_c = round(det_cm - gt_clear, 2)
        ae_c = abs(se_c)
        gt_rough = 91.4
        se_r = round(det_cm - gt_rough, 2)
        eval_out = {"per_feature": [{
            "feature_id": "door_01", "detected_cm": det_cm,
            "clear_width_cm": gt_clear, "signed_err_clear": se_c,
            "abs_err_clear": ae_c, "rel_err_clear": round(ae_c/gt_clear*100, 2),
            "rough_width_cm": gt_rough, "signed_err_rough": se_r,
            "abs_err_rough": abs(se_r), "method": "multi_frame_mean",
        }], "aggregate": {
            "MAE_vs_clear_cm": ae_c, "MAE_vs_rough_cm": abs(se_r),
            "n_frames": n_det,
        }}
        with open(RESULTS_DIR / "evaluation_run01_multiframe.json", "w") as f:
            json.dump(eval_out, f, indent=2)

    # ── plot ──
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) histogram
    ax = axes[0]
    if len(arr):
        ax.hist(arr, bins=15, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.axvline(arr.mean(), color="red", ls="-", lw=2, label=f"mean={arr.mean():.1f}")
        ax.axvline(np.median(arr), color="orange", ls="--", lw=2,
                   label=f"median={np.median(arr):.1f}")
        if gt_clear:
            ax.axvline(gt_clear, color="green", ls=":", lw=2, label=f"GT clear={gt_clear}")
    ax.set_xlabel("Width (cm)"); ax.set_ylabel("Count")
    ax.set_title("Width Distribution"); ax.legend(fontsize=8)

    # (b) width vs frame index
    ax = axes[1]
    frames_det = [r["frame"] for r in per_frame if r["width_cm"] is not None]
    widths_det = [r["width_cm"] for r in per_frame if r["width_cm"] is not None]
    ax.plot(frames_det, widths_det, "o-", ms=4, color="#1f77b4")
    if gt_clear:
        ax.axhline(gt_clear, color="green", ls=":", lw=2, label=f"GT clear={gt_clear}")
    if agg["mean_width_cm"]:
        ax.axhline(agg["mean_width_cm"], color="red", ls="-", lw=1, alpha=0.5, label="mean")
    ax.set_xlabel("Frame Index"); ax.set_ylabel("Width (cm)")
    ax.set_title("Width vs Frame"); ax.legend(fontsize=8)

    # (c) box plot
    ax = axes[2]
    if len(arr):
        bp = ax.boxplot(arr, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.5)
        if gt_clear:
            ax.axhline(gt_clear, color="green", ls=":", lw=2, label=f"GT clear={gt_clear}")
    ax.set_ylabel("Width (cm)"); ax.set_title("Box Plot"); ax.legend(fontsize=8)

    plt.suptitle(f"Sprint 8.7: Multi-Frame Averaging — {n_det}/{n_proc} frames | "
                 f"{np.datetime64('today')}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(FIG_PATH), dpi=150, facecolor="white"); plt.close()
    print(f"\nVisualization: {FIG_PATH}")


if __name__ == "__main__":
    main()
