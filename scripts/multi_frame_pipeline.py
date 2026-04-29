#!/usr/bin/env python3
"""Sprint 8.8: Spatial gap clustering and outlier-rejected multi-frame estimation.

Runs the full pipeline on N rosbag frames, then clusters detected gaps by
2D proximity using DBSCAN.  The cluster with the highest (frame_count *
mean_confidence) is identified as the true doorway.  Outlier widths within
the winning cluster (>2 sigma from median) are rejected before computing
the final width estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from sklearn.cluster import DBSCAN

# ── paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BAG_PATH = Path.home() / "autopass" / "bags" / "run_01"
GT_PATH = ROOT / "ground_truth.txt"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# ── pipeline params ──────────────────────────────────────────────────
ROI_MIN, ROI_MAX = np.array([-5, -5, -0.5]), np.array([5, 5, 2.5])
VOXEL_SIZE = 0.03; SOR_NB, SOR_STD = 20, 2.0
HORIZ_NZ = 0.9; MAX_HORIZ = 4; PLANE_DIST = 0.05; MIN_PLANE_PTS = 150
MAX_WALLS = 4; WALL_DIST = 0.03; WALL_NZ = 0.2; WALL_ITERS = 5000; WALL_MIN = 100
BIN_SIZE = 0.05; MIN_DENS = 5; REOCC = 2; GAP_RANGE = (0.6, 2.0)
MERGE_DOT = 0.95; MERGE_PERP = 0.30; MERGE_OVERLAP = 0.50

DBSCAN_EPS = 0.30; DBSCAN_MIN = 3; SIGMA_REJECT = 2.0

POINTFIELD_DTYPES = {1:("B",1),2:("b",1),3:("H",2),4:("h",2),
                     5:("I",4),6:("i",4),7:("f",4),8:("d",8)}


# ── extraction ───────────────────────────────────────────────────────
def parse_pc2(msg):
    fields = {f.name: (f.offset, POINTFIELD_DTYPES[f.datatype][0]) for f in msg.fields}
    ps, data, n = msg.point_step, bytes(msg.data), msg.width * msg.height
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        b = i * ps
        for j, ax in enumerate(("x", "y", "z")):
            pts[i, j] = struct.unpack_from(fields[ax][1], data, b + fields[ax][0])[0]
    return pts[np.isfinite(pts).all(axis=1)]

def read_frames(indices):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    want = set(indices)
    with AnyReader([BAG_PATH], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic == "/point_cloud2"]
        for idx, (conn, _, raw) in enumerate(reader.messages(connections=conns)):
            if idx in want:
                yield idx, parse_pc2(reader.deserialize(raw, conn.msgtype))
                want.discard(idx)
                if not want: break


# ── preprocess + segment ─────────────────────────────────────────────
def preprocess(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(ROI_MIN, ROI_MAX))
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(SOR_NB, SOR_STD)
    return pcd

def _wall_ransac(pts, seed):
    best_inl, best_mod = np.array([], dtype=int), None
    n = len(pts)
    if n < 3: return None, best_inl
    rng = np.random.default_rng(seed)
    for _ in range(WALL_ITERS):
        ix = rng.choice(n, 3, replace=False)
        nrm = np.cross(pts[ix[1]]-pts[ix[0]], pts[ix[2]]-pts[ix[0]])
        nl = np.linalg.norm(nrm)
        if nl < 1e-10: continue
        nrm /= nl
        if abs(nrm[2]) > WALL_NZ: continue
        inl = np.where(np.abs(pts @ nrm + (-nrm @ pts[ix[0]])) < WALL_DIST)[0]
        if len(inl) > len(best_inl):
            best_inl, best_mod = inl, nrm
    if best_mod is not None and len(best_inl) < WALL_MIN: return None, np.array([],dtype=int)
    return best_mod, best_inl

def segment_walls(pcd):
    rem = o3d.geometry.PointCloud(pcd)
    for _ in range(MAX_HORIZ):
        if len(rem.points) < MIN_PLANE_PTS: break
        mod, inl = rem.segment_plane(PLANE_DIST, 3, 1000)
        if len(inl) < MIN_PLANE_PTS: break
        rem = rem.select_by_index(inl, invert=True)
    pts = np.asarray(rem.points).copy()
    walls = []
    for i in range(MAX_WALLS):
        if len(pts) < WALL_MIN: break
        mod, inl = _wall_ransac(pts, 42 + i)
        if mod is None or len(inl) < WALL_MIN: break
        walls.append(pts[inl])
        mask = np.ones(len(pts), dtype=bool); mask[inl] = False; pts = pts[mask]
    return walls


# ── dedup + gap detect + refine → returns gap records ────────────────
def _pca(xy):
    c = xy.mean(0); ev, evec = np.linalg.eigh(np.cov((xy-c).T))
    d = evec[:,np.argmax(ev)]; d /= np.linalg.norm(d)
    return c, d, (xy-c) @ d

def _perp(c1,d1,c2,d2):
    a = d1+d2; a /= np.linalg.norm(a); p = np.array([-a[1],a[0]])
    return abs((c2-c1)@p)

def detect_gaps(wall_arrays):
    """Return list of dicts with gap_center_xy, width_m, confidence."""
    raws = []
    for pts3d in wall_arrays:
        xy = pts3d[:,:2].copy(); c, d, t = _pca(xy)
        raws.append(dict(xy=xy, c=c, d=d, t=t))
    n = len(raws); par = list(range(n))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i+1, n):
            if abs(raws[i]["d"]@raws[j]["d"]) < MERGE_DOT: continue
            if _perp(raws[i]["c"],raws[i]["d"],raws[j]["c"],raws[j]["d"]) > MERGE_PERP: continue
            ad = raws[i]["d"]+raws[j]["d"]; ad /= np.linalg.norm(ad)
            t_i = (raws[i]["xy"]-raws[i]["c"])@ad; t_j = (raws[j]["xy"]-raws[j]["c"])@ad
            ov = max(0,min(t_i.max(),t_j.max())-max(t_i.min(),t_j.min()))
            sh = min(np.ptp(t_i),np.ptp(t_j))
            if sh > 0 and ov/sh >= MERGE_OVERLAP:
                ri, rj = find(i), find(j)
                if ri != rj: par[rj] = ri
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)

    gaps = []
    for members in groups.values():
        xy = np.vstack([raws[m]["xy"] for m in members])
        c, d, t = _pca(xy)
        edges = np.arange(t.min(), t.max()+BIN_SIZE, BIN_SIZE)
        counts, edges = np.histogram(t, bins=edges); occ = counts >= MIN_DENS
        med = float(np.median(counts[occ])) if occ.any() else 1.0
        ctrs = (edges[:-1]+edges[1:])/2; nn = len(occ); ii = 0
        while ii < nn:
            if occ[ii]: ii += 1; continue
            gs = ii; ii += 1
            while ii < nn:
                if not occ[ii]: ii += 1; continue
                run, k = 0, ii
                while k < nn and occ[k]: run += 1; k += 1
                if run >= REOCC: break
                ii = k
            ge = ii
            if gs > 0 and ge < nn:
                gs_t = ctrs[gs]-BIN_SIZE/2; ge_t = ctrs[ge-1]+BIN_SIZE/2
                w_bin = ge_t - gs_t
                if GAP_RANGE[0] <= w_bin <= GAP_RANGE[1]:
                    ml = (t>=gs_t-BIN_SIZE)&(t<=gs_t+BIN_SIZE)
                    mr = (t>=ge_t-BIN_SIZE)&(t<=ge_t+BIN_SIZE)
                    rl = float(t[ml].max()) if ml.any() else gs_t
                    rr = float(t[mr].min()) if mr.any() else ge_t
                    conf = min(1.0, min(counts[gs-1], counts[ge]) / med) if med else 0.5
                    center = c + (rl+rr)/2 * d
                    gaps.append(dict(
                        gap_center_xy=center.round(4).tolist(),
                        width_m=round(rr-rl, 4), confidence=round(conf, 3),
                        source_wall="+".join(str(m) for m in sorted(members))))
    return gaps


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-start", type=int, default=20)
    ap.add_argument("--frame-end", type=int, default=50)
    ap.add_argument("--frame-step", type=int, default=1)
    args = ap.parse_args()
    indices = list(range(args.frame_start, args.frame_end+1, args.frame_step))

    print(f"Sprint 8.8: spatial gap clustering — {len(indices)} frames")

    # ── Phase 1: per-frame pipeline ──────────────────────────────────
    all_gaps = []
    t0 = time.time()
    for fidx, pts in read_frames(indices):
        pcd = preprocess(pts)
        walls = segment_walls(pcd)
        frame_gaps = detect_gaps(walls)
        n_gaps = len(frame_gaps)
        for g in frame_gaps: g["frame"] = fidx
        all_gaps.extend(frame_gaps)
        w_str = ", ".join(f"{g['width_m']*100:.0f}cm" for g in frame_gaps) or "none"
        print(f"  frame {fidx:3d}: {n_gaps} gap(s) [{w_str}]")
    elapsed = time.time() - t0

    print(f"\n  Total gaps collected: {len(all_gaps)} from {len(indices)} frames")
    if not all_gaps:
        print("  No gaps to cluster."); return

    # ── Phase 2: DBSCAN clustering ───────────────────────────────────
    centers = np.array([g["gap_center_xy"] for g in all_gaps])
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(centers)

    clusters = {}
    for i, lab in enumerate(labels):
        if lab == -1: continue
        clusters.setdefault(lab, []).append(i)

    print(f"  DBSCAN: {len(clusters)} cluster(s), {(labels==-1).sum()} noise points")

    cluster_stats = []
    for cid, idxs in sorted(clusters.items()):
        ws = np.array([all_gaps[i]["width_m"]*100 for i in idxs])
        confs = np.array([all_gaps[i]["confidence"] for i in idxs])
        frames = sorted(set(all_gaps[i]["frame"] for i in idxs))
        cx = np.mean([all_gaps[i]["gap_center_xy"][0] for i in idxs])
        cy = np.mean([all_gaps[i]["gap_center_xy"][1] for i in idxs])
        score = len(frames) * float(confs.mean())
        cluster_stats.append(dict(
            cluster_id=int(cid), center_xy=[round(cx,3), round(cy,3)],
            frame_count=len(frames), frames=frames,
            widths_cm=ws.round(2).tolist(),
            mean_width_cm=round(float(ws.mean()),2),
            median_width_cm=round(float(np.median(ws)),2),
            std_width_cm=round(float(ws.std()),2),
            mean_confidence=round(float(confs.mean()),3),
            score=round(score, 2)))
        print(f"    cluster {cid}: {len(frames)} frames, "
              f"mean={ws.mean():.1f}cm, std={ws.std():.1f}cm, score={score:.1f}")

    # ── Phase 3: winning cluster + outlier rejection ─────────────────
    winner = max(cluster_stats, key=lambda c: c["score"])
    wid = winner["cluster_id"]
    print(f"\n  Winner: cluster {wid} (score={winner['score']:.1f})")

    raw_ws = np.array(winner["widths_cm"])
    med = np.median(raw_ws)
    std = raw_ws.std()
    mask = np.abs(raw_ws - med) <= SIGMA_REJECT * std
    filt_ws = raw_ws[mask]
    n_rejected = len(raw_ws) - len(filt_ws)

    final = dict(
        cluster_id=wid, center_xy=winner["center_xy"],
        raw_count=len(raw_ws), filtered_count=len(filt_ws),
        rejected=n_rejected,
        raw_widths_cm=raw_ws.round(2).tolist(),
        filtered_widths_cm=filt_ws.round(2).tolist(),
        mean_cm=round(float(filt_ws.mean()),2) if len(filt_ws) else None,
        median_cm=round(float(np.median(filt_ws)),2) if len(filt_ws) else None,
        std_cm=round(float(filt_ws.std()),2) if len(filt_ws) else None)

    # ── GT comparison ────────────────────────────────────────────────
    gt_clear, gt_rough = 85.3, 91.4
    if final["mean_cm"]:
        em = round(final["mean_cm"] - gt_clear, 2)
        emd = round(final["median_cm"] - gt_clear, 2)
        print(f"\n  {'Method':<30s} {'Error vs clear':>15s}  {'MAE':>6s}")
        print(f"  {'-'*55}")
        print(f"  {'Single-frame (8.5)':<30s} {'−11.10 cm':>15s}  {'11.10':>6s}")
        print(f"  {'Multi-frame raw mean (8.7)':<30s} {'+21.69 cm':>15s}  {'21.69':>6s}")
        print(f"  {'Multi-frame raw median (8.7)':<30s} {'+10.21 cm':>15s}  {'10.21':>6s}")
        print(f"  {'Tracked-cluster mean (8.8)':<30s} {em:+.2f} cm{'':>7s}  {abs(em):.2f}")
        print(f"  {'Tracked-cluster median (8.8)':<30s} {emd:+.2f} cm{'':>7s}  {abs(emd):.2f}")

    print(f"  Runtime: {elapsed:.1f}s")

    # ── save ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gap_clusters_run01.json", "w") as f:
        json.dump({"clusters": cluster_stats, "all_gaps": len(all_gaps)}, f, indent=2)
    with open(RESULTS_DIR / "multi_frame_tracked_run01.json", "w") as f:
        json.dump({"winning_cluster": final, "comparison": {
            "gt_clear_cm": gt_clear, "gt_rough_cm": gt_rough,
            "err_mean_vs_clear": em if final["mean_cm"] else None,
            "err_median_vs_clear": emd if final["mean_cm"] else None,
        }}, f, indent=2)

    # ── plot ──────────────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    clrs = plt.cm.Set1(np.linspace(0, 1, max(len(clusters), 1)))

    # (a) gap centers colored by cluster
    ax = axes[0]
    for ci, (cid, idxs) in enumerate(sorted(clusters.items())):
        pts = np.array([all_gaps[i]["gap_center_xy"] for i in idxs])
        is_win = cid == wid
        ax.scatter(pts[:,0], pts[:,1], c=[clrs[ci % len(clrs)]], s=60 if is_win else 25,
                   marker="*" if is_win else "o", edgecolors="black" if is_win else "none",
                   linewidths=1.5 if is_win else 0, zorder=5 if is_win else 3,
                   label=f"C{cid} ({len(idxs)} pts){' ★' if is_win else ''}")
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx):
        npts = centers[noise_idx]
        ax.scatter(npts[:,0], npts[:,1], c="gray", s=10, alpha=0.4, label="noise")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
    ax.set_title("Gap Centers by Cluster"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (b) histogram within winning cluster
    ax = axes[1]
    ax.hist(raw_ws, bins=12, color="#1f77b4", edgecolor="black", alpha=0.4, label="raw")
    if len(filt_ws):
        ax.hist(filt_ws, bins=12, color="#2ca02c", edgecolor="black", alpha=0.6, label="filtered")
        ax.axvline(filt_ws.mean(), color="red", ls="-", lw=2,
                   label=f"mean={filt_ws.mean():.1f}")
        ax.axvline(np.median(filt_ws), color="orange", ls="--", lw=2,
                   label=f"median={np.median(filt_ws):.1f}")
    ax.axvline(gt_clear, color="green", ls=":", lw=2, label=f"GT clear={gt_clear}")
    ax.set_xlabel("Width (cm)"); ax.set_ylabel("Count")
    ax.set_title(f"Winning Cluster C{wid} ({n_rejected} outliers rejected)")
    ax.legend(fontsize=7)

    # (c) bar: cluster frame count
    ax = axes[2]
    cids = [c["cluster_id"] for c in cluster_stats]
    fcounts = [c["frame_count"] for c in cluster_stats]
    colors = ["#d62728" if c == wid else "#1f77b4" for c in cids]
    ax.bar([f"C{c}" for c in cids], fcounts, color=colors, edgecolor="black")
    ax.set_ylabel("Frame Count"); ax.set_title("Cluster Persistence")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"Sprint 8.8: Spatial Gap Clustering — {len(clusters)} clusters | "
                 f"{np.datetime64('today')}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "sprint8_8_clustering.png"), dpi=150, facecolor="white")
    plt.close()
    print(f"\nVisualization: {FIG_DIR / 'sprint8_8_clustering.png'}")


if __name__ == "__main__":
    main()
