#!/usr/bin/env python3
"""Sprint 10: RANSAC stability characterization.

Runs the full multi-frame pipeline N=50 times on the same run_01 data
with NO seed changes.  Open3D's segment_plane() uses internal random
sampling without an exposed seed, so each run produces a different
horizontal plane extraction → different remaining points → different
wall detection → different gap widths.  This script quantifies that
noise distribution honestly.

Optimization: the bag is read once and frames preprocessed once
(deterministic operations).  Only the non-deterministic stages
(segmentation + downstream) re-run each trial.
"""
from __future__ import annotations

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
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# ── pipeline params (UNCHANGED) ─────────────────────────────────────
ROI_MIN, ROI_MAX = np.array([-5,-5,-0.5]), np.array([5,5,2.5])
VOXEL_SIZE = 0.03; SOR_NB, SOR_STD = 20, 2.0
HORIZ_NZ = 0.9; MAX_HORIZ = 4; PLANE_DIST = 0.05; MIN_PLANE_PTS = 150
MAX_WALLS = 4; WALL_DIST = 0.03; WALL_NZ = 0.2; WALL_ITERS = 5000; WALL_MIN = 100
BIN_SIZE = 0.05; MIN_DENS = 5; REOCC = 2; GAP_RANGE = (0.6, 2.0)
MERGE_DOT = 0.95; MERGE_PERP = 0.30; MERGE_OVERLAP = 0.50
DBSCAN_EPS = 0.30; DBSCAN_MIN = 3; SIGMA_REJECT = 2.0
TRANSIT_THRESHOLD_M = 0.50

N_TRIALS = 50
FRAME_START, FRAME_END = 20, 50
GT_CLEAR = 85.3

POINTFIELD_DTYPES = {1:("B",1),2:("b",1),3:("H",2),4:("h",2),
                     5:("I",4),6:("i",4),7:("f",4),8:("d",8)}


# ── bag reading (once) ───────────────────────────────────────────────
def parse_pc2(msg):
    fields = {f.name: (f.offset, POINTFIELD_DTYPES[f.datatype][0]) for f in msg.fields}
    ps, data, n = msg.point_step, bytes(msg.data), msg.width * msg.height
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        b = i * ps
        for j, ax in enumerate(("x","y","z")):
            pts[i,j] = struct.unpack_from(fields[ax][1], data, b+fields[ax][0])[0]
    return pts[np.isfinite(pts).all(axis=1)]

def load_once():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    indices = list(range(FRAME_START, FRAME_END+1))
    want = set(indices); raw_frames = {}; odom = []; pc_times = []
    with AnyReader([BAG_PATH], default_typestore=ts) as reader:
        for conn, t, raw in reader.messages(
                [c for c in reader.connections if c.topic == "/odom"]):
            msg = reader.deserialize(raw, conn.msgtype)
            odom.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y))
        for idx, (conn, t, raw) in enumerate(reader.messages(
                [c for c in reader.connections if c.topic == "/point_cloud2"])):
            pc_times.append(t)
            if idx in want:
                raw_frames[idx] = parse_pc2(reader.deserialize(raw, conn.msgtype))
                want.discard(idx)
                if not want: break
    odom_ts = np.array([o[0] for o in odom])
    traj = {}
    for fi in indices:
        if fi < len(pc_times):
            oi = np.argmin(np.abs(odom_ts - pc_times[fi]))
            traj[fi] = (odom[oi][1], odom[oi][2])
    # Preprocess deterministically
    pcds = {}
    for fi, pts in raw_frames.items():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(ROI_MIN, ROI_MAX))
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        pcd, _ = pcd.remove_statistical_outlier(SOR_NB, SOR_STD)
        pcds[fi] = pcd
    traj_pts = np.array([traj[fi] for fi in sorted(traj)])
    return pcds, traj, traj_pts


# ── pipeline stages (NO seeds added) ────────────────────────────────
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
        inl = np.where(np.abs(pts @ nrm + (-nrm@pts[ix[0]])) < WALL_DIST)[0]
        if len(inl) > len(best_inl): best_inl, best_mod = inl, nrm
    if best_mod is not None and len(best_inl) < WALL_MIN:
        return None, np.array([],dtype=int)
    return best_mod, best_inl

def segment_walls(pcd):
    rem = o3d.geometry.PointCloud(pcd)
    for _ in range(MAX_HORIZ):
        if len(rem.points) < MIN_PLANE_PTS: break
        mod, inl = rem.segment_plane(PLANE_DIST, 3, 1000)
        if len(inl) < MIN_PLANE_PTS: break
        rem = rem.select_by_index(inl, invert=True)
    pts = np.asarray(rem.points).copy(); walls = []
    for i in range(MAX_WALLS):
        if len(pts) < WALL_MIN: break
        mod, inl = _wall_ransac(pts, 42+i)
        if mod is None or len(inl) < WALL_MIN: break
        walls.append(pts[inl])
        m = np.ones(len(pts), dtype=bool); m[inl] = False; pts = pts[m]
    return walls

def _pca(xy):
    c = xy.mean(0); ev, evec = np.linalg.eigh(np.cov((xy-c).T))
    d = evec[:,np.argmax(ev)]; d /= np.linalg.norm(d)
    return c, d, (xy-c)@d

def detect_gaps(wall_arrays):
    raws = []
    for p in wall_arrays:
        xy = p[:,:2].copy(); c, d, t = _pca(xy)
        raws.append(dict(xy=xy, c=c, d=d, t=t))
    n = len(raws); par = list(range(n))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i+1, n):
            if abs(raws[i]["d"]@raws[j]["d"]) < MERGE_DOT: continue
            pd = raws[i]["d"]+raws[j]["d"]; pd /= np.linalg.norm(pd)
            pp = np.array([-pd[1], pd[0]])
            if abs((raws[j]["c"]-raws[i]["c"])@pp) > MERGE_PERP: continue
            ti = (raws[i]["xy"]-raws[i]["c"])@pd; tj = (raws[j]["xy"]-raws[j]["c"])@pd
            ov = max(0, min(ti.max(),tj.max())-max(ti.min(),tj.min()))
            sh = min(np.ptp(ti), np.ptp(tj))
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
                if GAP_RANGE[0] <= ge_t-gs_t <= GAP_RANGE[1]:
                    ml = (t>=gs_t-BIN_SIZE)&(t<=gs_t+BIN_SIZE)
                    mr = (t>=ge_t-BIN_SIZE)&(t<=ge_t+BIN_SIZE)
                    rl = float(t[ml].max()) if ml.any() else gs_t
                    rr = float(t[mr].min()) if mr.any() else ge_t
                    conf = min(1.0, min(counts[gs-1],counts[ge])/med) if med else 0.5
                    center = c + (rl+rr)/2*d
                    gaps.append(dict(gap_center_xy=center.round(4).tolist(),
                                     width_m=round(rr-rl,4), confidence=round(conf,3)))
    return gaps


def run_trial(pcds, traj_pts):
    """One full multi-frame pipeline trial. Returns dict."""
    all_gaps = []
    for fidx in sorted(pcds):
        walls = segment_walls(pcds[fidx])
        fg = detect_gaps(walls)
        for g in fg: g["frame"] = fidx
        all_gaps.extend(fg)
    if not all_gaps:
        return dict(status="no_gaps", width_cm=None, min_dist=None, n_cluster_frames=0)
    centers = np.array([g["gap_center_xy"] for g in all_gaps])
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(centers)
    clusters = {}
    for i, lab in enumerate(labels):
        if lab >= 0: clusters.setdefault(lab, []).append(i)
    if not clusters:
        return dict(status="no_clusters", width_cm=None, min_dist=None, n_cluster_frames=0)

    best, best_dist = None, float("inf")
    for cid, idxs in clusters.items():
        ws = np.array([all_gaps[i]["width_m"]*100 for i in idxs])
        frs = sorted(set(all_gaps[i]["frame"] for i in idxs))
        cx = np.mean([all_gaps[i]["gap_center_xy"][0] for i in idxs])
        cy = np.mean([all_gaps[i]["gap_center_xy"][1] for i in idxs])
        md = float(np.linalg.norm(traj_pts - np.array([cx,cy]), axis=1).min())
        if md < TRANSIT_THRESHOLD_M and md < best_dist:
            best = dict(ws=ws, n_frames=len(frs), min_dist=md)
            best_dist = md
    if best is None:
        return dict(status="no_transited", width_cm=None, min_dist=None, n_cluster_frames=0)

    raw = best["ws"]
    med = np.median(raw); std = raw.std()
    filt = raw[np.abs(raw - med) <= SIGMA_REJECT * std]
    if len(filt) == 0: filt = raw
    return dict(status="detected",
                width_cm=round(float(filt.mean()), 2),
                median_cm=round(float(np.median(filt)), 2),
                min_dist=round(best_dist, 4),
                n_cluster_frames=best["n_frames"])


# ── main ─────────────────────────────────────────────────────────────
def main():
    print(f"Sprint 10: RANSAC Stability — {N_TRIALS} trials on run_01")
    print("Loading bag and preprocessing (one-time)...")
    pcds, traj, traj_pts = load_once()
    print(f"  {len(pcds)} frames preprocessed, {len(traj)} trajectory poses\n")

    results = []
    t0 = time.time()
    for i in range(N_TRIALS):
        r = run_trial(pcds, traj_pts)
        results.append(r)
        tag = f"{r['width_cm']:.1f}cm d={r['min_dist']:.3f}" if r["width_cm"] else r["status"]
        print(f"  trial {i+1:2d}/{N_TRIALS}: {tag}")
    elapsed = time.time() - t0

    # Analysis
    detected = [r for r in results if r["width_cm"] is not None]
    widths = np.array([r["width_cm"] for r in detected])
    dists = np.array([r["min_dist"] for r in detected])
    n_det = len(detected)

    within_10 = np.sum(np.abs(widths - GT_CLEAR) <= 10) if n_det else 0
    within_5 = np.sum(np.abs(widths - GT_CLEAR) <= 5) if n_det else 0

    print(f"\n{'='*55}")
    print(f"  Trials:              {N_TRIALS}")
    print(f"  Detected:            {n_det} ({n_det/N_TRIALS:.0%})")
    if n_det:
        print(f"  Mean width:          {widths.mean():.2f} cm")
        print(f"  Median width:        {np.median(widths):.2f} cm")
        print(f"  Std:                 {widths.std():.2f} cm")
        print(f"  Min / Max:           {widths.min():.1f} / {widths.max():.1f} cm")
        q1, q3 = np.percentile(widths, 25), np.percentile(widths, 75)
        print(f"  IQR:                 [{q1:.1f}, {q3:.1f}] cm (range {q3-q1:.1f})")
        print(f"  Within ±10cm of GT:  {within_10}/{n_det} ({within_10/n_det:.0%})")
        print(f"  Within ±5cm of GT:   {within_5}/{n_det} ({within_5/n_det:.0%})")
        best_i = np.argmin(np.abs(widths - GT_CLEAR))
        worst_i = np.argmax(np.abs(widths - GT_CLEAR))
        print(f"  Best case:           {widths[best_i]:.1f} cm (err {widths[best_i]-GT_CLEAR:+.1f})")
        print(f"  Worst case:          {widths[worst_i]:.1f} cm (err {widths[worst_i]-GT_CLEAR:+.1f})")
    print(f"  Runtime:             {elapsed:.1f}s ({elapsed/N_TRIALS:.1f}s/trial)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ransac_stability_run01.json", "w") as f:
        json.dump({"n_trials": N_TRIALS, "gt_clear_cm": GT_CLEAR,
                   "detection_rate": n_det/N_TRIALS,
                   "trials": results,
                   "stats": {
                       "mean": round(float(widths.mean()),2) if n_det else None,
                       "median": round(float(np.median(widths)),2) if n_det else None,
                       "std": round(float(widths.std()),2) if n_det else None,
                       "within_10cm_pct": round(within_10/n_det,2) if n_det else 0,
                       "within_5cm_pct": round(within_5/n_det,2) if n_det else 0,
                   }}, f, indent=2)

    # Plot
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) histogram of detected widths
    ax = axes[0]
    if n_det:
        ax.hist(widths, bins=20, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.axvline(GT_CLEAR, color="green", ls=":", lw=2, label=f"GT clear={GT_CLEAR}")
        ax.axvline(85.4, color="red", ls="--", lw=1.5, label="Sprint 8.10 = 85.4")
        ax.axvline(177.5, color="orange", ls="--", lw=1.5, label="Sprint 9 = 177.5")
        ax.axvline(widths.mean(), color="black", ls="-", lw=1.5,
                   label=f"mean={widths.mean():.1f}")
    ax.set_xlabel("Detected Width (cm)"); ax.set_ylabel("Count")
    ax.set_title(f"Width Distribution (N={N_TRIALS})"); ax.legend(fontsize=7)

    # (b) running mean and std
    ax = axes[1]
    if n_det >= 2:
        rmean = np.array([widths[:i+1].mean() for i in range(n_det)])
        rstd = np.array([widths[:i+1].std() for i in range(n_det)])
        x = np.arange(1, n_det+1)
        ax.plot(x, rmean, "b-", lw=2, label="running mean")
        ax.fill_between(x, rmean-rstd, rmean+rstd, alpha=0.2, color="blue",
                        label="±1 std")
        ax.axhline(GT_CLEAR, color="green", ls=":", lw=2, label=f"GT={GT_CLEAR}")
    ax.set_xlabel("Trial #"); ax.set_ylabel("Width (cm)")
    ax.set_title("Running Mean ± Std"); ax.legend(fontsize=7)

    # (c) width vs traj distance
    ax = axes[2]
    if n_det:
        colors = ["green" if abs(w-GT_CLEAR) <= 10 else "red" for w in widths]
        ax.scatter(dists, widths, c=colors, s=30, edgecolors="black", linewidths=0.5)
        ax.axhline(GT_CLEAR, color="green", ls=":", lw=2, label=f"GT={GT_CLEAR}")
        ax.axhline(GT_CLEAR+10, color="green", ls="--", lw=0.5, alpha=0.5)
        ax.axhline(GT_CLEAR-10, color="green", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Min Dist to Trajectory (m)"); ax.set_ylabel("Width (cm)")
    ax.set_title("Width vs Cluster Distance"); ax.legend(fontsize=7)

    plt.suptitle(f"Sprint 10: RANSAC Stability ({N_TRIALS} trials, "
                 f"{n_det} detected) | {np.datetime64('today')}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "sprint10_stability.png"), dpi=150, facecolor="white")
    plt.close()
    print(f"\n  Viz: {FIG_DIR / 'sprint10_stability.png'}")


if __name__ == "__main__":
    main()
