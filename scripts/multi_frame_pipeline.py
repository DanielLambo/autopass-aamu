#!/usr/bin/env python3
"""Sprint 8.10: Geometric transit-threshold selector.

Selects the gap cluster whose center is closest to the robot's odometry
trajectory, subject to a hard transit threshold (TRANSIT_THRESHOLD_M).

Transit threshold rationale:
    A transited doorway's gap center lies on or very near the robot's
    path.  TRANSIT_THRESHOLD_M = 0.50 m is derived from geometry, not
    ground truth: ADA §403.5.1 sets minimum corridor clear width at
    0.91 m, so a robot walking through the center of ANY passable
    corridor is at most ~0.46 m from either wall.  A cluster center
    within 0.50 m of the trajectory polyline is therefore plausibly
    the passage the robot transited, while clusters further away are
    observed-but-not-traversed structural features.

    This replaces the combined frame_count * confidence * proximity
    score from Sprint 8.9 — that formulation required tuning the
    composition weights, and frame-count dominance caused it to select
    a persistent non-doorway gap.  The transit-threshold selector uses
    a single, geometrically motivated cut with no learned parameters.
"""
from __future__ import annotations

import argparse
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
ROI_MIN, ROI_MAX = np.array([-5,-5,-0.5]), np.array([5,5,2.5])
VOXEL_SIZE = 0.03; SOR_NB, SOR_STD = 20, 2.0
HORIZ_NZ = 0.9; MAX_HORIZ = 4; PLANE_DIST = 0.05; MIN_PLANE_PTS = 150
MAX_WALLS = 4; WALL_DIST = 0.03; WALL_NZ = 0.2; WALL_ITERS = 5000; WALL_MIN = 100
BIN_SIZE = 0.05; MIN_DENS = 5; REOCC = 2; GAP_RANGE = (0.6, 2.0)
MERGE_DOT = 0.95; MERGE_PERP = 0.30; MERGE_OVERLAP = 0.50
DBSCAN_EPS = 0.30; DBSCAN_MIN = 3; SIGMA_REJECT = 2.0
TRANSIT_THRESHOLD_M = 0.50

POINTFIELD_DTYPES = {1:("B",1),2:("b",1),3:("H",2),4:("h",2),
                     5:("I",4),6:("i",4),7:("f",4),8:("d",8)}


# ── bag reading ──────────────────────────────────────────────────────
def parse_pc2(msg):
    fields = {f.name: (f.offset, POINTFIELD_DTYPES[f.datatype][0]) for f in msg.fields}
    ps, data, n = msg.point_step, bytes(msg.data), msg.width * msg.height
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        b = i * ps
        for j, ax in enumerate(("x", "y", "z")):
            pts[i,j] = struct.unpack_from(fields[ax][1], data, b+fields[ax][0])[0]
    return pts[np.isfinite(pts).all(axis=1)]


def read_bag(indices):
    """Read PC frames + odom trajectory in one pass through the bag."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    want = set(indices)
    frames = {}  # idx -> pts
    odom_data = []  # (timestamp_ns, x, y)
    pc_times = []  # all PC timestamps

    with AnyReader([BAG_PATH], default_typestore=ts) as reader:
        pc_conns = [c for c in reader.connections if c.topic == "/point_cloud2"]
        od_conns = [c for c in reader.connections if c.topic == "/odom"]

        # Collect odom
        for conn, t, raw in reader.messages(connections=od_conns):
            msg = reader.deserialize(raw, conn.msgtype)
            p = msg.pose.pose.position
            odom_data.append((t, p.x, p.y))

        # Collect PC frames
        for idx, (conn, t, raw) in enumerate(reader.messages(connections=pc_conns)):
            pc_times.append(t)
            if idx in want:
                frames[idx] = parse_pc2(reader.deserialize(raw, conn.msgtype))
                want.discard(idx)
                if not want: break
            if idx > max(indices): break

    # Match each PC frame to nearest odom
    odom_ts = np.array([o[0] for o in odom_data])
    trajectory = {}  # frame_idx -> (x, y)
    for fi in indices:
        if fi < len(pc_times):
            oi = np.argmin(np.abs(odom_ts - pc_times[fi]))
            trajectory[fi] = (odom_data[oi][1], odom_data[oi][2])

    return frames, trajectory


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


# ── gap detection ────────────────────────────────────────────────────
def _pca(xy):
    c = xy.mean(0); ev, evec = np.linalg.eigh(np.cov((xy-c).T))
    d = evec[:,np.argmax(ev)]; d /= np.linalg.norm(d)
    return c, d, (xy-c)@d

def detect_gaps(wall_arrays):
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
            pd = raws[i]["d"]+raws[j]["d"]; pd /= np.linalg.norm(pd)
            pp = np.array([-pd[1], pd[0]])
            if abs((raws[j]["c"]-raws[i]["c"])@pp) > MERGE_PERP: continue
            ti = (raws[i]["xy"]-raws[i]["c"])@pd; tj = (raws[j]["xy"]-raws[j]["c"])@pd
            ov = max(0,min(ti.max(),tj.max())-max(ti.min(),tj.min()))
            sh = min(np.ptp(ti),np.ptp(tj))
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


# ── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-start", type=int, default=20)
    ap.add_argument("--frame-end", type=int, default=50)
    ap.add_argument("--frame-step", type=int, default=1)
    args = ap.parse_args()
    indices = list(range(args.frame_start, args.frame_end+1, args.frame_step))

    print(f"Sprint 8.10: transit-threshold selector — {len(indices)} frames")

    # Phase 1: read bag (odom + PC frames in one pass)
    t0 = time.time()
    frames, trajectory = read_bag(indices)
    traj_pts = np.array([trajectory[fi] for fi in sorted(trajectory)])
    print(f"  Odom trajectory: {len(trajectory)} poses")

    # Phase 2: per-frame gap detection
    all_gaps = []
    for fidx in sorted(frames):
        pcd = preprocess(frames[fidx])
        walls = segment_walls(pcd)
        fg = detect_gaps(walls)
        for g in fg: g["frame"] = fidx
        all_gaps.extend(fg)
        w_str = ", ".join(f"{g['width_m']*100:.0f}" for g in fg) or "-"
        print(f"  frame {fidx:3d}: {len(fg)} gap(s) [{w_str}] cm")
    elapsed = time.time() - t0

    print(f"\n  {len(all_gaps)} gaps from {len(frames)} frames ({elapsed:.1f}s)")
    if not all_gaps: print("  No gaps."); return

    # Phase 3: DBSCAN
    centers = np.array([g["gap_center_xy"] for g in all_gaps])
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(centers)
    clusters = {}
    for i, lab in enumerate(labels):
        if lab >= 0: clusters.setdefault(lab, []).append(i)
    print(f"  DBSCAN: {len(clusters)} clusters, {(labels==-1).sum()} noise")

    # Phase 4: compute per-cluster metadata + trajectory distance
    cluster_stats = []
    for cid, idxs in sorted(clusters.items()):
        ws = np.array([all_gaps[i]["width_m"]*100 for i in idxs])
        confs = np.array([all_gaps[i]["confidence"] for i in idxs])
        frs = sorted(set(all_gaps[i]["frame"] for i in idxs))
        cx = np.mean([all_gaps[i]["gap_center_xy"][0] for i in idxs])
        cy = np.mean([all_gaps[i]["gap_center_xy"][1] for i in idxs])
        cc = np.array([cx, cy])

        dists = np.linalg.norm(traj_pts - cc, axis=1)
        min_dist = float(dists.min())

        cluster_stats.append(dict(
            cluster_id=int(cid), center_xy=[round(cx,3),round(cy,3)],
            frame_count=len(frs), frames=frs,
            widths_cm=ws.round(2).tolist(),
            mean_width_cm=round(float(ws.mean()),2),
            median_width_cm=round(float(np.median(ws)),2),
            std_width_cm=round(float(ws.std()),2),
            mean_confidence=round(float(confs.mean()),3),
            min_traj_dist_m=round(min_dist,3)))
        tag = " ◄ transited" if min_dist < TRANSIT_THRESHOLD_M else ""
        print(f"    C{cid}: {len(frs)}fr, mean={ws.mean():.1f}cm, "
              f"traj_dist={min_dist:.3f}m{tag}")

    # Phase 5: transit-threshold selection (closest cluster within threshold)
    transited = [c for c in cluster_stats if c["min_traj_dist_m"] < TRANSIT_THRESHOLD_M]
    if not transited:
        print("\n  No transited doorway detected in this run.")
        winner = None
    else:
        winner = min(transited, key=lambda c: c["min_traj_dist_m"])

    wid = winner["cluster_id"] if winner else -1
    if winner:
        print(f"\n  Selected: C{wid} (traj_dist={winner['min_traj_dist_m']:.3f}m "
              f"< {TRANSIT_THRESHOLD_M}m threshold)")

    # 2-sigma filter on winning cluster
    raw_ws = np.array(winner["widths_cm"]) if winner else np.array([])
    if len(raw_ws):
        med = np.median(raw_ws); std = raw_ws.std()
        filt_ws = raw_ws[np.abs(raw_ws - med) <= SIGMA_REJECT * std]
    else:
        filt_ws = np.array([])
    n_rej = len(raw_ws) - len(filt_ws)

    final = dict(
        cluster_id=wid if winner else None,
        center_xy=winner["center_xy"] if winner else None,
        raw_count=len(raw_ws), filtered_count=len(filt_ws), rejected=n_rej,
        filtered_widths_cm=filt_ws.round(2).tolist() if len(filt_ws) else [],
        mean_cm=round(float(filt_ws.mean()),2) if len(filt_ws) else None,
        median_cm=round(float(np.median(filt_ws)),2) if len(filt_ws) else None,
        std_cm=round(float(filt_ws.std()),2) if len(filt_ws) else None,
        min_traj_dist_m=winner["min_traj_dist_m"] if winner else None,
        transit_threshold_m=TRANSIT_THRESHOLD_M,
        selection_method="closest_cluster_within_transit_threshold")

    # Comparison table
    gt_c, gt_r = 85.3, 91.4
    print(f"\n  {'Method':<38s} {'Width':>7s} {'vs clear':>9s} {'vs rough':>9s}")
    print(f"  {'-'*66}")
    prev = [("Single-frame (8.5)", 74.2),
            ("Multi-frame raw median (8.7)", 95.5),
            ("Persistence-scored C1 (8.8)", 114.4),
            ("Combined-scored C1 (8.9)", 118.0)]
    if final["mean_cm"]:
        prev.append((f"Transit-threshold C{wid} mean (8.10)", final["mean_cm"]))
        prev.append((f"Transit-threshold C{wid} median (8.10)", final["median_cm"]))
    for label, w in prev:
        ec = round(w - gt_c, 2); er = round(w - gt_r, 2)
        print(f"  {label:<38s} {w:6.1f}  {ec:+7.1f} cm  {er:+7.1f} cm")

    print(f"  Runtime: {elapsed:.1f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "clusters_with_trajectory_run01.json", "w") as f:
        json.dump({"clusters": cluster_stats,
                   "transit_threshold_m": TRANSIT_THRESHOLD_M}, f, indent=2)
    with open(RESULTS_DIR / "multi_frame_tracked_run01_v3.json", "w") as f:
        json.dump({"winning_cluster": final,
                   "gt_clear_cm": gt_c, "gt_rough_cm": gt_r}, f, indent=2)

    # Plot
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    clrs = plt.cm.Set1(np.linspace(0, 1, max(len(clusters), 1)))

    # (a) trajectory + clusters with distance labels
    ax = axes[0]
    ax.plot(traj_pts[:,0], traj_pts[:,1], "b-", lw=2, alpha=0.6, label="trajectory")
    ax.plot(traj_pts[0,0], traj_pts[0,1], "bs", ms=8, label="start")
    ax.plot(traj_pts[-1,0], traj_pts[-1,1], "b^", ms=8, label="end")
    for ci, cs in enumerate(cluster_stats):
        cid = cs["cluster_id"]
        idxs = clusters[cid]
        pts = np.array([all_gaps[i]["gap_center_xy"] for i in idxs])
        is_w = cid == wid
        ax.scatter(pts[:,0], pts[:,1], c="limegreen" if is_w else [clrs[ci%len(clrs)]],
                   s=80 if is_w else 25, marker="*" if is_w else "o",
                   edgecolors="black" if is_w else "none", linewidths=1.5 if is_w else 0,
                   zorder=5 if is_w else 3,
                   label=f"C{cid}{' ★' if is_w else ''} d={cs['min_traj_dist_m']:.2f}m")
        ax.annotate(f"{cs['min_traj_dist_m']:.2f}m", cs["center_xy"],
                    textcoords="offset points", xytext=(6, 6), fontsize=7,
                    color="darkred" if cs["min_traj_dist_m"] < TRANSIT_THRESHOLD_M else "gray")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
    ax.set_title(f"Transit Threshold = {TRANSIT_THRESHOLD_M}m")
    ax.legend(fontsize=5, loc="lower right"); ax.grid(True, alpha=0.3)

    # (b) min distance per cluster (bar chart, threshold line)
    ax = axes[1]
    cids = [c["cluster_id"] for c in cluster_stats]
    mdists = [c["min_traj_dist_m"] for c in cluster_stats]
    bar_colors = ["limegreen" if c == wid else
                  ("#aaddaa" if d < TRANSIT_THRESHOLD_M else "#1f77b4")
                  for c, d in zip(cids, mdists)]
    ax.bar([f"C{c}" for c in cids], mdists, color=bar_colors, edgecolor="black")
    ax.axhline(TRANSIT_THRESHOLD_M, color="red", ls="--", lw=2,
               label=f"threshold = {TRANSIT_THRESHOLD_M}m")
    ax.set_ylabel("Min Distance to Trajectory (m)")
    ax.set_title("Distance to Trajectory"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (c) winning cluster histogram
    ax = axes[2]
    if len(raw_ws):
        ax.hist(raw_ws, bins=10, color="#1f77b4", edgecolor="black", alpha=0.4, label="raw")
    if len(filt_ws):
        ax.hist(filt_ws, bins=10, color="#2ca02c", edgecolor="black", alpha=0.6, label="filtered")
        ax.axvline(filt_ws.mean(), color="red", ls="-", lw=2,
                   label=f"mean={filt_ws.mean():.1f}")
        ax.axvline(np.median(filt_ws), color="orange", ls="--", lw=2,
                   label=f"median={np.median(filt_ws):.1f}")
    ax.axvline(gt_c, color="green", ls=":", lw=2, label=f"GT={gt_c}")
    ax.set_xlabel("Width (cm)"); ax.set_ylabel("Count")
    title = f"C{wid} widths ({n_rej} rejected)" if winner else "No selection"
    ax.set_title(title); ax.legend(fontsize=7)

    plt.suptitle(f"Sprint 8.10: Transit-Threshold Selector | "
                 f"{np.datetime64('today')}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "sprint8_10_selection.png"), dpi=150, facecolor="white")
    plt.close()
    print(f"\n  Viz: {FIG_DIR / 'sprint8_10_selection.png'}")


if __name__ == "__main__":
    main()
