#!/usr/bin/env python3
"""Sprint 9: Cross-run held-out validation pipeline.

Runs the complete AutoPass pipeline on one or more rosbags with ZERO
parameter changes from the run_01 development set.  All thresholds are
imported from the Sprint 8.10 pipeline or hardcoded to the same values.

Frame range: indices 20-50 (fixed).  All three runs recorded the same
doorway transit; the robot enters around frame 20 and exits by frame 50
in each.  If a bag has fewer than 51 frames, the range is clamped.

Ground truth: door_01 clear width = 85.3 cm, rough width = 91.4 cm.
These measurements apply to all three runs (same physical doorway).
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
BAGS_DIR = Path.home() / "autopass" / "bags"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# ── pipeline params (UNCHANGED from run_01 development) ─────────────
ROI_MIN, ROI_MAX = np.array([-5,-5,-0.5]), np.array([5,5,2.5])
VOXEL_SIZE = 0.03; SOR_NB, SOR_STD = 20, 2.0
HORIZ_NZ = 0.9; MAX_HORIZ = 4; PLANE_DIST = 0.05; MIN_PLANE_PTS = 150
MAX_WALLS = 4; WALL_DIST = 0.03; WALL_NZ = 0.2; WALL_ITERS = 5000; WALL_MIN = 100
BIN_SIZE = 0.05; MIN_DENS = 5; REOCC = 2; GAP_RANGE = (0.6, 2.0)
MERGE_DOT = 0.95; MERGE_PERP = 0.30; MERGE_OVERLAP = 0.50
DBSCAN_EPS = 0.30; DBSCAN_MIN = 3; SIGMA_REJECT = 2.0
TRANSIT_THRESHOLD_M = 0.50
FRAME_START, FRAME_END = 20, 50

GT_CLEAR, GT_ROUGH = 85.3, 91.4

POINTFIELD_DTYPES = {1:("B",1),2:("b",1),3:("H",2),4:("h",2),
                     5:("I",4),6:("i",4),7:("f",4),8:("d",8)}


# ── bag reading ──────────────────────────────────────────────────────
def parse_pc2(msg):
    fields = {f.name: (f.offset, POINTFIELD_DTYPES[f.datatype][0]) for f in msg.fields}
    ps, data, n = msg.point_step, bytes(msg.data), msg.width * msg.height
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        b = i * ps
        for j, ax in enumerate(("x","y","z")):
            pts[i,j] = struct.unpack_from(fields[ax][1], data, b+fields[ax][0])[0]
    return pts[np.isfinite(pts).all(axis=1)]

def read_bag(bag_path, indices):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    want = set(indices); frames = {}; odom = []; pc_times = []
    with AnyReader([bag_path], default_typestore=ts) as reader:
        od_c = [c for c in reader.connections if c.topic == "/odom"]
        pc_c = [c for c in reader.connections if c.topic == "/point_cloud2"]
        for conn, t, raw in reader.messages(connections=od_c):
            msg = reader.deserialize(raw, conn.msgtype)
            odom.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y))
        for idx, (conn, t, raw) in enumerate(reader.messages(connections=pc_c)):
            pc_times.append(t)
            if idx in want:
                frames[idx] = parse_pc2(reader.deserialize(raw, conn.msgtype))
                want.discard(idx)
                if not want: break
    if not odom or not pc_times:
        return frames, {}
    odom_ts = np.array([o[0] for o in odom])
    trajectory = {}
    for fi in indices:
        if fi < len(pc_times):
            oi = np.argmin(np.abs(odom_ts - pc_times[fi]))
            trajectory[fi] = (odom[oi][1], odom[oi][2])
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


# ── run one bag ──────────────────────────────────────────────────────
def process_run(run_id, bag_path):
    """Return dict with all metrics for one run, or None if no data."""
    indices = list(range(FRAME_START, FRAME_END + 1))
    frames, trajectory = read_bag(bag_path, indices)
    if not frames:
        return dict(run_id=run_id, status="no_data", n_pc_frames=0)

    traj_pts = np.array([trajectory[fi] for fi in sorted(trajectory)]) if trajectory else None

    all_gaps = []
    for fidx in sorted(frames):
        pcd = preprocess(frames[fidx])
        walls = segment_walls(pcd)
        fg = detect_gaps(walls)
        for g in fg: g["frame"] = fidx
        all_gaps.extend(fg)

    if not all_gaps:
        return dict(run_id=run_id, status="no_gaps", n_pc_frames=len(frames),
                    n_gaps=0)

    centers = np.array([g["gap_center_xy"] for g in all_gaps])
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(centers)
    clusters = {}
    for i, lab in enumerate(labels):
        if lab >= 0: clusters.setdefault(lab, []).append(i)

    if not clusters or traj_pts is None:
        return dict(run_id=run_id, status="no_clusters", n_pc_frames=len(frames),
                    n_gaps=len(all_gaps), n_clusters=len(clusters))

    # Compute min traj dist per cluster, select via transit threshold
    best, best_dist = None, float("inf")
    cluster_info = []
    for cid, idxs in sorted(clusters.items()):
        ws = np.array([all_gaps[i]["width_m"]*100 for i in idxs])
        frs = sorted(set(all_gaps[i]["frame"] for i in idxs))
        cx = np.mean([all_gaps[i]["gap_center_xy"][0] for i in idxs])
        cy = np.mean([all_gaps[i]["gap_center_xy"][1] for i in idxs])
        md = float(np.linalg.norm(traj_pts - np.array([cx,cy]), axis=1).min())
        cluster_info.append(dict(cid=cid, n_frames=len(frs), mean_w=ws.mean(),
                                 median_w=np.median(ws), std_w=ws.std(),
                                 min_dist=md, widths=ws))
        if md < TRANSIT_THRESHOLD_M and md < best_dist:
            best, best_dist = cluster_info[-1], md

    if best is None:
        return dict(run_id=run_id, status="no_transited_cluster",
                    n_pc_frames=len(frames), n_gaps=len(all_gaps),
                    n_clusters=len(clusters),
                    cluster_dists=[round(c["min_dist"], 3) for c in cluster_info],
                    cluster_means=[round(float(c["mean_w"]), 1) for c in cluster_info])

    # 2-sigma filter
    raw = best["widths"]
    med = np.median(raw); std = raw.std()
    filt = raw[np.abs(raw - med) <= SIGMA_REJECT * std]

    return dict(
        run_id=run_id, status="detected",
        n_pc_frames=len(frames), n_gaps=len(all_gaps),
        n_clusters=len(clusters),
        winning_cluster_id=int(best["cid"]),
        winning_cluster_min_dist_m=round(best_dist, 3),
        n_frames_in_cluster=int(best["n_frames"]),
        raw_widths_cm=raw.round(2).tolist(),
        filtered_widths_cm=filt.round(2).tolist(),
        n_rejected=int(len(raw)-len(filt)),
        mean_width_cm=round(float(filt.mean()), 2),
        median_width_cm=round(float(np.median(filt)), 2),
        std_within_cluster_cm=round(float(filt.std()), 2),
        error_vs_clear_cm=round(float(filt.mean()) - GT_CLEAR, 2),
        error_vs_rough_cm=round(float(filt.mean()) - GT_ROUGH, 2),
    )


# ── main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Sprint 9: Cross-Run Held-Out Validation (no parameter changes)")
    print("=" * 65)

    runs = []
    for name in ["run_01", "run_02", "run_03"]:
        bag = BAGS_DIR / name
        if not bag.exists():
            print(f"\n  {name}: bag not found at {bag}")
            runs.append(dict(run_id=name, status="bag_not_found"))
            continue
        print(f"\n  Processing {name}...")
        t0 = time.time()
        result = process_run(name, bag)
        elapsed = round(time.time() - t0, 1)
        result["elapsed_s"] = elapsed
        runs.append(result)
        if result["status"] == "detected":
            print(f"    Detected: mean={result['mean_width_cm']:.1f} cm, "
                  f"median={result['median_width_cm']:.1f} cm, "
                  f"traj_dist={result['winning_cluster_min_dist_m']:.3f} m, "
                  f"err_clear={result['error_vs_clear_cm']:+.1f} cm "
                  f"({elapsed}s)")
        else:
            extra = ""
            if "cluster_dists" in result:
                extra = f", cluster dists={result['cluster_dists']}"
            print(f"    Status: {result['status']}{extra} ({elapsed}s)")

    # Console table
    detected = [r for r in runs if r["status"] == "detected"]
    print(f"\n  {'Run':<8s} {'Status':<12s} {'Frames':>6s} {'Gaps':>5s} "
          f"{'Mean':>7s} {'Median':>7s} {'Std':>6s} {'Err/C':>7s} {'Err/R':>7s}")
    print(f"  {'-'*72}")
    for r in runs:
        if r["status"] == "detected":
            print(f"  {r['run_id']:<8s} {'OK':<12s} {r['n_pc_frames']:>6d} "
                  f"{r['n_gaps']:>5d} {r['mean_width_cm']:>6.1f}  "
                  f"{r['median_width_cm']:>6.1f}  {r['std_within_cluster_cm']:>5.1f}  "
                  f"{r['error_vs_clear_cm']:>+6.1f}  {r['error_vs_rough_cm']:>+6.1f}")
        else:
            nf = r.get('n_pc_frames', 0)
            print(f"  {r['run_id']:<8s} {r['status']:<12s} {nf:>6d}")

    if len(detected) >= 2:
        means = np.array([r["mean_width_cm"] for r in detected])
        maes_c = np.array([abs(r["error_vs_clear_cm"]) for r in detected])
        maes_r = np.array([abs(r["error_vs_rough_cm"]) for r in detected])
        print(f"  {'-'*72}")
        print(f"  {'AGGREGATE':<8s} {'':12s} {'':>6s} {'':>5s} "
              f"{means.mean():>6.1f}  {'':>7s} {means.std():>5.1f}  "
              f"{maes_c.mean():>+6.1f}  {maes_r.mean():>+6.1f}")
        print(f"\n  Cross-run mean:  {means.mean():.2f} cm")
        print(f"  Cross-run std:   {means.std():.2f} cm (run-to-run consistency)")
        print(f"  MAE vs clear:    {maes_c.mean():.2f} cm")
        print(f"  MAE vs rough:    {maes_r.mean():.2f} cm")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cross_run_evaluation.json", "w") as f:
        json.dump({"runs": runs, "gt_clear_cm": GT_CLEAR, "gt_rough_cm": GT_ROUGH,
                   "parameters_changed": False, "transit_threshold_m": TRANSIT_THRESHOLD_M},
                  f, indent=2)

    # Plot
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) bar chart: mean width per run
    ax = axes[0]
    names = [r["run_id"] for r in detected]
    means_v = [r["mean_width_cm"] for r in detected]
    stds_v = [r["std_within_cluster_cm"] for r in detected]
    x = np.arange(len(names))
    ax.bar(x, means_v, yerr=stds_v, capsize=5, color="#1f77b4",
           edgecolor="black", alpha=0.8)
    ax.axhline(GT_CLEAR, color="green", ls=":", lw=2, label=f"GT clear={GT_CLEAR}")
    ax.axhline(GT_ROUGH, color="orange", ls="--", lw=1.5, label=f"GT rough={GT_ROUGH}")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Width (cm)"); ax.set_title("Mean Width per Run")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # (b) box plot per run
    ax = axes[1]
    box_data = [np.array(r["filtered_widths_cm"]) for r in detected]
    bp = ax.boxplot(box_data, tick_labels=names, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4"); patch.set_alpha(0.5)
    ax.axhline(GT_CLEAR, color="green", ls=":", lw=2, label=f"GT clear={GT_CLEAR}")
    ax.set_ylabel("Width (cm)"); ax.set_title("Per-Frame Widths (filtered)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # (c) summary table
    ax = axes[2]
    ax.axis("off")
    rows_t = [["Run", "Mean", "Median", "Std", "Err/C", "Err/R", "N"]]
    for r in detected:
        rows_t.append([r["run_id"],
                       f"{r['mean_width_cm']:.1f}", f"{r['median_width_cm']:.1f}",
                       f"{r['std_within_cluster_cm']:.1f}",
                       f"{r['error_vs_clear_cm']:+.1f}",
                       f"{r['error_vs_rough_cm']:+.1f}",
                       str(r["n_frames_in_cluster"])])
    if len(detected) >= 2:
        rows_t.append(["AGG",
                       f"{np.mean(means_v):.1f}", "",
                       f"{np.std(means_v):.1f}",
                       f"{np.mean(maes_c):.1f}",
                       f"{np.mean(maes_r):.1f}", ""])
    for r in runs:
        if r["status"] != "detected":
            rows_t.append([r["run_id"], r["status"], "", "", "", "", ""])
    tbl = ax.table(cellText=rows_t[1:], colLabels=rows_t[0], loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.5)
    ax.set_title("Cross-Run Summary", fontsize=11, pad=20)

    plt.suptitle(f"Sprint 9: Cross-Run Validation (0 param changes) | "
                 f"{np.datetime64('today')}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "sprint9_cross_run.png"), dpi=150, facecolor="white")
    plt.close()
    print(f"\n  Viz: {FIG_DIR / 'sprint9_cross_run.png'}")


if __name__ == "__main__":
    main()
