#!/usr/bin/env python3
"""Sprint 11: 50-trial median-aggregated pipeline.

The published algorithm is: run the full Sprint 8.10 pipeline N times
(default 50) on the same rosbag, take the median of detected widths.
This absorbs the bimodal RANSAC noise discovered in Sprint 10 — the
median is robust to the ~20% of trials that lock onto a secondary
structural gap (~185 cm) rather than the true doorway.

Usage:
    python scripts/run_aggregated_pipeline.py ~/autopass/bags/run_01
    python scripts/run_aggregated_pipeline.py ~/autopass/bags/run_02 -n 50
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
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# ── pipeline params (UNCHANGED from run_01 development) ─────────────
ROI_MIN, ROI_MAX = np.array([-5,-5,-0.5]), np.array([5,5,2.5])
VOXEL_SIZE = 0.03; SOR_NB, SOR_STD = 20, 2.0
MAX_HORIZ = 4; PLANE_DIST = 0.05; MIN_PLANE_PTS = 150
MAX_WALLS = 4; WALL_DIST = 0.03; WALL_NZ = 0.2; WALL_ITERS = 5000; WALL_MIN = 100
BIN_SIZE = 0.05; MIN_DENS = 5; REOCC = 2; GAP_RANGE = (0.6, 2.0)
MERGE_DOT = 0.95; MERGE_PERP = 0.30; MERGE_OVERLAP = 0.50
DBSCAN_EPS = 0.30; DBSCAN_MIN = 3; SIGMA_REJECT = 2.0
TRANSIT_THRESHOLD_M = 0.50
FRAME_START, FRAME_END = 20, 50
GT_CLEAR, GT_ROUGH = 85.3, 91.4

POINTFIELD_DTYPES = {1:("B",1),2:("b",1),3:("H",2),4:("h",2),
                     5:("I",4),6:("i",4),7:("f",4),8:("d",8)}


# ── bag reading (one-time) ───────────────────────────────────────────
def parse_pc2(msg):
    fields = {f.name: (f.offset, POINTFIELD_DTYPES[f.datatype][0]) for f in msg.fields}
    ps, data, n = msg.point_step, bytes(msg.data), msg.width * msg.height
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        b = i * ps
        for j, ax in enumerate(("x","y","z")):
            pts[i,j] = struct.unpack_from(fields[ax][1], data, b+fields[ax][0])[0]
    return pts[np.isfinite(pts).all(axis=1)]

def load_bag(bag_path):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    indices = list(range(FRAME_START, FRAME_END+1))
    want = set(indices); raw = {}; odom = []; pc_times = []
    with AnyReader([bag_path], default_typestore=ts) as reader:
        for conn, t, rd in reader.messages(
                [c for c in reader.connections if c.topic == "/odom"]):
            msg = reader.deserialize(rd, conn.msgtype)
            odom.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y))
        for idx, (conn, t, rd) in enumerate(reader.messages(
                [c for c in reader.connections if c.topic == "/point_cloud2"])):
            pc_times.append(t)
            if idx in want:
                raw[idx] = parse_pc2(reader.deserialize(rd, conn.msgtype))
                want.discard(idx)
                if not want: break
    if not odom or not pc_times:
        return {}, None
    odom_ts = np.array([o[0] for o in odom])
    traj = {}
    for fi in indices:
        if fi < len(pc_times):
            oi = np.argmin(np.abs(odom_ts - pc_times[fi]))
            traj[fi] = (odom[oi][1], odom[oi][2])
    pcds = {}
    for fi, pts in raw.items():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(ROI_MIN, ROI_MAX))
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        pcd, _ = pcd.remove_statistical_outlier(SOR_NB, SOR_STD)
        pcds[fi] = pcd
    traj_pts = np.array([traj[fi] for fi in sorted(traj)])
    return pcds, traj_pts


# ── pipeline stages ──────────────────────────────────────────────────
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
                    conf = min(1.0,min(counts[gs-1],counts[ge])/med) if med else 0.5
                    center = c + (rl+rr)/2*d
                    gaps.append(dict(gap_center_xy=center.round(4).tolist(),
                                     width_m=round(rr-rl,4), confidence=round(conf,3)))
    return gaps

def run_trial(pcds, traj_pts):
    all_gaps = []
    for fidx in sorted(pcds):
        walls = segment_walls(pcds[fidx])
        fg = detect_gaps(walls)
        for g in fg: g["frame"] = fidx
        all_gaps.extend(fg)
    if not all_gaps:
        return None
    centers = np.array([g["gap_center_xy"] for g in all_gaps])
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(centers)
    clusters = {}
    for i, lab in enumerate(labels):
        if lab >= 0: clusters.setdefault(lab, []).append(i)
    if not clusters:
        return None
    best, best_dist = None, float("inf")
    for cid, idxs in clusters.items():
        ws = np.array([all_gaps[i]["width_m"]*100 for i in idxs])
        cx = np.mean([all_gaps[i]["gap_center_xy"][0] for i in idxs])
        cy = np.mean([all_gaps[i]["gap_center_xy"][1] for i in idxs])
        md = float(np.linalg.norm(traj_pts - np.array([cx,cy]), axis=1).min())
        if md < TRANSIT_THRESHOLD_M and md < best_dist:
            best = ws; best_dist = md
    if best is None:
        return None
    raw = best
    med = np.median(raw); std = raw.std()
    filt = raw[np.abs(raw - med) <= SIGMA_REJECT * std]
    return round(float(filt.mean()), 2) if len(filt) else round(float(raw.mean()), 2)


# ── main ─────────────────────────────────────────────────────────────
def run_aggregated(bag_path, n_trials, run_id):
    print(f"\n  Loading {run_id}...")
    pcds, traj_pts = load_bag(bag_path)
    if not pcds:
        print(f"  {run_id}: no data")
        return None
    print(f"  {len(pcds)} frames, {len(traj_pts)} trajectory poses")

    widths = []
    t0 = time.time()
    for i in range(n_trials):
        w = run_trial(pcds, traj_pts)
        widths.append(w)
        tag = f"{w:.1f}cm" if w else "none"
        if (i+1) % 10 == 0 or i == 0:
            print(f"    trial {i+1:3d}/{n_trials}: {tag}")
    elapsed = time.time() - t0

    detected = [w for w in widths if w is not None]
    arr = np.array(detected) if detected else np.array([])
    n_det = len(detected)
    med_val = float(np.median(arr)) if n_det else None
    main_mode = np.sum(np.abs(arr - np.median(arr)) <= 50) if n_det else 0

    result = dict(
        run_id=run_id, n_trials=n_trials, n_detected=n_det,
        detection_rate=round(n_det/n_trials, 2),
        trial_widths_cm=[round(w, 2) if w else None for w in widths],
        median_cm=round(med_val, 2) if med_val else None,
        mean_cm=round(float(arr.mean()), 2) if n_det else None,
        std_cm=round(float(arr.std()), 2) if n_det else None,
        min_cm=round(float(arr.min()), 2) if n_det else None,
        max_cm=round(float(arr.max()), 2) if n_det else None,
        iqr_low=round(float(np.percentile(arr, 25)), 2) if n_det else None,
        iqr_high=round(float(np.percentile(arr, 75)), 2) if n_det else None,
        fraction_main_mode=round(main_mode / n_det, 2) if n_det else 0,
        error_vs_clear=round(med_val - GT_CLEAR, 2) if med_val else None,
        error_vs_rough=round(med_val - GT_ROUGH, 2) if med_val else None,
        elapsed_s=round(elapsed, 1),
    )
    print(f"  {run_id}: median={result['median_cm']}cm, "
          f"IQR=[{result['iqr_low']},{result['iqr_high']}], "
          f"err_clear={result['error_vs_clear']:+.1f}cm, "
          f"main_mode={result['fraction_main_mode']:.0%} ({elapsed:.0f}s)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="*",
                    default=[str(Path.home()/"autopass"/"bags"/r)
                             for r in ["run_01", "run_02"]])
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()

    print("=" * 60)
    print(f"Sprint 11: {args.n}-Trial Median-Aggregated Pipeline")
    print("=" * 60)

    all_results = []
    for bag in args.bags:
        bp = Path(bag)
        rid = bp.name
        result = run_aggregated(bp, args.n, rid)
        if result:
            all_results.append(result)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_DIR / f"aggregated_{rid}.json", "w") as f:
                json.dump(result, f, indent=2)

    # Cross-run summary
    valid = [r for r in all_results if r["median_cm"] is not None]
    print(f"\n{'='*65}")
    print(f"  {'Run':<8s} {'Median':>8s} {'IQR':>16s} {'Main%':>6s} "
          f"{'Err/C':>7s} {'Err/R':>7s}")
    print(f"  {'-'*60}")
    for r in valid:
        print(f"  {r['run_id']:<8s} {r['median_cm']:>7.1f}  "
              f"[{r['iqr_low']:>6.1f}, {r['iqr_high']:>6.1f}]  "
              f"{r['fraction_main_mode']:>5.0%}  "
              f"{r['error_vs_clear']:>+6.1f}  {r['error_vs_rough']:>+6.1f}")
    if len(valid) >= 2:
        meds = np.array([r["median_cm"] for r in valid])
        maes_c = np.array([abs(r["error_vs_clear"]) for r in valid])
        maes_r = np.array([abs(r["error_vs_rough"]) for r in valid])
        print(f"  {'-'*60}")
        print(f"  {'AGG':<8s} {meds.mean():>7.1f}  "
              f"{'':>16s}  {'':>6s}  {maes_c.mean():>+6.1f}  {maes_r.mean():>+6.1f}")
        print(f"\n  Mean of medians:  {meds.mean():.2f} cm")
        print(f"  Std of medians:   {meds.std():.2f} cm")
        print(f"  MAE vs clear:     {maes_c.mean():.2f} cm")
        print(f"  MAE vs rough:     {maes_r.mean():.2f} cm")

    # Plot
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) side-by-side histograms
    ax = axes[0]
    colors = ["#1f77b4", "#ff7f0e"]
    for i, r in enumerate(valid):
        ws = [w for w in r["trial_widths_cm"] if w is not None]
        ax.hist(ws, bins=20, alpha=0.5, color=colors[i % 2], edgecolor="black",
                label=f"{r['run_id']} (med={r['median_cm']:.1f})")
    ax.axvline(GT_CLEAR, color="green", ls=":", lw=2, label=f"GT clear={GT_CLEAR}")
    ax.set_xlabel("Width (cm)"); ax.set_ylabel("Count")
    ax.set_title(f"Trial Distributions (N={args.n} each)"); ax.legend(fontsize=7)

    # (b) bar chart: median with IQR
    ax = axes[1]
    names = [r["run_id"] for r in valid]
    meds_v = [r["median_cm"] for r in valid]
    lo = [r["median_cm"] - r["iqr_low"] for r in valid]
    hi = [r["iqr_high"] - r["median_cm"] for r in valid]
    x = np.arange(len(names))
    ax.bar(x, meds_v, yerr=[lo, hi], capsize=8, color="#2ca02c",
           edgecolor="black", alpha=0.8)
    ax.axhline(GT_CLEAR, color="green", ls=":", lw=2, label=f"GT clear={GT_CLEAR}")
    ax.axhline(GT_ROUGH, color="orange", ls="--", lw=1.5, label=f"GT rough={GT_ROUGH}")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Width (cm)"); ax.set_title("Median ± IQR per Run")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # (c) summary table
    ax = axes[2]; ax.axis("off")
    rows_t = [["Run", "N", "Median", "IQR", "Main%", "Err/C", "Err/R"]]
    for r in valid:
        rows_t.append([r["run_id"], str(r["n_detected"]),
                       f"{r['median_cm']:.1f}",
                       f"[{r['iqr_low']:.0f},{r['iqr_high']:.0f}]",
                       f"{r['fraction_main_mode']:.0%}",
                       f"{r['error_vs_clear']:+.1f}",
                       f"{r['error_vs_rough']:+.1f}"])
    if len(valid) >= 2:
        rows_t.append(["AGG", "", f"{meds.mean():.1f}", "",
                       "", f"{maes_c.mean():.1f}", f"{maes_r.mean():.1f}"])
    tbl = ax.table(cellText=rows_t[1:], colLabels=rows_t[0], loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.6)
    ax.set_title("Cross-Run Summary", fontsize=11, pad=20)

    plt.suptitle(f"Sprint 11: Median-Aggregated Pipeline | "
                 f"{np.datetime64('today')}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "sprint11_aggregated.png"), dpi=150, facecolor="white")
    plt.close()
    print(f"\n  Viz: {FIG_DIR / 'sprint11_aggregated.png'}")


if __name__ == "__main__":
    main()
