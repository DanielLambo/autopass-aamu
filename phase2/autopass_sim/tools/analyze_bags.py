"""Width error of tape-measured features (doors, passages) in REAL Go2 bags.

For every /point_cloud2 message (a rolling 5 cm voxel map in `odom`) and every feature:
  1. cut a square window (default 1.6 m) around the feature, keep 0.3 < z < 1.9 m;
  2. find the wall axis by PCA of the window's xy points;
  3. keep points within a slab around the wall line (drops things seen through the opening);
  4. width = gap between the innermost voxel centres on each side of the feature centre.
Also records the robot's view geometry at the message time (odom interpolated):
  d   = distance to the feature centre,
  psi = atan(max(0, |along| - w/2) / depth), the near-jamb sightline angle (deg),
  rho = voxels within 10 cm of each edge, the smaller of the two sides.

Run with system Python and ROS 2 sourced (not the venv):
  source /opt/ros/humble/setup.bash
  /usr/bin/python3 tools/analyze_bags.py ~/autopass/bags/run_01 ~/autopass/bags/run_02 \
      --feature A:-0.20,0.12 --out results/bag_widths_A          # no tape width: no error columns
  add :true_cm to a feature (e.g. door:X,Y:91.4) to get bias and MAE against the tape measurement.
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOXEL = 0.05
INK, GRAY, MAROON = "#222222", "#5A5A5A", "#7A1E2C"
COLORS = ["#7A1E2C", "#1F4E79", "#2E7D32", "#B26A00", "#5E35B1", "#00838F"]


def load_messages(path):
    """Same reader pattern as render_bag.load_bag, but keeps each message separately.

    Returns clouds = [(t, xyz)] and odom = array of (t, x, y, z), both on the bag clock.
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from sensor_msgs_py import point_cloud2 as pc2
    r = SequentialReader()
    r.open(StorageOptions(uri=os.path.expanduser(path), storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    r.set_filter(StorageFilter(topics=["/point_cloud2", "/odom"]))
    clouds, odom = [], []
    while r.has_next():
        topic, data, t = r.read_next()
        if topic == "/odom":
            p = deserialize_message(data, Odometry).pose.pose.position
            odom.append((t / 1e9, p.x, p.y, p.z))
        else:
            m = deserialize_message(data, PointCloud2)
            xyz = pc2.read_points_numpy(m, field_names=("x", "y", "z")).astype(float)
            clouds.append((t / 1e9, xyz))
    return clouds, np.array(odom)


def robot_xy(odom, t):
    return np.array([np.interp(t, odom[:, 0], odom[:, 1]), np.interp(t, odom[:, 0], odom[:, 2])])


def parse_feature(s):
    """name:x,y[:true_cm] -> (name, centre xy in m, true width in m or nan if not given)."""
    parts = s.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"expected name:x,y[:true_cm], got {s!r}")
    x, y = map(float, parts[1].split(","))
    w = float(parts[2]) / 100.0 if len(parts) == 3 else np.nan
    return parts[0], np.array([x, y]), w


def measure(P, c, w_true, rob, half=0.8, z_lo=0.3, z_hi=1.9, slab=0.15, edge_band=0.10, k=1, min_pts=3):
    """Width and view geometry for one message and one feature. Returns a dict, width nan if not seen."""
    m = (np.abs(P[:, 0] - c[0]) < half) & (np.abs(P[:, 1] - c[1]) < half) & (P[:, 2] > z_lo) & (P[:, 2] < z_hi)
    Q = P[m, :2]
    out = dict(n_win=len(Q), width=np.nan, rho=0, rho_l=0, rho_r=0, axis_deg=np.nan)
    rel = rob - c
    out["d"] = float(np.hypot(*rel))
    if len(Q) < 2 * min_pts:
        out.update(psi=np.nan, along=np.nan, depth=np.nan)
        return out
    X = Q - Q.mean(0)
    evals, evecs = np.linalg.eigh(X.T @ X)
    u = evecs[:, 1]                                    # wall direction (largest variance)
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):
        u = -u                                         # fixed sign so "left/right" is stable
    n = np.array([-u[1], u[0]])
    a, b = (Q - c) @ u, (Q - c) @ n
    b0 = np.median(b)                                  # the wall line may sit off the given centre
    keep = np.abs(b - b0) < slab
    a = a[keep]
    L, R = np.sort(a[a < 0]), np.sort(a[a > 0])
    along, depth = float(rel @ u), float(abs(rel @ n - b0))
    out.update(along=along, depth=depth, axis_deg=float(np.degrees(np.arctan2(u[1], u[0]))), psi=np.nan)
    if len(L) < min_pts or len(R) < min_pts:
        return out
    el, er = L[-k], R[k - 1]
    rl = int(np.sum((L >= el - edge_band) & (L <= el)))
    rr = int(np.sum((R <= er + edge_band) & (R >= er)))
    out.update(width=float(er - el), rho=min(rl, rr), rho_l=rl, rho_r=rr)
    # psi uses the tape width when given, otherwise this message's measured width
    w = w_true if np.isfinite(w_true) else out["width"]
    out["psi"] = float(np.degrees(np.arctan2(max(0.0, abs(along) - w / 2), depth)))
    return out


def spearman(x, y):
    """Spearman rank correlation and two-sided p (scipy if present, else rho only)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan, np.nan
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x[ok], y[ok])
        return float(r.statistic), float(r.pvalue)
    except ImportError:
        rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
        return float(np.corrcoef(rx, ry)[0, 1]), np.nan


def summarize(rows):
    """Per-bag, per-feature summary over messages where the width was measured."""
    out = []
    keys = sorted({(r["bag"], r["feature"]) for r in rows}, key=lambda k: (k[0], k[1]))
    for bag, feat in keys:
        R = [r for r in rows if r["bag"] == bag and r["feature"] == feat]
        V = [r for r in R if np.isfinite(r["width"])]
        s = dict(bag=bag, feature=feat, true_cm=R[0]["true_cm"], n=len(V), n_msgs=len(R))
        if V:
            w = np.array([r["width"] for r in V]) * 100
            e = np.array([r["err_cm"] for r in V])
            d = np.array([r["d"] for r in V]); p = np.array([r["psi"] for r in V])
            rho = np.array([r["rho"] for r in V], float)
            uw = np.unique(np.round(w, 1))
            s.update(mean_cm=w.mean(), median_cm=float(np.median(w)), sd_cm=w.std(ddof=1) if len(w) > 1 else 0.0,
                     w_min=w.min(), w_max=w.max(), mae_cm=np.abs(e).mean(), bias_cm=e.mean(),
                     d_min=d.min(), d_max=d.max(), psi_min=np.nanmin(p), psi_max=np.nanmax(p),
                     rho_med=float(np.median(rho)), n_widths=len(uw), widths=uw,
                     step_cm=float(np.median(np.diff(uw))) if len(uw) > 1 else np.nan,
                     r_d=spearman(d, w), r_psi=spearman(p, w), r_rho=spearman(rho, w))
        out.append(s)
    return out


def print_summary(S):
    hdr = f"{'bag':8s} {'feature':10s} {'true':>6s} {'n':>4s} {'mean':>6s} {'SD':>5s} {'range':>11s} {'bias':>6s} {'MAE':>5s} " \
          f"{'d range (m)':>12s} {'psi range':>12s} {'rho~':>5s} {'#w':>3s}"
    print(hdr); print("-" * len(hdr))
    for s in S:
        true = f"{s['true_cm']:6.1f}" if np.isfinite(s["true_cm"]) else f"{'-':>6s}"
        if s["n"] == 0:
            print(f"{s['bag']:8s} {s['feature']:10s} {true} {0:4d}  (no message saw both edges, of {s['n_msgs']})")
            continue
        has_true = np.isfinite(s["true_cm"])
        bias = f"{s['bias_cm']:+6.1f}" if has_true else f"{'-':>6s}"
        mae = f"{s['mae_cm']:5.1f}" if has_true else f"{'-':>5s}"
        print(f"{s['bag']:8s} {s['feature']:10s} {true} {s['n']:4d} {s['mean_cm']:6.1f} {s['sd_cm']:5.1f} "
              f"{s['w_min']:5.1f}-{s['w_max']:<5.1f} {bias} {mae} {s['d_min']:5.2f}-{s['d_max']:<5.2f} "
              f"{s['psi_min']:5.1f}-{s['psi_max']:<5.1f} {s['rho_med']:5.1f} {s['n_widths']:3d}")
    print("widths and errors in cm; psi in deg; rho~ = median edge density (voxels); #w = distinct widths (1 mm bins)")
    print("\nDistinct widths and trend of width vs view geometry (Spearman r, p):")
    for s in S:
        if s["n"] == 0:
            continue
        tr = "  ".join(f"{k}: r={s[key][0]:+.2f} p={s[key][1]:.3g}" for k, key in (("d", "r_d"), ("psi", "r_psi"), ("rho", "r_rho")))
        print(f"  {s['bag']} {s['feature']}: widths (cm) {', '.join(f'{x:.1f}' for x in s['widths'])}; "
              f"median step {s['step_cm']:.1f} cm\n      {tr}")


def plot(rows, out):
    V = [r for r in rows if np.isfinite(r["width"])]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), dpi=160, sharey=True)
    fig.patch.set_facecolor("#FBFBF8")
    groups = sorted({(r["bag"], r["feature"]) for r in V})
    marks = {b: mk for b, mk in zip(sorted({g[0] for g in groups}), "os^Dv")}
    feats = sorted({g[1] for g in groups})
    use_err = all(np.isfinite(r["true_cm"]) for r in V)    # with no tape width, plot the width itself
    for bag, feat in groups:
        G = [r for r in V if r["bag"] == bag and r["feature"] == feat]
        e = [r["err_cm"] if use_err else r["width"] * 100 for r in G]
        kw = dict(s=16, alpha=0.75, marker=marks[bag], color=COLORS[feats.index(feat) % len(COLORS)],
                  label=f"{feat} ({bag})", edgecolors="none")
        ax[0].scatter([r["d"] for r in G], e, **kw)
        ax[1].scatter([r["psi"] for r in G], e, **kw)
        ax[2].scatter([r["rho"] for r in G], e, **kw)
    for a, xl in zip(ax, ["distance to feature d (m)", "near-jamb angle psi (deg)", "edge density rho (voxels, min side)"]):
        if use_err:
            a.axhline(0, color=GRAY, lw=0.8)
        a.set_xlabel(xl); a.grid(lw=0.3, alpha=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel("width error, measured - tape (cm)" if use_err else "measured width (cm)")
    ax[2].legend(fontsize=7, frameon=False, loc="best")
    what = "width error" if use_err else "width (no tape reference)"
    fig.suptitle(f"Real Go2 bags: per-message voxel-map {what} vs view geometry", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--feature", action="append", required=True, type=parse_feature,
                    help="name:x,y[:true_cm], centre of the opening in odom (repeatable); "
                         "without true_cm no error is computed and psi uses the measured width")
    ap.add_argument("--out", default="results/bag_widths", help="prefix for .csv and .png")
    ap.add_argument("--window", type=float, default=1.6, help="side of the square window (m)")
    ap.add_argument("--slab", type=float, default=0.15, help="half-thickness kept around the wall line (m)")
    ap.add_argument("--edge-band", type=float, default=0.10, help="band for edge density rho (m)")
    ap.add_argument("--k", type=int, default=1, help="use the k-th innermost voxel per side (1 = innermost)")
    a = ap.parse_args()

    rows = []
    for bag in a.bags:
        name = os.path.basename(os.path.normpath(bag))
        clouds, odom = load_messages(bag)
        print(f"{name}: {len(clouds)} /point_cloud2, {len(odom)} /odom")
        for i, (t, P) in enumerate(clouds):
            rob = robot_xy(odom, t)
            for fname, c, w in a.feature:
                r = measure(P, c, w, rob, half=a.window / 2, slab=a.slab, edge_band=a.edge_band, k=a.k)
                r.update(bag=name, feature=fname, msg=i, t=t - clouds[0][0], robot_x=rob[0], robot_y=rob[1],
                         true_cm=w * 100, err_cm=(r["width"] - w) * 100)
                rows.append(r)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["bag", "feature", "msg", "t", "true_cm", "width", "err_cm", "d", "psi", "along", "depth",
            "rho", "rho_l", "rho_r", "n_win", "axis_deg", "robot_x", "robot_y"]
    with open(a.out + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", a.out + ".csv")
    print_summary(summarize(rows))
    plot(rows, a.out + ".png")


if __name__ == "__main__":
    main()
