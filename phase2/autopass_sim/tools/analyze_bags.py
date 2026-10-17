"""Width error vs distance, angle and density on REAL Go2 bags (Report 2 revision, point 1).

For each feature you name (approximate centre in the odom frame + tape-measured width), every
/point_cloud2 message (the Go2's accumulated 5 cm voxel map) is cut to a 1.6 m window around the
feature, the wall axis is found by PCA, and the opening is the widest gap between voxels along
that axis near the centre. For each message we record

  d     robot-to-feature distance (m), from /odom at the message time
  psi   near-jamb sightline angle (deg), same definition as the simulator
  rho   voxels within 10 cm of each edge (min of the two sides), 0.3-1.9 m height band
  w     measured clear width (cm) = gap between the innermost voxel centres on each side
  err   w - tape-measured width (cm)

Run on the TensorBook (system Python, ROS 2 sourced):
  source /opt/ros/humble/setup.bash
  /usr/bin/python3 tools/analyze_bags.py ~/autopass/bags/run_01 ~/autopass/bags/run_02 \
      --feature door:1.2,0.4:91.4 --feature passage_a:-2.0,1.1:118.9 --feature passage_b:0.5,-2.6:97.5 \
      --out results/bag_widths
  (x,y for each feature: read them off the plan view from tools/render_bag.py; ~0.3 m accuracy is enough)
Writes <out>.csv (one row per message per feature) and <out>.png, and prints a summary table.
"""
import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOX = 0.05


def read_bag(path):
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from sensor_msgs_py import point_cloud2 as pc2
    r = SequentialReader()
    r.open(StorageOptions(uri=os.path.expanduser(path), storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    r.set_filter(StorageFilter(topics=["/point_cloud2", "/odom"]))
    odom, clouds = [], []
    while r.has_next():
        topic, data, t = r.read_next()
        if topic == "/odom":
            p = deserialize_message(data, Odometry).pose.pose.position
            odom.append((t / 1e9, p.x, p.y))
        else:
            m = deserialize_message(data, PointCloud2)
            clouds.append((t / 1e9, pc2.read_points_numpy(m, field_names=("x", "y", "z")).astype(float)))
    return np.array(odom), clouds


def measure(P, centre, win=0.8, search=0.9):
    """Width (cm) of the opening nearest `centre`, plus edge positions and density."""
    c = np.asarray(centre)
    m = (np.abs(P[:, 0] - c[0]) < win) & (np.abs(P[:, 1] - c[1]) < win) & (P[:, 2] > 0.3) & (P[:, 2] < 1.9)
    Q = P[m, :2]
    if len(Q) < 20:
        return None
    mu = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - mu, full_matrices=False)
    axis, normal = vt[0], vt[1]
    u = (Q - c) @ axis                                   # coordinate along the wall, 0 at the centre
    s = np.unique(np.round(u / VOX * 2) / 2 * VOX)       # distinct positions (half-voxel bins)
    left, right = s[s < 0], s[s > 0]
    if len(left) == 0 or len(right) == 0:
        return None
    gL, gR = left.max(), right.min()
    if gR - gL > 2 * search:
        return None
    w = (gR - gL) * 100                                 # voxel centres sit on the edge on average
    rhoL = int((np.abs(u - gL) <= 0.10).sum())
    rhoR = int((np.abs(u - gR) <= 0.10).sum())
    eL, eR = c + gL * axis, c + gR * axis
    return w, eL, eR, normal, min(rhoL, rhoR)


def view(pos, eL, eR, normal):
    mid = (eL + eR) / 2
    n = normal if np.dot(pos - mid, normal) > 0 else -normal   # normal pointing at the robot
    d = float(np.linalg.norm(pos - mid))
    half = np.linalg.norm(eR - eL) / 2
    along = np.dot(pos - mid, (eR - eL) / (2 * half))
    depth = max(float(np.dot(pos - mid, n)), 1e-3)
    psi = float(np.degrees(np.arctan(max(0.0, abs(along) - half) / depth)))
    return d, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--feature", action="append", required=True, help="name:x,y:true_cm")
    ap.add_argument("--out", default="results/bag_widths")
    a = ap.parse_args()
    feats = []
    for f in a.feature:
        name, xy, true = f.split(":")
        feats.append((name, np.array([float(v) for v in xy.split(",")]), float(true)))
    rows = []
    for bag in a.bags:
        odom, clouds = read_bag(bag)
        if len(odom) == 0:
            print(f"{bag}: no /odom, skipped"); continue
        for t, P in clouds:
            pos = np.array([np.interp(t, odom[:, 0], odom[:, 1]), np.interp(t, odom[:, 0], odom[:, 2])])
            for name, c, true in feats:
                r = measure(P, c)
                if r is None:
                    continue
                w, eL, eR, normal, rho = r
                d, psi = view(pos, eL, eR, normal)
                rows.append(dict(bag=os.path.basename(bag.rstrip("/")), t=round(t, 2), feature=name,
                                 true_cm=true, w_cm=round(w, 2), err_cm=round(w - true, 2),
                                 d=round(d, 3), psi=round(psi, 2), rho=rho))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"{'bag':8} {'feature':10} {'n':>4} {'true':>6} {'mean w':>7} {'MAE':>5} {'sd':>5} {'d range':>11} {'psi range':>11} {'rho':>7} {'distinct w':>10}")
    for bag in sorted({r["bag"] for r in rows}):
        for name, _, true in feats:
            R = [r for r in rows if r["bag"] == bag and r["feature"] == name]
            if not R:
                continue
            w = np.array([r["w_cm"] for r in R]); e = w - true
            d = [r["d"] for r in R]; p = [r["psi"] for r in R]; rho = [r["rho"] for r in R]
            print(f"{bag:8} {name:10} {len(R):4d} {true:6.1f} {w.mean():7.1f} {np.abs(e).mean():5.1f} {w.std():5.1f} "
                  f"{min(d):4.1f}-{max(d):4.1f} m {min(p):4.0f}-{max(p):3.0f} deg {int(np.median(rho)):7d} {len(np.unique(w)):10d}")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
    for name, _, _ in feats:
        R = [r for r in rows if r["feature"] == name]
        for k, key in enumerate(["d", "psi", "rho"]):
            ax[k].scatter([r[key] for r in R], [r["err_cm"] for r in R], s=10, label=name, alpha=0.7)
    for k, lab in enumerate(["distance d (m)", "near-jamb angle psi (deg)", "edge density rho (voxels)"]):
        ax[k].set_xlabel(lab); ax[k].axhline(0, color="k", lw=0.6)
        for q in (-5, 5):
            ax[k].axhline(q, color="gray", lw=0.5, ls=":")
    ax[0].set_ylabel("width error (cm)"); ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Real Go2 data (/point_cloud2, 5 cm voxel map): width error vs viewing geometry", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out + ".png")
    print("wrote", a.out + ".csv", a.out + ".png")


if __name__ == "__main__":
    main()
