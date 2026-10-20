"""Render REAL Go2 data from a Phase 1 bag: the 5 cm voxel map plus the odometry trajectory.

Run on the TensorBook (system Python with ROS 2 sourced, not the venv):
  source /opt/ros/humble/setup.bash
  /usr/bin/python3 tools/render_bag.py ~/autopass/bags/run_01 --out results/bag_run_01.png
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRAY, MAROON = "#222222", "#5A5A5A", "#7A1E2C"


def load_bag(path):
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from sensor_msgs_py import point_cloud2 as pc2
    r = SequentialReader()
    r.open(StorageOptions(uri=os.path.expanduser(path), storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    r.set_filter(StorageFilter(topics=["/point_cloud2", "/odom"]))
    keys, traj, n_clouds = set(), [], 0
    while r.has_next():
        topic, data, t = r.read_next()
        if topic == "/odom":
            m = deserialize_message(data, Odometry)
            p = m.pose.pose.position
            traj.append((t / 1e9, p.x, p.y, p.z))
        else:
            m = deserialize_message(data, PointCloud2)
            xyz = pc2.read_points_numpy(m, field_names=("x", "y", "z"))
            keys.update(map(tuple, np.floor(xyz / 0.05).astype(np.int64)))
            n_clouds += 1
    P = (np.array(list(keys), float) + 0.5) * 0.05
    return P, np.array(traj), n_clouds


def render(P, traj, out, title, z_lo=-0.2, z_hi=2.2, margin=3.0):
    T = traj[:, 1:3]
    lo, hi = T.min(0) - margin, T.max(0) + margin
    m = (P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1]) & \
        (P[:, 2] > z_lo) & (P[:, 2] < z_hi)
    P = P[m]
    fig = plt.figure(figsize=(12.8, 5.6), dpi=200)
    fig.patch.set_facecolor("#FBFBF8")
    a1 = fig.add_subplot(1, 2, 1)
    above = P[:, 2] > 0.15                                   # drop the floor for the plan view
    a1.scatter(P[above, 0], P[above, 1], s=0.6, c=P[above, 2], cmap="viridis", vmin=0, vmax=z_hi)
    a1.plot(T[:, 0], T[:, 1], color=MAROON, lw=2, label="robot trajectory (/odom)")
    a1.plot(*T[0], "o", color=INK, ms=6); a1.text(T[0, 0], T[0, 1] - 0.3, "start", fontsize=8, ha="center")
    a1.plot(*T[-1], "s", color=INK, ms=6); a1.text(T[-1, 0], T[-1, 1] + 0.2, "end", fontsize=8, ha="center")
    a1.set_aspect("equal"); a1.set_xlabel("x (m, odom)"); a1.set_ylabel("y (m, odom)")
    a1.set_title("Plan view (points above 15 cm, colored by height)", fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    a2 = fig.add_subplot(1, 2, 2, projection="3d")
    sub = P[np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)]
    a2.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.3, c=sub[:, 2], cmap="viridis", vmin=0, vmax=z_hi, depthshade=False)
    a2.plot(T[:, 0], T[:, 1], np.full(len(T), 0.05), color=MAROON, lw=2)
    a2.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], (z_hi - z_lo) * 1.5))
    a2.view_init(elev=35, azim=-60)
    a2.set_xlabel("x (m)", fontsize=8); a2.set_ylabel("y (m)", fontsize=8); a2.set_zlabel("z (m)", fontsize=8)
    a2.tick_params(labelsize=7)
    a2.set_title("3D view", fontsize=10)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
    print("wrote", out, f"({len(P)} voxels shown)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P, traj, n = load_bag(a.bag)
    name = os.path.basename(os.path.normpath(a.bag))
    dur = traj[-1, 0] - traj[0, 0]
    render(P, traj, a.out, f"Real Go2 EDU data, {name}: 5 cm voxel map from /point_cloud2 ({n} messages, "
                           f"{len(P):,} voxels) and /odom trajectory ({dur:.0f} s)")


if __name__ == "__main__":
    main()
