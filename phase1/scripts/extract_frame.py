#!/usr/bin/env python3
"""Extract a single PointCloud2 frame from a rosbag and save as PCD."""

import struct
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

BAG_PATH = Path.home() / "autopass" / "bags" / "run_01"
OUTPUT_PATH = Path.home() / "autopass" / "data" / "frame_raw.pcd"
TOPIC = "/point_cloud2"
FRAME_INDEX = 30

# Map PointField datatype constants to struct format chars
POINTFIELD_DTYPES = {
    1: ("B", 1),  # UINT8
    2: ("b", 1),  # INT8
    3: ("H", 2),  # UINT16
    4: ("h", 2),  # INT16
    5: ("I", 4),  # UINT32
    6: ("i", 4),  # INT32
    7: ("f", 4),  # FLOAT32
    8: ("d", 8),  # FLOAT64
}


def parse_pointcloud2(msg):
    """Parse a PointCloud2 message into a numpy array of xyz points."""
    # Build field info from message fields
    fields = {}
    for field in msg.fields:
        fmt, size = POINTFIELD_DTYPES[field.datatype]
        fields[field.name] = (field.offset, fmt, size, field.count)

    if "x" not in fields or "y" not in fields or "z" not in fields:
        raise ValueError(f"PointCloud2 missing xyz fields. Available: {list(fields.keys())}")

    point_step = msg.point_step
    data = bytes(msg.data)
    n_points = msg.width * msg.height

    points = np.zeros((n_points, 3), dtype=np.float64)
    for i in range(n_points):
        base = i * point_step
        for j, axis in enumerate(("x", "y", "z")):
            offset, fmt, size, count = fields[axis]
            val = struct.unpack_from(fmt, data, base + offset)[0]
            points[i, j] = val

    # Filter out NaN/inf points
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    return points


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    print(f"Opening bag: {BAG_PATH}")
    with AnyReader([BAG_PATH], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == TOPIC]
        if not connections:
            print(f"ERROR: Topic '{TOPIC}' not found. Available topics:")
            for c in reader.connections:
                print(f"  {c.topic} ({c.msgtype})")
            sys.exit(1)

        print(f"Reading topic: {TOPIC}")
        frame_idx = 0
        target_msg = None
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            if frame_idx == FRAME_INDEX:
                target_msg = reader.deserialize(rawdata, conn.msgtype)
                break
            frame_idx += 1

        if target_msg is None:
            print(f"ERROR: Frame {FRAME_INDEX} not found (only {frame_idx} frames available)")
            sys.exit(1)

    print(f"Extracted frame {FRAME_INDEX}")
    print(f"  height={target_msg.height}, width={target_msg.width}, point_step={target_msg.point_step}")
    print(f"  Fields: {[f.name for f in target_msg.fields]}")

    points = parse_pointcloud2(target_msg)
    print(f"  Valid points: {len(points)}")

    if len(points) == 0:
        print("ERROR: No valid points in this frame")
        sys.exit(1)

    # Stats
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(f"  Bounds X: [{mins[0]:.3f}, {maxs[0]:.3f}]")
    print(f"  Bounds Y: [{mins[1]:.3f}, {maxs[1]:.3f}]")
    print(f"  Bounds Z: [{mins[2]:.3f}, {maxs[2]:.3f}]")

    # Create Open3D point cloud and save
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(OUTPUT_PATH), pcd)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
