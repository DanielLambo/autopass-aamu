# Sprint 3: Preprocessing Pipeline

## Scripts

### extract_frame.py

Extracts a single PointCloud2 frame from a ROS2 bag file and saves it as a PCD file.

**Usage:**
```bash
python scripts/extract_frame.py
```

**Parameters (edit at top of file):**
- `BAG_PATH` — path to rosbag directory (default: `bags/run_01`)
- `FRAME_INDEX` — which frame to extract (default: 30)
- `OUTPUT_PATH` — output PCD file (default: `data/frame_raw.pcd`)

**Output:** `data/frame_raw.pcd`

### preprocess.py

Loads a raw PCD file and applies three cleaning stages:

1. **ROI crop** — keeps points within (-5, -5, -0.5) to (5, 5, 2.5) meters
2. **Voxel downsample** — reduces density with voxel_size=0.03 m
3. **Statistical outlier removal** — removes noise (nb_neighbors=20, std_ratio=2.0)

The cleaned cloud is height-colored (red=floor, blue=ceiling).

**Usage:**
```bash
python scripts/preprocess.py
```

**Parameters (edit at top of file):**
- `ROI_MIN` / `ROI_MAX` — bounding box for cropping
- `VOXEL_SIZE` — downsample resolution
- `SOR_NEIGHBORS` / `SOR_STD_RATIO` — outlier removal sensitivity

**Outputs:**
- `data/frame_clean.pcd` — cleaned point cloud
- `data/preprocess_comparison.png` — side-by-side raw vs cleaned visualization

## Running

```bash
source autopass_env/bin/activate
python scripts/extract_frame.py
python scripts/preprocess.py
```

## Dependencies

- open3d
- numpy
- matplotlib
- rosbags
