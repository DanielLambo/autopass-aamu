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

### segment.py

Two-phase RANSAC plane segmentation on the cleaned point cloud:

1. **Phase 1 — Horizontal planes**: Standard iterative RANSAC extracts floor/ceiling planes (|nz| > 0.9)
2. **Phase 2 — Wall planes**: Custom constrained RANSAC on remaining points — only accepts planes with |nz| < 0.2, ensuring vertical wall surfaces are found even when scattered

Horizontal planes are sub-classified by height:
- Lowest = `floor_main`
- Within 0.1m of floor = `floor_secondary` (e.g., bathroom threshold)
- Above 1.0m = `ceiling`
- Otherwise = `horizontal_other`

**Usage:**
```bash
python scripts/segment.py
```

**Parameters (edit at top of file):**
- `MAX_HORIZ_PLANES` / `MAX_WALL_PLANES` — max planes per phase
- `DISTANCE_THRESHOLD` — RANSAC inlier distance for horizontal planes (0.05m)
- `WALL_DIST_THRESHOLD` — tighter threshold for walls (0.03m)
- `WALL_MAX_NZ` — normal constraint for wall detection (0.2)
- `MIN_PLANE_POINTS` / `WALL_MIN_POINTS` — minimum inliers to keep a plane

**Outputs:**
- `data/planes/` — individual PCD files per labeled plane + remaining.pcd
- `data/segmentation_result.png` — 4-panel visualization (3D, top-down, two side views)

**Note:** The Go2 LiDAR captures walls at oblique angles during doorway transit, so wall planes are smaller (400-600 pts) than floor planes (9000-13000 pts). The constrained RANSAC approach is necessary because standard RANSAC preferentially finds horizontal planes and starves wall detection.

## Running

```bash
source autopass_env/bin/activate
python scripts/extract_frame.py
python scripts/preprocess.py
python scripts/segment.py
```

## Dependencies

- open3d
- numpy
- matplotlib
- rosbags
