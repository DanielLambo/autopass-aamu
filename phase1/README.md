# Phase 1: doorway width from real Go2 LiDAR

Recorded on the Go2 EDU over ROS 2 Humble (`/point_cloud2`, `/imu`, `/odom`), processed offline with Open3D.
Test doorway: clear width 85.3 cm, rough opening 91.4 cm (`ground_truth.txt`). Bags live in
`~/autopass/bags/` and are not in the repo.

## Pipeline

| Sprint | Step | Script |
|---|---|---|
| 1–2 | LiDAR + ROS 2 bring-up, bag recording, tape ground truth | `notes/go2_session_checklist.md`, `notes/sprint2_log.md` |
| 3 | Frame extraction; ROI crop, voxel downsample, outlier removal | `scripts/extract_frame.py`, `scripts/preprocess.py` |
| 4 | Two-phase RANSAC: floors first, then orientation-constrained walls | `scripts/segment.py`, `scripts/visualize_segments.py` |
| 6–6.6 | Doorway detection on wall planes (segment gaps, then density, then wall dedup) | `scripts/detect_doorways.py`, `_v2.py`, `_v3.py` |
| 7 | Agent profiles and the passability classifier | `profiles/*.json`, `scripts/passability.py`, `tests/test_passability.py` |
| 8.5 | Sub-bin edge refinement (removes histogram quantization bias) | `scripts/detect_doorways_v4.py` |
| 8.6 | Evaluation against both ground-truth widths | `scripts/evaluate.py` |
| 8.7–8.10 | Multi-frame tracking, gap clustering, odometry-based transit selection | `scripts/multi_frame_pipeline.py` |
| 9 | Held-out cross-run validation, parameters frozen | `scripts/run_full_pipeline.py` |
| 10 | RANSAC stability over 50 repeated runs | `scripts/ransac_stability.py` |
| 11 | 50-trial median aggregation | `scripts/run_aggregated_pipeline.py` |

The older detector versions are kept because each sprint's figures and results were produced with them.

## Results

| Run | Detection rate (50 trials) | Median width | Error vs clear width | IQR |
|---|---|---|---|---|
| run_01 | 100% | 96.1 cm | +10.8 cm | 87.4–104.8 cm |
| run_02 | 36% | 82.6 cm | −2.7 cm | 70.8–93.1 cm |

`results/aggregated_run_0{1,2}.json`. The single-pass cross-run check (`results/cross_run_evaluation.json`)
was worse, off by up to 92 cm on run_01. Widths read while walking past a door are not reliable enough for a
pass/fail call near the limit: oblique views and partly hidden jambs bias the estimate, and repeating the
same view does not remove the bias. Phase 2 addresses this by planning where to look from.

## Layout

```
scripts/    pipeline scripts (run from this folder: python scripts/<name>.py)
data/       extracted frames, segmented planes, detected doorways per detector version
figures/    one figure per sprint
results/    evaluation outputs (JSON/CSV)
profiles/   agent profiles: ADA default, wheelchair, walker
tests/      passability classifier tests
notes/      session checklist and sprint logs
```

## Setup

ROS 2 Humble's system Python for reading bags (`rosbags`), plus `open3d`, `numpy`, `matplotlib`.
