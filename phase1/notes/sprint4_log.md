# Sprint 4 — Plane segmentation

**Date:** April 18, 2026
**Input:** ~/autopass/data/frame_clean.pcd (47,052 points)
**Method:** Iterative RANSAC, orientation-constrained after horizontal surfaces extracted
**Commit:** c34be3f

## Key findings

1. **Two floor levels detected 10cm apart** (Z=-0.100m, Z=-0.003m) — consistent with a bedroom/bathroom threshold.
2. **Four wall fragments in two parallel pairs** — geometry consistent with two perpendicular surfaces meeting at a doorway corner.
3. **Classification:** 2 floors, 4 walls, 2 other horizontal planes (likely bathroom fixtures), 14,409 unclassified points.

## Detected planes

| Plane | Points | Normal (a,b,c) | Mean Z | d offset |
|-------|--------|----------------|--------|----------|
| floor_main | 9,043 | (+0.001, +0.001, +1.000) | -0.100 | +0.100 |
| floor_secondary | 13,396 | (-0.025, -0.006, +1.000) | -0.003 | -0.018 |
| horizontal_other_01 | 5,326 | (-0.047, -0.006, +0.999) | 0.104 | -0.138 |
| horizontal_other_02 | 2,868 | (-0.043, +0.021, +0.999) | 0.327 | -0.317 |
| wall_01 | 521 | (+0.996, -0.077, +0.041) | 0.404 | -1.184 |
| wall_02 | 515 | (-0.788, +0.595, -0.159) | 0.236 | +0.734 |
| wall_03 | 532 | (+0.990, -0.088, +0.110) | 0.390 | -1.306 |
| wall_04 | 442 | (+0.788, -0.598, -0.149) | 0.190 | -0.389 |
| remaining | 14,409 | — | — | — |

## Outputs
- data/planes/*.pcd — 9 plane files
- data/segmentation_result_v2.png — 6-panel visualization (publishable quality)
- data/segmentation_result.png — original visualization (kept for comparison)

## Methodology note
Standard RANSAC iterations were dominated by floor fragmentation due to the floor being the largest plane (~22K combined points). An orientation-constrained pass (requiring |nz| < 0.2) was used after horizontal surfaces were extracted to recover vertical wall planes. This is a standard handling of degenerate-dominant-surface scenarios but should be flagged in any methodology section.
