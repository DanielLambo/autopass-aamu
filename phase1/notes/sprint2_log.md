# Sprint 2 — Recording session

**Date:** April 18, 2026
**Location:** Bedroom
**Runs recorded:** 3
**Phone video:** [yes/no — fill in]

## Run details

| Run | Duration | /point_cloud2 messages | /imu messages | /odom messages | Size |
|-----|----------|------------------------|---------------|----------------|------|
| run_01 | 72.6s | 112 | 1438 | 1342 | 149 MB |
| run_02 | TBD | TBD | TBD | TBD | 123 MB |
| run_03 | TBD | TBD | TBD | TBD | 132 MB |

## Ground truth measurements
See `ground_truth.txt`. Three features measured before recording:
- Bedroom door width: 91.4 cm
- Door-to-wall passage: 118.9 cm
- Bed-to-wall passage: 97.5 cm

## Observations
- LiDAR rate during recording was ~1.5 Hz, lower than the ~7 Hz seen 
  during Sprint 1's idle test. Likely cause: WebRTC bandwidth limits 
  over Wi-Fi when actively recording. Frame count (112 per run) is 
  still sufficient for Sprint 3 preprocessing development.
- No red LED faults during recording.
- Bags initially saved to wrong directory (cwd was old folder); 
  manually moved to ~/autopass/bags/ post-recording. Fixed for next session.

## Backup
- Local: ~/autopass/bags/
- Cloud: One Drive/
