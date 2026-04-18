# AutoPass

Autonomous Passability Profiling for Indoor Environments using Unitree Go2 LiDAR.

**AAMU CS Research** · Daniel Lambo · Advisor: Dr. Yujian Fu

## Research problem

Can a LiDAR-equipped quadruped robot autonomously assess whether an indoor
environment is navigable for a specified agent, with measurement accuracy
comparable to manual inspection?

## Pipeline

Sensor → Preprocessing → Plane segmentation → Feature detection → Passability classification → Evaluation

## Sprint status

- [x] Sprint 1: LiDAR + ROS2 pipeline stable
- [ ] Sprint 2: Record rosbag dataset
- [x] Sprint 3: Preprocessing pipeline (voxel, outlier removal, ROI)
- [x] Sprint 4: Floor + wall segmentation (RANSAC)
- [ ] Sprint 5: Progress report to advisor

## Platform

Unitree Go2 (EDU) · ROS2 Humble · Ubuntu 22.04 · Open3D for point cloud processing
