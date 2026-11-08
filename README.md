# AutoPass

Autonomous passability profiling for indoor environments with a Unitree Go2 and its L1 LiDAR.

AAMU Computer Science research. Daniel Lambo, advisor Dr. Yujian Fu.

## Question

Can a LiDAR-equipped quadruped decide on its own whether a doorway is wide enough for a given person
(wheelchair user, walker user), with accuracy close to measuring it by hand, and do it safely when a
door is within a few centimetres of the limit?

## Repository layout

| Path | What it holds |
|---|---|
| `phase1/` | Phase 1: real Go2 data. ROS 2 bag recording, preprocessing, RANSAC floor/wall segmentation, doorway detection, agent profiles and passability rules, evaluation against tape measurements. |
| `phase2/autopass_sim/` | Phase 2 simulator and everything built on it. |
| `phase2/autopass_sim/autopass_sim/` | Report 2: L1 sensor model, procedural floor plans, observability-aware A*/RRT* planners, mission executor, experiments. |
| `phase2/autopass_sim/autopass_rl/` | Report 3: constrained MDP (Gymnasium env) and masked PPO-Lagrangian. |
| `phase2/autopass_sim/gui/` | Report 3: running a learned plan with the trained Go2 walking policy in Isaac Lab. |
| `phase2/autopass_sim/tools/` | Figures, statistics and replays used in the reports. |
| `phase2/autopass_sim/results*/` | Saved results. Each folder has a README or summary. |

## Stages

**Phase 1 (Sprints 1–11).** Doorway width from real Go2 point clouds. Details in [`phase1/README.md`](phase1/README.md).
Measuring while walking past a door turned out to be unreliable: oblique views and occluded jambs make
openings read too wide or too narrow. That result set up Phase 2.

**Report 2: observability-aware planning (simulation).** The robot only commits to a door when a z-test on
width estimates from trusted views passes, and plans paths that give it those views. On 600 simulated worlds
(300 room, 300 corridor) the trusted-view rule removed unsafe commits (0 vs 0.3–2.3% for a shortest-path
robot deciding while walking), and the observability-aware path cut the width error at the decision from
about 0.2 to 0.14 cm for 0.6–1.8 s more per mission. Code and results: [`phase2/autopass_sim/README.md`](phase2/autopass_sim/README.md).

**Report 3: constrained reinforcement learning (simulation).** A masked PPO-Lagrangian policy decides
which door to look at, from where, and when to commit, with the Report 2 z-test as a shield. It matches
the rule's zero unsafe commits and saves about 1 s per mission on average, more on worlds with a door
near the limit, because it moves on instead of dwelling on that door. A sim-to-sim check in Isaac Lab
runs the learned plan on the trained Go2 walking policy.
Run log and results: [`phase2/autopass_sim/results_rl/`](phase2/autopass_sim/results_rl/).

All Report 2 and Report 3 results are from simulation. They have not been validated on the robot yet.

## Report snapshots

Each report's final state is tagged:

| Tag | Contents |
|---|---|
| `report2` | Report 2 code and results |
| `report3-v1` | Report 3 first version: environment, PPO-Lagrangian, evaluation |
| `report3-v2` | Report 3 method (mask shield, budget-aware policy) plus the addendum experiments |
| `report3-v3-dwellcap` | Dwell-cap retraining (recorded negative result; its code is only in this tag) |
| `report3-gui` | Isaac Lab check of the learned plan |

## Videos

- Report 2 replays and the angle-sweep animation: [`phase2/autopass_sim/media/`](phase2/autopass_sim/media/)
- Rule vs learned policy, 2D replay: [`replay_rule_vs_v2A_5081.mp4`](phase2/autopass_sim/results_rl/v2/addendum/replay_rule_vs_v2A_5081.mp4)
- Go2 in Isaac Lab following the learned plan: [straight door pass](phase2/autopass_sim/results_rl/gui/go2_v2A_5081_straight.mp4), [path as planned](phase2/autopass_sim/results_rl/gui/go2_v2A_5081_faithful.mp4)

## Platform

Unitree Go2 EDU, ROS 2 Humble, Ubuntu 22.04. Open3D for Phase 1 point clouds. NumPy/SciPy/PyTorch (CPU)
for the simulator and RL. Isaac Sim 5.1 / Isaac Lab 2.3 with unitree_rl_lab for the Go2 walking policy.
