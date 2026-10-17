# autopass_sim — observability-aware path planning testbed (AutoPass Phase 2)

A 2D/3D-hybrid simulator for comparing a shortest-path planner with an
observability-aware planner on doorway passability decisions. Everything here is
**simulation**. Nothing in this folder is experimentally validated on the Go2 yet.

## What is real, simulated, or assumed

| Item | Status | Source / value |
|---|---|---|
| L1 point rate, FOV, range accuracy, resolution | from spec sheet | 21,600 pts/s, 360°×90°, ±2 cm, 8 mm |
| Door frame profile | from standards | SDI steel: 5/8 in stop, 1-15/16 in rabbet; wood: 3/8 in stop |
| Wheelchair clear width | from standard | ADA §404.2.3, 81.3 cm (32 in) |
| Sensor height 0.30 m, elevation band −15..75°, 1 cm range noise | **assumed** | calibrate on bags |
| 1 cm per-observation error floor (plane fit, registration, odometry) | **assumed** | calibrate on bags |
| Uniform random scan pattern | **assumed** | real L1 pattern is non-repetitive |
| Width estimator | simplified GAIT edge step | innermost jamb points in the frame slab |
| Floor plans, door widths, furniture | simulated | procedural (`world.py`) |
| Go2 motion | geometric only | disc of radius 0.20 m; no gait dynamics |
| `/point_cloud2` from the Go2 bags | **measured** (run_01) | 5 cm voxel map in `odom`, ~1.5 msg/s; modeled by `--sensor voxelmap` |

## Layout

```
autopass_sim/
  geometry.py       segment ray casting, grid collision checks
  world.py          procedural floor plans: dividing wall, 2-3 doors, furniture, room/corridor layouts
  sensor.py         L1-based scan, 5 cm voxel-map mode, GAIT-style width estimator, view geometry (d, psi)
  calibrate.py      sweep -> calib.json (bias vs distance, noise vs distance, hidden-stop bias vs psi)
  observability.py  q(d, psi), trust rule, planner cost field c(x) = 1 + lambda * g(x)
  planners.py       A*, RRT* (same cost), cost-aware shortcutting
  mission.py        plan -> drive -> observe -> decide at commit point -> reposition/dwell -> replan
  experiments.py    batch runner (all methods on the same worlds)
  analyze.py        summary table with 95% CIs + figures
  visualize.py      draw one world with two methods' paths
tests/              sanity tests (pytest)
```

## Run (TensorBook)

```bash
python3 -m venv ~/venvs/apsim && source ~/venvs/apsim/bin/activate
pip install numpy scipy matplotlib pytest
cd autopass_sim
python -m pytest -q tests
python -m autopass_sim.calibrate --trials 200            # ~1 min, square-on b0(d)
python -m autopass_sim.calib_density --trials 200        # ~2 min, writes calib_rho.json (sigma(rho,psi), rho_hat, B(rho))
python -m autopass_sim.experiments --worlds 300 --rrt-worlds 100 --out results
python -m autopass_sim.analyze --results results
python tools/paired_stats.py results                       # McNemar, paired times, RRT*/A* agreement -> report_numbers.json
python -m autopass_sim.visualize --layout corridor --seed 1003 --out results/fig_example.png
# realism check with the Go2's voxel map:
python -m autopass_sim.experiments --worlds 100 --rrt-worlds 0 --sensor voxelmap --out results
python -m autopass_sim.analyze --results results --sensor voxelmap
```
Every run is seeded (`--seed0`), so the same command reproduces the same numbers.

## Renders for reports and talks

```bash
# side-by-side 2D replay (MP4) + key-moment stills
python -m autopass_sim.render2d --layout corridor --seed 1016 --methods "B1,P-A*(4)" \
    --out results/replay.mp4 --stills results/replay_stills.png
# angle-sweep animation (robot walks an arc around one frame)
python -m autopass_sim.render_sweep --out results/sweep_door.mp4
# 3D: one door scanned square-on vs oblique (hidden-stop effect)
python -m autopass_sim.render3d door --out results/render3d_door.png
# 3D: whole mission with extruded walls, both paths, LiDAR hits
python -m autopass_sim.render3d world --layout corridor --seed 1016 --out results/render3d_world.png
# REAL Go2 data (system Python + ROS 2, not the venv):
source /opt/ros/humble/setup.bash && /usr/bin/python3 tools/render_bag.py ~/autopass/bags/run_01 --out results/bag_run_01.png
```
Every simulated render is labeled "simulated" on the image. Only `render_bag.py` shows robot data.

## Methods compared

| Name | Path | Decision rule |
|---|---|---|
| B1 | A*, λ = 0 (shortest) | naive: all scans, no bias allowance |
| B2 | A*, λ = 0 (shortest) | trusted scans + bias allowance, reposition when undecided |
| P-A*(λ) | observability-aware A* | same as B2 |
| P-A*(4)n | observability-aware A* | naive (isolates the effect of the path) |
| P-RRT*(4) | observability-aware RRT* | same as B2 |

Mission: reach the goal through a door the agent (wheelchair, 81.3 cm) can pass. The robot
must decide PASS / FAIL / FLAG at a commit point 1.2 m before the door. FAIL or unresolved
FLAG means the door is treated as blocked, and the robot replans.

## Next steps
1. Replace the assumed values above with fits from the Phase 1 bags (and new raw-L1 bags).
2. Add Go2 velocity and turning limits (dynamic feasibility), then port the planner to
   Nav2 / Isaac Sim.

## Report 2 revision (density-aware model)
`experiments.py` now uses `calib_rho.json` by default (`--rho-calib ''` reproduces the original distance-only model).
Per sweep the decision rule uses the measured edge density rho: sigma = hypot(3.19 rho^-0.665 e^{0.032 psi}, 1 cm),
bias bound B(rho) = max(0.5, 2.12/sqrt(rho)); the fused bound is the inverse-variance-weighted mean of B.
The planner's cost field uses the expected density rho_hat(d, psi) = 71.6 d^-1.92 e^-0.011 psi.

Real-bag width analysis (system Python + ROS 2):
```
/usr/bin/python3 tools/analyze_bags.py ~/autopass/bags/run_01 ~/autopass/bags/run_02 \
   --feature door:X,Y:91.4 --feature passage_a:X,Y:118.9 --feature passage_b:X,Y:97.5 --out results/bag_widths
```

## Result folders (Report 2)
- `results_tb/`: first-draft distance-only model, 300 worlds per layout (run on the TensorBook).
- `results_rev/`: revised density-aware model: main run (`missions_raw.csv`), voxel-map run, harder scenarios and the rho ablation (`missions_raw_scen.csv`); `report_numbers.json` and `scenario_numbers.json` hold every number quoted in Report 2.
- `results/calib_density.csv`: the 31,200-sweep density/angle calibration behind `calib_rho.json`.
