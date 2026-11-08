# autopass_sim

Simulator and experiments for AutoPass Phase 2: deciding whether a doorway is passable from LiDAR, and
planning where to look from. Report 2 (observability-aware planning) and Report 3 (constrained RL) both
run on it. Everything here is simulation; none of it has been validated on the Go2 yet.

## What is measured, assumed or simulated

| Item | Status | Source / value |
|---|---|---|
| L1 point rate, FOV, range accuracy, resolution | spec sheet | 21,600 pts/s, 360°×90°, ±2 cm, 8 mm |
| Door frame profile | standards | SDI steel: 5/8 in stop, 1-15/16 in rabbet; wood: 3/8 in stop |
| Wheelchair clear width | standard | ADA §404.2.3, 81.3 cm (32 in) |
| Sensor height 0.30 m, elevation band −15..75°, 1 cm range noise | assumed | to calibrate on bags |
| 1 cm per-observation error floor (plane fit, registration, odometry) | assumed | to calibrate on bags |
| Uniform random scan pattern | assumed | the real L1 pattern is non-repetitive |
| Width estimator | simplified GAIT edge step | innermost jamb points in the frame slab |
| Floor plans, door widths, furniture | simulated | procedural (`world.py`) |
| Go2 motion | geometric | disc of radius 0.20 m, no gait dynamics (see `gui/` for a legged check) |
| `/point_cloud2` from the Go2 bags | measured (run_01) | 5 cm voxel map in `odom`, ~1.5 msg/s; modelled by `--sensor voxelmap` |

## Layout

```
autopass_sim/      Report 2 simulator
  geometry.py        segment ray casting, grid collision checks
  world.py           procedural floor plans: dividing wall, 2-3 doors, furniture; room and corridor layouts
  sensor.py          L1 scan, 5 cm voxel-map mode, width estimator, view geometry (d, psi)
  calibrate.py       sweep -> calib.json (bias and noise vs distance, hidden-stop bias vs psi)
  calib_density.py   density/angle calibration -> calib_rho.json
  observability.py   view quality q(d, psi), trust rule, planner cost field c(x) = 1 + lambda * g(x)
  planners.py        A*, RRT* (same cost), cost-aware shortcutting
  mission.py         plan -> drive -> observe -> decide at the commit point -> reposition/dwell -> replan
  experiments.py     batch runner, all methods on the same worlds
  analyze.py         summary tables with 95% CIs and figures
  render2d.py, render3d.py, render_sweep.py, visualize.py   replays and figures
autopass_rl/       Report 3: constrained MDP and learner
  env.py             Gymnasium env (AutoPassEnv) on top of the Report 2 simulator
  policies.py        the Report 2 decision rule as an in-env policy
  ppo_lag.py         masked PPO-Lagrangian (single file, CPU)
  evaluate.py        deterministic evaluation, paired statistics, McNemar
  MDP.md             problem formulation
gui/               Report 3: run a learned plan with the Go2 walking policy in Isaac Lab
tools/             report figures, statistics, replays, bag analysis
tests/             pytest (simulator and env)
media/             Report 2 videos (GIF + MP4)
```

## Setup

```bash
python3 -m venv ~/venvs/apsim && source ~/venvs/apsim/bin/activate
pip install -r requirements.txt          # CPU torch is enough
cd ~/autopass/phase2/autopass_sim
python -m pytest -q tests
```
Every experiment is seeded, so the same command reproduces the same numbers.

## Report 2: observability-aware planning

| Method | Path | Decision rule |
|---|---|---|
| B1 | A*, λ = 0 (shortest) | naive: all scans, no bias allowance |
| B2 | A*, λ = 0 (shortest) | trusted scans + bias allowance, reposition when undecided |
| P-A*(λ) | observability-aware A* | same as B2 |
| P-A*(4)n | observability-aware A* | naive (isolates the effect of the path) |
| P-RRT*(4) | observability-aware RRT* | same as B2 |

Mission: reach the goal through a door the agent (wheelchair, 81.3 cm) can pass. The robot decides
PASS / FAIL / FLAG at a commit point 1.2 m before the door. FAIL or an unresolved FLAG marks the door
blocked and the robot replans. `-nr` variants never reposition.

The decision rule is density-aware: per sweep, sigma = hypot(3.19 rho^-0.665 e^{0.032 psi}, 1 cm) with
bias bound B(rho) = max(0.5, 2.12/sqrt(rho)), fused by inverse-variance weighting. The planner's cost field
uses the expected density rho_hat(d, psi) = 71.6 d^-1.92 e^-0.011 psi (`calib_rho.json`; pass
`--rho-calib ''` for the original distance-only model).

```bash
python -m autopass_sim.calibrate --trials 200             # ~1 min
python -m autopass_sim.calib_density --trials 200         # ~2 min, writes calib_rho.json
python -m autopass_sim.experiments --worlds 300 --rrt-worlds 100 --out results_rev
python -m autopass_sim.analyze --results results_rev
python tools/paired_stats.py results_rev                  # McNemar, paired times -> report_numbers.json
python -m autopass_sim.experiments --worlds 300 --rrt-worlds 0 --out results_rev --tag _scen \
    --methods "B2,P-A*(4),B2-nr,P-A*(4)-nr,B2-b4,P-A*(4)-b4,B1-rhohat"   # harder scenarios, no-reposition
python tools/scenario_stats.py results_rev
python -m autopass_sim.experiments --worlds 100 --rrt-worlds 0 --sensor voxelmap --out results_rev
python -m autopass_sim.analyze --results results_rev --sensor voxelmap
```

Result folders:
- `results_rev/`: the Report 2 numbers. Main run (`missions_raw.csv`), voxel-map run, harder scenarios and
  the rho ablation (`missions_raw_scen.csv`). `report_numbers.json` and `scenario_numbers.json` hold every
  number quoted in the report.
- `results_tb/`: first draft with the distance-only model (300 worlds per layout).
- `results/`: calibration data (`calib_density.csv`, 31,200 sweeps) and bag width analysis.
- `RESULTS.md`: notes from the first preliminary run (v0, 200 worlds).

Replays and figures:
```bash
python -m autopass_sim.render2d --layout room --seed 1077 --methods "B1,P-A*(4)" --out media/replay_room_1077.mp4
python -m autopass_sim.render_sweep --out media/sweep_door.mp4
python -m autopass_sim.render3d door --out results/render3d_door.png
python -m autopass_sim.render3d world --layout corridor --seed 1016 --out results/render3d_world.png
```

Real bag data (ROS 2 system Python, not the venv):
```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 tools/render_bag.py ~/autopass/bags/run_01 --out results/bag_run_01.png
/usr/bin/python3 tools/analyze_bags.py ~/autopass/bags/run_01 ~/autopass/bags/run_02 \
    --feature A:-0.20,0.12 --out results/bag_widths
```
Add `:true_cm` to a feature (`--feature door:X,Y:91.4`) to report error against a tape value.

## Report 3: constrained RL

The formulation is in [`autopass_rl/MDP.md`](autopass_rl/MDP.md). Training, evaluation and the full run
log are in [`results_rl/`](results_rl/).

```bash
python tools/sanity_env.py --episodes 200                 # random vs rule, shield on/off, determinism
python -m autopass_rl.ppo_lag --total-steps 500000 --envs 18 --seed 0 --lam-lr 0.5 \
    --shield --shield-mode mask --budget-dist --out results_rl/v2/A_mask_budget_s0
python -m autopass_rl.evaluate --set v2 --settings main uniform budget20 budget12 --out results_rl/v2/eval
```

### Isaac Lab check (`gui/`)

`gui/export_plan.py` writes one world and the learned policy's actions as waypoints.
`gui/isaac_go2_plan.py` rebuilds the world in Isaac Lab (walls, doors at their true clear widths,
furniture) and drives the Go2 velocity policy trained with unitree_rl_lab through the waypoints with a
pure-pursuit controller, recording video, screenshots and a contact/clearance log. It runs in the Isaac Lab
conda environment:

```bash
python gui/export_plan.py --seed 5081 --out results_rl/gui/plan_5081.json
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES ~/anaconda3/envs/env_isaaclab/bin/python gui/isaac_go2_plan.py \
    --plan results_rl/gui/plan_5081.json --out results_rl/gui --tag straight --door-straight
python gui/plot_walk.py --dir results_rl/gui --seed 5081
```

## Next steps

1. Fit the assumed sensor values from the Phase 1 bags and new raw-L1 recordings.
2. Give the RL environment the robot's footprint and turning limits; the Isaac check shows the
   point-robot commit path can clip a door frame.
3. Run the planner on the Go2.
