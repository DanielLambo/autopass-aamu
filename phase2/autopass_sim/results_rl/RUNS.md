# Report 3 run log

All runs are simulated. Training is CPU only: `~/venvs/apsim` (Python 3.10, torch CPU), launched with
`env -u PYTHONPATH PYTHONUNBUFFERED=1` so the ROS 2 Python path does not leak in. Machine: i7-12800H
(20 threads), 62 GB RAM. Each training run uses 18 env worker processes (14 for the pilots) and one torch
thread in the learner. Runs in a batch go one after another from a nohup script.

Evaluation is deterministic (greedy policy) on episode seeds 5000-5599 in four settings: `main`
(wheelchair or walker), `uniform` (w_A ~ U(66, 92) cm, unseen profiles), `budget20` and `budget12`
(scan budgets). PPO rows are mean ± std over 3 training seeds.

## v1: environment and first policies (tag `report3-v1`)

Pilots (`v1/pilots/`, seed 0, 14 envs, λ learning rate 0.05): code `91c1caf` plus an uncommitted A*
route cache keyed by exact position (env.py blob `d898c32`; same seeded behaviour as `91c1caf`).

| Run | Iterations | Time |
|---|---|---|
| ppolag_s0 | 280 | 7,852 s |
| ppolag_shield_s0 | 280 | 7,488 s |
| ppolag_s0_aborted (Isaac training sharing the machine) | 50 | 1,892 s |

The unshielded pilot ended at cost ~0.12 against a limit of 0.01, with λ still rising ~0.006 per
iteration. The full batch therefore uses `--lam-lr 0.5`.

Full batch, code `4a5c148` (distance-field routes, GOTO path cache), `v1/run_full.sh`:
500k steps, 18 envs, `--lam-lr 0.5`, seeds 0-2, without and with the post-hoc shield. Six runs,
218 iterations each, 58-60 min each (2026-11-06 19:37 to 2026-11-07 02:16).
Outputs: `v1/ppolag_s{0,1,2}/`, `v1/ppolag_shield_s{0,1,2}/`, sanity check `v1/sanity.json`.

Eval (`5b34a20`, `v1/eval/`): on `main`, the shield-trained policy with the shield on reaches 98.0 ± 0.3%
success, 0 unsafe, 1.4 ± 0.3 s faster than the rule. Without the shield the policies make 0.8-1.3% unsafe
commits.

## v2: mask shield and budget-aware policy (tag `report3-v2`)

Code `2ad8d1d`: `shield_mode="mask"` removes COMMIT from the action mask unless the z-test passes (instead
of replacing it after the fact), and the remaining scan budget is part of the observation (51 values).
Training samples the budget from {none 0.4, 30/20/12 sweeps 0.2 each}.

`v2/run_v2.sh`: 500k steps, 18 envs, `--lam-lr 0.5`, seeds 0-2.
- `v2/A_mask_budget_s*`: `--shield --shield-mode mask --budget-dist` (reported method, "v2-A")
- `v2/B_noshield_budget_s*`: `--budget-dist`, no shield (Lagrangian only, "v2-B")

2026-11-07 02:48 to 07:41, 47-51 min per run. v2-A had 0 unsafe commits in every training iteration.
v2-B ended at cost 0.019-0.027 with λ ≈ 7.

Eval (`567501b`, `v2/eval/`, figure `v2/eval/fig_main.png`). Rule rows are identical to v1 in all settings.

| setting | policy (shield at eval) | success | unsafe | timeout | time vs rule |
|---|---|---|---|---|---|
| main | rule | 97.5% | 0 | 0 | — |
| main | v2-A (on) | 97.8 ± 1.0% | 0 | 1.1 ± 1.1% | −1.0 ± 0.6 s |
| main | v2-B (off) | 96.2 ± 0.4% | 0.2 ± 0.2% | 3.6 ± 0.5% | +2.8 ± 0.8 s |
| budget12 | rule | 30.0% | 0 | 0 | — |
| budget12 | v2-A (on) | 32.4 ± 0.2% | 0 | 0 | — |

Success against the rule, paired exact McNemar per training seed (b = only the policy succeeds,
c = only the rule succeeds), shield on:

| policy | setting | seed 0: Δ, b/c, p | seed 1 | seed 2 |
|---|---|---|---|---|
| v2-A | budget20 | +2.7, 46/30, 0.085 | +2.0, 38/26, 0.17 | +2.3, 45/31, 0.14 |
| v2-A | budget12 | +2.7, 28/12, 0.017 | +2.3, 24/10, 0.024 | +2.3, 28/14, 0.044 |

Not a claimable success gain: budget12 is p < 0.05 for each seed alone, but the worst seed after Holm
correction over the two budget settings is p ≈ 0.087.

## v3: per-door dwell cap (tag `report3-v3-dwellcap`, negative result)

Goal: remove v2-A's leftover timeouts (mostly seed 1, 14/600 on `main`). Code `e97e8a4` adds `--dwell-cap 20`.
Before launch, the rule with the cap was checked to be identical to the uncapped rule on all 600 seeds in all
four settings. A first version that charged dwells to the nearest door changed the rule on seeds 5294 and
5509, so dwells are counted at the door whose viewpoint the robot stands at.

`v3/run_v3.sh`: v2-A settings plus the cap, seeds 0-2, 2026-11-07 11:28 to 13:51, 47-49 min per run,
0 unsafe in every iteration. Eval code `db903e4`, `v3/eval/`.

Timeouts with the shield on, out of 600 episodes:

| setting | v2-A seeds 0/1/2 | v3 seeds 0/1/2 |
|---|---|---|
| main | 3 / 14 / 2 | 5 / 5 / 5 |
| uniform | 3 / 17 / 3 | 5 / 6 / 4 |
| budget20, budget12 | 0 | 0 |

The cap removed seed 1's outlier but not the timeouts. On seeds 5039, 5222 and 5226 the v3 policy never
dwells; it walks back and forth between two viewpoints of one door until the 80-step limit, scanning on
the way. A dwell-only cap cannot bound that. v2-A loops by dwelling instead (~78 DWELLs on a door that
never reaches PASS). v3 is not used in the report. The timeouts (< 1% of episodes, all safe) stay a known
limitation; a GOTO cap per door or a timeout penalty would be the next fix.

## Addendum (tag `report3-v2`, folder `v2/addendum/`)

Code `2763954` (`--hide-wA`) and `09ea1ac` (env options `world_seed` and `viewpoints`, eval set `v2nowa`,
tools `fig_margin.py`, `nr_compare.py`, `fig_train_curves.py`, `replay_rl.py`, `nowa_compare.py`). New options
are off by default; tests pass and the v2-A eval rows reproduce `v2/eval` exactly.

**v2-A without w_A in the state.** Same as v2-A with the w_A feature set to 0 in training and eval
(`v2/addendum/run_noWA.sh`, `A_noWA_s*`, 2026-11-07 15:12 to 17:39, 0 unsafe in every iteration).
Eval: `python -m autopass_rl.evaluate --set v2nowa --settings main uniform --out results_rl/v2/addendum/eval_noWA`,
paired comparison `python tools/nowa_compare.py` (`eval_noWA/nowa_vs_v2A.md`).

| setting (shield on) | noWA success s0/s1/s2 | v2-A success s0/s1/s2 | noWA timeouts | v2-A timeouts |
|---|---|---|---|---|
| main | 97.0 / 98.3 / 98.3 (mean 97.9) | 98.3 / 96.7 / 98.3 (97.8) | 11 / 3 / 2 | 3 / 14 / 2 |
| uniform | 97.0 / 98.3 / 98.3 (97.9) | 98.3 / 96.3 / 98.2 (97.6) | 12 / 3 / 2 | 3 / 17 / 3 |

Both have 0 unsafe with the shield. Time vs the rule is −1.0 s for both on `main`; paired noWA − v2-A time per
seed is −0.5 to +0.7 s (Wilcoxon p > 0.15). Seeds 0 and 1 differ in success (McNemar p < 0.01) but in opposite
directions, and every differing episode is a timeout. Without the shield, noWA makes more unsafe commits on
seeds 0 and 1 (27.7 / 26.0% vs 21.2 / 18.7%). Only the explicit w_A feature is hidden: the per-door margin
features (width − w_A) still carry it, so this does not show the policy works without knowing w_A.

**Sweeps per door vs normalized margin** (`fig_margin.png`, `python tools/fig_margin.py`; `main`, rule and
v2-A seeds 0-2). A decision is a COMMIT or REJECT; sweeps are the door's trusted sweeps at that moment.
Decisions with no trusted sweep are left out (rule 35, v2-A 28).
- Rule: ~25 sweeps for |m̂|/σ < 3 (it never commits there; it dwells to the cap and flags), ~10.5 at 3-4,
  ~5 at ≥ 10.
- v2-A: 15 decisions with |m̂|/σ < 3 over three seeds, against 100 for the rule. At 3-4 it commits after
  ~4.8 sweeps (97% commits). It moves to another door instead of dwelling on an ambiguous one.

**No-repositioning setting** (`python tools/nr_compare.py`, `nr_table.md`). Report 2's worlds: room +
corridor, world seeds 1000-1299, w_A = 81.3 cm, built with Report 2's `make_world` and sensor seed. No
repositioning in the RL env means GOTO only to viewpoint 0. Report 2's planners were rerun from the same
code into `report2_rerun/` and match the committed Report 2 results on 1800/1800 missions.

| policy | success | unsafe | time, all missions |
|---|---|---|---|
| rule, RL env, no reposition | 98.5% | 0 | 28.0 s |
| v2-A s0/s1/s2, no reposition | 98.7 / 97.3 / 98.5% | 0 | 25.4 / 26.5 / 25.8 s |
| v2-A s0/s1/s2, all viewpoints | 99.0 / 97.5 / 98.7% | 0 | 25.9 / 26.6 / 26.5 s |
| P-A*(4)-nr (Report 2) | 98.3% | 0 | 29.0 s |
| B2-nr (Report 2) | 21.5% | 0 | 38.5 s |
| P-A*(4) (Report 2, repositioning) | 99.7% | 0 | 28.1 s |

Paired against the RL-env rule, v2-A saves 2.5 / 1.5 / 2.2 s. The two simulators execute paths differently
(discrete viewpoints vs a planned path with a 1.2 m commit point), so cross-simulator times are indicative.

**Training curves** (`fig_train_curves.png`): return, cost and λ for v2-A and v2-B, mean of three seeds with
a min-max band. v2-B's cost peaks at ~0.24 and falls to ~0.02 as λ rises to ~7. v2-A's cost and λ stay at 0.
The return does not penalize unsafe commits, so v2-B's higher return includes unsafe ones.

**Replay** (`replay_rule_vs_v2A_5081.{mp4,png}`, `python tools/replay_rl.py --seed 5081`). The world is the
lowest `main` seed with 3 doors, a wheelchair, at least one too-narrow door and both policies succeeding,
whose time difference equals the 25th-percentile saving (−10 s). The rule dwells on door 0 (margin +0.75 cm)
to the cap, flags it and goes through door 1 (35.7 s, 45 sweeps). v2-A skips door 0 and goes through door 1
(25.7 s, 25 sweeps). Both match the eval CSV. MP4 encoding uses `imageio-ffmpeg` in the apsim venv.

## Isaac Lab check (tag `report3-gui`, folder `gui/`)

Does the trained Go2 walking policy carry out v2-A's plan? Isaac Sim 5.1 / Isaac Lab 2.3 (conda env
`env_isaaclab`), GUI on, RTX 3080 Ti Laptop GPU.

Plan (`gui/export_plan.py`): world 5081 from the replay (corridor, door margins +0.75 / +8.90 / −0.37 cm),
v2-A seed 0 with the mask shield. Three actions: GOTO door 0 viewpoint 0, GOTO door 1 viewpoint 0, COMMIT
door 1 (z-test PASS, fused margin +8.7 ± 0.5 cm). 25.7 s and 25 sweeps in the RL env, all taken while
walking (no DWELL).

Scene (`gui/isaac_go2_plan.py`): flat ground; 2.4 m door wall with openings at the soffit width plus the
door stops, so the clear width equals the true width; lintels from 2.03 m; 1.0 m outer walls (kept low for
the cameras); furniture as 0.75 m boxes (the 2D world has no heights). Thin floor strips mark each door's
margin (yellow 0-3 cm, green ≥ 3 cm, red < 0), the viewpoints (blue) and the goal; they have no collision.

Robot: Unitree-Go2-Velocity from unitree_rl_lab, checkpoint `model_800.pt`, one env, no observation noise,
pushes or randomization. A pure-pursuit controller (lookahead 0.25 m, vx ≤ 0.5 m/s, |wz| ≤ 1 rad/s) writes
the velocity command. At each viewpoint the robot turns to face the door and stands for 1 s; this pause is
not in v2-A's plan. Logged at 50 Hz: pose, tilt, contact force on non-foot links, and clearance from each
link origin to the nearest wall or furniture footprint.

Two runs of the same decisions:
- `faithful`: the RL env's paths as given.
- `straight` (`--door-straight`): the commit path near the door replaced by a straight pass through the door
  centre from 0.5 m before to 0.7 m after the wall, then the rest of the plan.

| run | waypoints reached | fall | max tilt | non-foot contacts | min clearance | final error | time |
|---|---|---|---|---|---|---|---|
| faithful | 5/5 | no | 3.9° | 11 steps on door 1's frame, max 364 N (RL_hip, FL_calf) | 0.013 m | 0.149 m | 36.1 s |
| straight | 6/6 | no | 3.9° | 0 | 0.258 m | 0.147 m | 36.8 s |

The env plans for a point with 0.2 m of inflation. On this world its commit path crosses the wall 0.27 m left
of the door centre and turns ~60° about 0.1 m past it, so the Go2's rear hip clips the frame. Walking straight
through the door centre before turning avoids it. The decisions are unaffected; the env needs the robot's
footprint. Run time is longer than the env's 25.7 s because of the two turn-and-wait stops and slower turns.

Outputs: `plan_5081.json`, `go2_v2A_5081_{faithful,straight}.mp4` (25 fps, ~36 s),
`shot_{faithful,straight}_{overview,at_viewpoint,at_commit_door,through_door}.png`,
`trajectory_*.csv`, `walkability_*.json`, `walk_paths_5081.png` (`python gui/plot_walk.py`).
GPU PhysX is not bit-reproducible: rerunning `straight` gave 36.8 s instead of 37.1 s with the same contacts
and clearance.
