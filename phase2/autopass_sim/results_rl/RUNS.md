# results_rl run log

## Pilot s0 runs (old code, position-keyed route cache)

Both runs use the `report3-env` working copy at HEAD `91c1caf` plus one uncommitted change to
`autopass_rl/env.py`: an A* route cache keyed by (exact position, door), cleared on REJECT.
The code is identical to `91c1caf` in behaviour: the seeded per-step fingerprint and sanity.json match exactly.

| | Value |
|---|---|
| env.py blob (`git hash-object`) | `d898c322d98b5deccf962b8ba933ae0618d672fc` |
| `git diff env.py \| sha256sum` (first 16) | `6adef1630ee7648c` |
| env.py mtime | 2026-11-06 15:17:30 CST |
| git stash | empty (change is uncommitted in the working tree, not stashed) |

| Run | Command | Launched | env.py at launch |
|---|---|---|---|
| ppolag_s0 | `ppo_lag --total-steps 500000 --envs 14 --seed 0` | 2026-11-06 15:20:26 CST | blob d898c32 (verified) |
| ppolag_shield_s0 | `ppo_lag --total-steps 500000 --envs 14 --seed 0 --shield` | 2026-11-06 17:31:20 CST | blob d898c32 (verified) |

ppolag_s0 finished at 2026-11-06 ~17:31 CST: 280 iterations, 501,760 steps, 7,852 s (~64 steps/s), no errors.

Environment: `~/venvs/apsim` (Python 3.10.12, torch 2.14.1 CPU), `env -u PYTHONPATH`, `PYTHONUNBUFFERED=1`,
no thread env vars. Runs are sequential from one nohup script.

Earlier aborted attempt (same seed, code without the cache, Isaac training sharing the machine) is in
`pilot_oldcode/ppolag_s0_aborted/` (log `pilot_oldcode/ppolag_s0_aborted.log`, not in git). It was stopped at iteration 50 at ~66 steps/s.

## Decision for the full batch (2026-11-06 17:35 CST)

The unshielded pilot ended at cost ~0.12 (limit 0.01) with λ still rising ~0.006/iteration
(lam_lr 0.05) and entropy ~0.07. The full batch uses `--lam-lr 0.5` (10x faster λ response). Other
settings unchanged: 500k steps, --envs 18, seeds 0-2, unshielded and shielded, on the merged
report3-env (distance-field routes + GOTO path cache).

## Full batch (launched 2026-11-06 19:37 CST)

Code: report3-env `4a5c148` (distance-field routes + GOTO path cache; ppo_lag thread defaults = 1).
Script: `results_rl/run_full.sh`. 500k steps, --envs 18, --lam-lr 0.5, seeds 0-2, unshielded then shielded.
Pilots moved to `pilot_oldcode/` (code: 91c1caf + position-keyed cache, env.py blob d898c32, --lam-lr 0.05, --envs 14).
`sanity.json` was regenerated on 4a5c148 (rule path 11.97 m; earlier copy had 12.00 m).

Full batch finished 2026-11-07 02:16 CST: all 6 runs reached 218 iterations / 502,272 steps, ~58-60 min each, no errors.

## Report 3 v2 training (launched 2026-11-07 02:48 CST)

Branch report3-v2, code commit `2ad8d1d` (shield_mode mask + observable scan budget). Script: `results_rl/v2/run_v2.sh`.
500k steps, --envs 18, --lam-lr 0.5, seeds 0-2, run sequentially (A then B per seed):
- A `v2/A_mask_budget_s*`: --shield --shield-mode mask --budget-dist (main method)
- B `v2/B_noshield_budget_s*`: --budget-dist, no shield (Lagrangian-only ablation)
budget_dist = {None: .4, 30: .2, 20: .2, 12: .2}. v1 results above are unchanged.

v2 training finished 2026-11-07 07:41 CST: all 6 runs at 218 iterations / 502,272 steps, ~48-51 min each, no errors.
A (mask shield) runs: 0 unsafe commits in every training iteration. B final cost 0.019-0.027, λ ≈ 7.
v2 eval: `python -m autopass_rl.evaluate --set v2 --settings main uniform budget20 budget12 --out results_rl/v2/eval`
(same seeds 5000-5599; rule rows byte-identical to v1 eval in all 4 settings).

## Report 3 v3 training (launched 2026-11-07 11:28 CST)

Branch report3-v3-dwellcap (renamed from report3-v3), code commit `e97e8a4` (per-door dwell cap). Script: `results_rl/v3/run_v3.sh`.
Variant A only (the v2 timeouts were in A): --shield --shield-mode mask --budget-dist --dwell-cap 20,
500k steps, --envs 18, --lam-lr 0.5, seeds 0-2 -> `v3/A_mask_budget_cap20_s*`. v1/v2 results unchanged.
Gate before launch: rule with dwell_cap=20 identical to v2 rule on seeds 5000-5599 in all 4 eval settings.

v3 training finished 2026-11-07 13:51 CST: 3 runs at 218 iterations / 502,272 steps, ~47-49 min each, no errors,
0 unsafe commits in every iteration, λ = 0 throughout.
v3 eval: code `db903e4` on report3-v3-dwellcap,
`python -m autopass_rl.evaluate --set v3 --settings main uniform budget20 budget12 --out results_rl/v3/eval`.
v3 policies are evaluated with dwell_cap=20; v2-A and v1 rows are evaluated as before (no cap).

### v3 result: negative (not used in the report tables)

Timeouts with the mask shield on at eval (out of 600 episodes per training seed):

| setting | v2-A seeds 0/1/2 | v3-A seeds 0/1/2 |
|---|---|---|
| main | 3 / 14 / 2 | 5 / 5 / 5 |
| uniform | 3 / 17 / 3 | 5 / 6 / 4 |
| budget20 | 0 / 0 / 0 | 0 / 0 / 0 |
| budget12 | 0 / 0 / 0 | 0 / 0 / 0 |

The cap removed seed 1's outlier but not the timeouts. Cause (traced on main seeds 5039, 5222, 5226):
the v3 greedy policy never dwells (door dwell counts stay 0); it walks back and forth between two
viewpoints of one door, scanning while walking, until the 80-step limit, e.g. for seed 5222:

    g(0,0) g(0,3) g(0,0) g(0,3) ... (80 steps; ztests FLAG/FLAG)

where g(i,k) = GOTO door i, viewpoint k. The v2 policy loops by dwelling instead (~78 DWELLs in a row on
a door that never reaches PASS). A DWELL-only cap cannot bound walking sweeps.

Cap design notes: a nearest-door attribution was tried first and changed the rule on 2/600 seeds
(5294, 5509: dwells for the route-first door were charged to a closer door); the committed version
counts DWELLs at a door's viewpoints plus consecutive DWELLs at other stops, and leaves the rule
identical on all 600 seeds in all 4 settings.

### Success under scan budgets: paired exact McNemar vs rule (mask shield on)

b = episodes only the policy succeeds, c = only the rule succeeds.

| policy (mask shield on at eval) | setting | seed 0: Δpts, b/c, p | seed 1 | seed 2 |
|---|---|---|---|---|
| v2-A | budget20 | +2.7, 46/30, p=0.0846 | +2.0, 38/26, p=0.169 | +2.3, 45/31, p=0.135 |
| v2-A | budget12 | +2.7, 28/12, p=0.0166 | +2.3, 24/10, p=0.0243 | +2.3, 28/14, p=0.0436 |
| v3-A (dwell cap 20) | budget20 | +2.2, 44/31, p=0.165 | +1.8, 43/32, p=0.248 | +2.3, 43/29, p=0.125 |
| v3-A (dwell cap 20) | budget12 | +1.8, 27/16, p=0.126 | +1.3, 24/16, p=0.268 | +2.7, 28/12, p=0.0166 |

No claimable success gain. v2-A budget12 is p < 0.05 for each seed alone, but the worst seed after Holm
correction over the two budget settings is p ≈ 0.087; budget20 p = 0.08-0.17.

## Decisions (2026-11-07)

- Reported method: v2-A (mask shield + budget-aware), branch report3-v2 @ `567501b`.
- v3 (per-door dwell cap 20, report3-v3-dwellcap @ `e97e8a4`/`db903e4`) is a recorded negative result.
- Remaining timeouts (< 1 % of episodes, all safe) are a known limitation; the fix (GOTO cap per door
  or a terminal timeout penalty) is deferred. No further retraining for Report 3.

## Report 3 addendum (2026-11-07, branch report3-v2; all results simulated)

Code: `2763954` (env/ppo_lag `hide_wA`), `09ea1ac` (env `world_seed` + `viewpoints` reset options, evaluate set
`v2nowa`, tools `fig_margin.py`, `nr_compare.py`, `fig_train_curves.py`, `replay_rl.py`), plus `tools/nowa_compare.py`
(committed with these results). All new options default off; tests 25 passed; v2-A eval rows reproduce `v2/eval`
exactly. Existing results were not modified. Outputs in `v2/addendum/`.
MP4 encoding: `imageio-ffmpeg==0.5.1` installed into `~/venvs/apsim` (env_isaaclab untouched).

### 1. v2-A-noWA: w_A removed from the state

Identical to v2-A (mask shield, budget-aware, 500k steps, --envs 18, --lam-lr 0.5, seeds 0-2) with `--hide-wA`:
obs feature 4 ((w_A − 76)/10) is set to 0 in training and eval. Script `v2/addendum/run_noWA.sh` ->
`v2/addendum/A_noWA_s*`. Launched 15:12 CST, finished 17:39 CST, 218 iterations each, no errors,
0 unsafe in every iteration, λ = 0.
Eval: `python -m autopass_rl.evaluate --set v2nowa --settings main uniform --out results_rl/v2/addendum/eval_noWA`;
paired comparison: `python tools/nowa_compare.py` -> `eval_noWA/nowa_vs_v2A.{md,csv}`.

| setting (eval shield on) | noWA success % s0/s1/s2 | v2-A success % s0/s1/s2 | noWA timeouts | v2-A timeouts | unsafe |
|---|---|---|---|---|---|
| main | 97.0 / 98.3 / 98.3 (mean 97.9) | 98.3 / 96.7 / 98.3 (97.8) | 11 / 3 / 2 | 3 / 14 / 2 | 0 |
| uniform | 97.0 / 98.3 / 98.3 (97.9) | 98.3 / 96.3 / 98.2 (97.6) | 12 / 3 / 2 | 3 / 17 / 3 | 0 |

Time vs rule: noWA −1.0 ± 0.1 s (main), v2-A −1.0 ± 0.6 s; paired noWA − v2-A time per seed −0.5 to +0.7 s, all
Wilcoxon p > 0.15. Per-seed success differences are significant for seeds 0 and 1 (McNemar p < 0.01) but in
opposite directions and are all timeouts, i.e. which seed hits the known timeout loop, not an effect of w_A.
Without the shield at eval, noWA has more unsafe commits on seeds 0/1 (27.7 / 26.0 % vs 21.2 / 18.7 %) and
similar on seed 2 (21.8 vs 22.8 %); the mask shield removes all of them.
Conclusion: with the shield, hiding w_A makes no consistent difference. Caveat: only the explicit w_A feature is
hidden; the per-door fused-margin features (measured width − w_A) still carry w_A, so this does not show the
policy is independent of w_A.

### 2. fig_margin.png: sweeps per door vs |m̂ − w_A| / σ_m

`python tools/fig_margin.py --out results_rl/v2/addendum` (main setting, 600 seeds, rule + v2-A s0-2 with mask
shield). A decision = COMMIT or REJECT of a door; sweeps = trusted sweeps of that door fused at that moment.
Decisions with 0 trusted sweeps are excluded (rule 35, v2-A 28 pooled). Table: `fig_margin_table.csv`.
- Rule: ~25 sweeps for |m̂|/σ < 3 (never commits there; dwells to the cap and flags), ~10.5 at 3-4, ~5 at ≥ 10.
- v2-A: decides ambiguous doors rarely (15 decisions with |m̂|/σ < 3 over 3 seeds vs 100 for the rule);
  commits at 3-4 after ~4.8 sweeps (97 % commits). It tends to go to another door instead.

### 3. Report 2's no-repositioning setting

`python tools/nr_compare.py --out results_rl/v2/addendum` -> `nr_table.{md,csv}`, `nr_episodes.csv`.
Worlds: room + corridor, world seeds 1000-1299 (600), w_A = 81.3 cm, built with Report 2's
`make_world(default_rng(seed), layout)` and sensor seed `seed*101+7` (env reset option `world_seed`).
"No repositioning" in the RL env = GOTO limited to viewpoint 0 (square-on commit point). Report 2 planners
(B2-nr, P-A*(4)-nr, P-A*(4)) rerun with `autopass_sim.experiments` into `report2_rerun/`; matches Report 2's
committed `results_rev` on 1800/1800 missions (autopass_sim code identical to phase2-sim, so P-A*(4) is comparable).

| policy | success % | unsafe % | time s (all missions) |
|---|---|---|---|
| rule, RL env, no reposition | 98.5 | 0 | 28.0 |
| v2-A s0/s1/s2, no reposition | 98.7 / 97.3 / 98.5 | 0 | 25.4 / 26.5 / 25.8 |
| v2-A s0/s1/s2, all viewpoints | 99.0 / 97.5 / 98.7 | 0 | 25.9 / 26.6 / 26.5 |
| P-A*(4)-nr (Report 2) | 98.3 | 0 | 29.0 |
| B2-nr (Report 2) | 21.5 | 0 | 38.5 |
| P-A*(4) (Report 2, with repositioning) | 99.7 | 0 | 28.1 |

v2-A vs RL-env rule paired Δtime: −2.5 / −1.5 / −2.2 s (no reposition). Executors differ between the two
simulators (discrete viewpoints vs planned path), so cross-source time differences are indicative only.
v2-A s1 has 1.7 % timeouts here (same loop as above).

### 4. Figures

- `fig_train_curves.png` (`python tools/fig_train_curves.py`): return, cost, λ for v2-A and v2-B, mean of
  3 seeds (10-iteration moving average), min-max band. v2-B cost peaks ~0.24 and falls to ~0.02 as λ rises
  to ~7; v2-A cost and λ are 0 throughout (mask shield). Return does not penalize unsafe commits.
- `replay_rule_vs_v2A_5081.{mp4,png}` (`python tools/replay_rl.py --seed 5081 ...`, render2d.Panel +
  mission.Tracer; v2-A training seed 0). Seed choice rule: lowest main-setting eval seed with 3 doors, a
  wheelchair, ≥ 1 too-narrow door and both policies succeeding whose Δtime equals the 25th-percentile
  saving (−10 s). Rule: dwells on door 0 (0.75 cm margin) to the cap, flags it, commits door 1 (35.7 s,
  45 sweeps); v2-A skips door 0 and commits door 1 (25.7 s, 25 sweeps). Matches the eval CSV.
