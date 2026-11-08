# Report 3 results

Constrained RL on the AutoPass simulator, plus a check in Isaac Lab. All results are simulated.
Commands, commits, timings and findings for every run are in [RUNS.md](RUNS.md).

Each training run folder holds `config.json` (arguments), `train.csv` (one row per iteration) and
`agent.pt` (final weights). Console logs are not tracked; `train.csv` has the same numbers.

| Folder | Contents |
|---|---|
| `v1/` | First version: PPO-Lagrangian with and without the post-hoc shield, seed-0 pilots, env sanity check, evaluation |
| `v2/A_mask_budget_s*/` | **Reported method (v2-A)**: mask shield, budget-aware |
| `v2/B_noshield_budget_s*/` | Ablation (v2-B): budget-aware, no shield, Lagrangian only |
| `v2/eval/` | **Report tables and `fig_main.png`** (600 episodes × 4 settings) |
| `v2/addendum/` | w_A ablation, sweeps vs margin figure, no-repositioning comparison with Report 2, training curves, rule vs v2-A replay |
| `v3/` | Per-door dwell cap (negative result; code in tag `report3-v3-dwellcap`) |
| `gui/` | Go2 walking policy carrying out v2-A's plan in Isaac Lab: videos, screenshots, walkability logs |
