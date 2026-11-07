# results_rl: Report 3 reinforcement-learning results (simulated)

Run log with commands, code commits and decisions: [RUNS.md](RUNS.md).
Training runs hold `config.json` (arguments), `train.csv` (per-iteration log) and `agent.pt` (final weights).
Console logs (`*.log`) are not tracked; `train.csv` holds the same numbers.

| Path | What | Code |
|---|---|---|
| `sanity.json` | env sanity check (rule / random, ± shield) | report3-env `4a5c148` |
| `pilot_oldcode/` | s0 pilots on pre-speedup code (+ an aborted s0 run) | `91c1caf` + route cache |
| `ppolag_s{0,1,2}/`, `ppolag_shield_s{0,1,2}/`, `run_full.sh` | v1: PPO-Lag without / with post shield | report3-env `4a5c148` |
| `eval/` | v1 deterministic eval (600 seeds, 4 settings) + figure | report3-env `5b34a20` |
| `v2/A_mask_budget_s*/` | **v2-A, reported method**: mask shield + budget-aware | report3-v2 `2ad8d1d` |
| `v2/B_noshield_budget_s*/` | v2-B ablation: budget-aware, no shield (Lagrangian only) | report3-v2 `2ad8d1d` |
| `v2/eval/`, `v2/run_v2.sh` | **v2 eval tables + fig_main.png (report tables)** | report3-v2 `567501b` |
| `v3/A_mask_budget_cap20_s*/`, `v3/run_v3.sh` | v3: v2-A + per-door dwell cap 20 (negative result) | report3-v3-dwellcap `e97e8a4` |
| `v3/eval/` | v3 eval incl. McNemar tables | report3-v3-dwellcap `db903e4` |
