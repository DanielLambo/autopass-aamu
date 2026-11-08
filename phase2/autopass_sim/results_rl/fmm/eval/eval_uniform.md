# Report 3 deterministic evaluation (simulated)

### uniform (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.5 | 0.0 | 2.5 | 0.0 | 12.16 [11.79, 12.54] | 27.5 [26.6, 28.5] | 32.7 [31.1, 34.4] | — | — | — |
| rule+shield | 97.5 | 0.0 | 2.5 | 0.0 | 12.16 [11.80, 12.51] | 27.5 [26.5, 28.5] | 32.7 [31.1, 34.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 78.8 ± 1.2 | 20.5 ± 2.2 | 0.0 ± 0.0 | 0.7 ± 1.1 | 10.64 ± 0.17 | 21.6 ± 0.4 | 19.0 ± 1.1 | -6.0 ± 0.4 | -13.7 ± 1.1 | 1.7e-30 / 2.1e-42 |
| v2 PPO-Lag(mask shield + budget), eval shield | 97.6 ± 1.1 | 0.0 ± 0.0 | 1.2 ± 0.3 | 1.3 ± 1.3 | 13.02 ± 0.58 | 26.5 ± 0.6 | 29.5 ± 1.0 | -1.0 ± 0.6 | -3.2 ± 1.0 | 1.7e-06 / 1.4e-12 |
| v2 PPO-Lag(no shield + budget), eval no shield | 96.3 ± 0.6 | 0.2 ± 0.0 | 0.0 ± 0.0 | 3.6 ± 0.6 | 15.24 ± 0.38 | 30.5 ± 0.8 | 37.9 ± 1.6 | +3.0 ± 0.8 | +5.2 ± 1.6 | 0.72 / 5.1e-07 |
| v2 PPO-Lag(no shield + budget), eval shield | 96.3 ± 0.6 | 0.0 ± 0.0 | 0.1 ± 0.1 | 3.7 ± 0.6 | 15.41 ± 0.40 | 30.9 ± 0.9 | 38.7 ± 1.7 | +3.4 ± 0.9 | +6.0 ± 1.7 | 0.93 / 2.5e-06 |
| v1 PPO-Lag(train shield, post shield), eval shield | 98.2 ± 0.2 | 0.0 ± 0.0 | 0.9 ± 0.2 | 0.9 ± 0.3 | 12.95 ± 0.17 | 26.2 ± 0.3 | 28.6 ± 0.9 | -1.3 ± 0.3 | -4.1 ± 0.9 | 9e-05 / 4.2e-14 |
| v1 PPO-Lag, eval no shield | 97.9 ± 0.1 | 1.5 ± 0.6 | 0.0 ± 0.0 | 0.6 ± 0.5 | 12.48 ± 0.38 | 25.0 ± 0.8 | 26.0 ± 1.8 | -2.5 ± 0.8 | -6.7 ± 1.8 | 1.3e-15 / 5.9e-26 |
| v1 PPO-Lag, eval shield | 97.4 ± 0.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.6 ± 0.3 | 12.42 ± 0.37 | 25.9 ± 0.5 | 28.5 ± 1.3 | -1.6 ± 0.5 | -4.2 ± 1.3 | 8.6e-11 / 2.9e-17 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v2 PPO-Lag(mask shield + budget), eval no shield | 0 | 79.0 | -18.5 | 5 | 116 | 1.56e-28 |
| v2 PPO-Lag(mask shield + budget), eval shield | 0 | 98.2 | +0.7 | 5 | 1 | 0.219 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 1 | 79.8 | -17.7 | 3 | 109 | 9.02e-29 |
| v2 PPO-Lag(mask shield + budget), eval shield | 1 | 96.3 | -1.2 | 2 | 9 | 0.0654 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 2 | 77.5 | -20.0 | 6 | 126 | 2.52e-30 |
| v2 PPO-Lag(mask shield + budget), eval shield | 2 | 98.2 | +0.7 | 5 | 1 | 0.219 |
| v2 PPO-Lag(no shield + budget), eval no shield | 0 | 97.0 | -0.5 | 5 | 8 | 0.581 |
| v2 PPO-Lag(no shield + budget), eval shield | 0 | 97.0 | -0.5 | 5 | 8 | 0.581 |
| v2 PPO-Lag(no shield + budget), eval no shield | 1 | 95.8 | -1.7 | 3 | 13 | 0.0213 |
| v2 PPO-Lag(no shield + budget), eval shield | 1 | 95.8 | -1.7 | 3 | 13 | 0.0213 |
| v2 PPO-Lag(no shield + budget), eval no shield | 2 | 96.0 | -1.5 | 2 | 11 | 0.0225 |
| v2 PPO-Lag(no shield + budget), eval shield | 2 | 96.0 | -1.5 | 2 | 11 | 0.0225 |
| v1 PPO-Lag(train shield, post shield), eval shield | 0 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v1 PPO-Lag(train shield, post shield), eval shield | 1 | 98.0 | +0.5 | 4 | 1 | 0.375 |
| v1 PPO-Lag(train shield, post shield), eval shield | 2 | 98.2 | +0.7 | 4 | 0 | 0.125 |
| v1 PPO-Lag, eval no shield | 0 | 97.8 | +0.3 | 6 | 4 | 0.754 |
| v1 PPO-Lag, eval shield | 0 | 97.5 | +0.0 | 3 | 3 | 1 |
| v1 PPO-Lag, eval no shield | 1 | 98.0 | +0.5 | 6 | 3 | 0.508 |
| v1 PPO-Lag, eval shield | 1 | 97.0 | -0.5 | 3 | 6 | 0.508 |
| v1 PPO-Lag, eval no shield | 2 | 98.0 | +0.5 | 6 | 3 | 0.508 |
| v1 PPO-Lag, eval shield | 2 | 97.7 | +0.2 | 4 | 3 | 1 |
