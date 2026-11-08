# Report 3 deterministic evaluation (simulated)

### budget20 (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 50.5 | 0.0 | 49.5 | 0.0 | 7.82 [7.51, 8.15] | 16.3 [15.8, 16.9] | 17.9 [17.6, 18.2] | — | — | — |
| rule+shield | 50.5 | 0.0 | 49.5 | 0.0 | 7.82 [7.52, 8.13] | 16.3 [15.8, 16.9] | 17.9 [17.6, 18.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 62.1 ± 0.5 | 24.7 ± 0.9 | 13.2 ± 0.4 | 0.0 ± 0.0 | 9.88 ± 0.07 | 19.8 ± 0.1 | 15.5 ± 0.1 | +3.5 ± 0.1 | -2.5 ± 0.1 | 1.4e-22 / 1e-34 |
| v2 PPO-Lag(mask shield + budget), eval shield | 53.5 ± 0.3 | 0.0 ± 0.0 | 46.5 ± 0.3 | 0.0 ± 0.0 | 8.72 ± 0.19 | 17.5 ± 0.3 | 17.3 ± 0.0 | +1.2 ± 0.3 | -0.6 ± 0.0 | 7.4e-05 / 6.8e-08 |
| v2 PPO-Lag(no shield + budget), eval no shield | 61.1 ± 1.2 | 3.6 ± 0.3 | 35.3 ± 1.0 | 0.0 ± 0.0 | 9.54 ± 0.04 | 19.1 ± 0.1 | 17.2 ± 0.3 | +2.8 ± 0.1 | -0.7 ± 0.3 | 4.6e-19 / 0.0016 |
| v2 PPO-Lag(no shield + budget), eval shield | 53.0 ± 1.4 | 0.0 ± 0.0 | 47.0 ± 1.4 | 0.0 ± 0.0 | 8.67 ± 0.04 | 17.3 ± 0.1 | 17.3 ± 0.3 | +1.0 ± 0.1 | -0.6 ± 0.3 | 4e-09 / 0.054 |
| v1 PPO-Lag(train shield, post shield), eval shield | 52.1 ± 1.1 | 0.0 ± 0.0 | 34.4 ± 2.9 | 13.5 ± 1.8 | 8.74 ± 0.08 | 23.9 ± 0.2 | 17.5 ± 0.1 | +7.6 ± 0.2 | -0.5 ± 0.1 | 5.2e-26 / 0.018 |
| v1 PPO-Lag, eval no shield | 62.9 ± 0.3 | 21.3 ± 4.6 | 15.8 ± 4.6 | 0.0 ± 0.0 | 11.15 ± 0.45 | 22.3 ± 0.9 | 17.1 ± 0.1 | +6.0 ± 0.9 | -0.9 ± 0.1 | 2.5e-26 / 6.2e-11 |
| v1 PPO-Lag, eval shield | 52.9 ± 0.9 | 0.0 ± 0.0 | 15.8 ± 4.6 | 31.3 ± 5.5 | 8.51 ± 0.06 | 29.3 ± 2.0 | 17.4 ± 0.0 | +13.0 ± 2.0 | -0.6 ± 0.0 | 4.9e-32 / 3.9e-07 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v2 PPO-Lag(mask shield + budget), eval no shield | 0 | 61.7 | +11.2 | 90 | 23 | 1.49e-10 |
| v2 PPO-Lag(mask shield + budget), eval shield | 0 | 53.8 | +3.3 | 47 | 27 | 0.0265 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 1 | 62.7 | +12.2 | 91 | 18 | 6.37e-13 |
| v2 PPO-Lag(mask shield + budget), eval shield | 1 | 53.2 | +2.7 | 39 | 23 | 0.0559 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 2 | 62.0 | +11.5 | 92 | 23 | 5.79e-11 |
| v2 PPO-Lag(mask shield + budget), eval shield | 2 | 53.5 | +3.0 | 46 | 28 | 0.0474 |
| v2 PPO-Lag(no shield + budget), eval no shield | 0 | 61.0 | +10.5 | 80 | 17 | 6.01e-11 |
| v2 PPO-Lag(no shield + budget), eval shield | 0 | 52.8 | +2.3 | 40 | 26 | 0.109 |
| v2 PPO-Lag(no shield + budget), eval no shield | 1 | 60.0 | +9.5 | 79 | 22 | 1.01e-08 |
| v2 PPO-Lag(no shield + budget), eval shield | 1 | 51.7 | +1.2 | 39 | 32 | 0.477 |
| v2 PPO-Lag(no shield + budget), eval no shield | 2 | 62.3 | +11.8 | 86 | 15 | 2.83e-13 |
| v2 PPO-Lag(no shield + budget), eval shield | 2 | 54.5 | +4.0 | 46 | 22 | 0.0049 |
| v1 PPO-Lag(train shield, post shield), eval shield | 0 | 50.8 | +0.3 | 38 | 36 | 0.908 |
| v1 PPO-Lag(train shield, post shield), eval shield | 1 | 52.7 | +2.2 | 41 | 28 | 0.148 |
| v1 PPO-Lag(train shield, post shield), eval shield | 2 | 52.8 | +2.3 | 41 | 27 | 0.114 |
| v1 PPO-Lag, eval no shield | 0 | 63.2 | +12.7 | 93 | 17 | 7.24e-14 |
| v1 PPO-Lag, eval shield | 0 | 52.2 | +1.7 | 34 | 24 | 0.237 |
| v1 PPO-Lag, eval no shield | 1 | 62.8 | +12.3 | 88 | 14 | 2.78e-14 |
| v1 PPO-Lag, eval shield | 1 | 53.8 | +3.3 | 37 | 17 | 0.00907 |
| v1 PPO-Lag, eval no shield | 2 | 62.7 | +12.2 | 88 | 15 | 9.65e-14 |
| v1 PPO-Lag, eval shield | 2 | 52.7 | +2.2 | 33 | 20 | 0.0984 |
