# Report 3 deterministic evaluation (simulated)

### budget12 (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 29.3 | 0.0 | 70.7 | 0.0 | 5.96 [5.69, 6.24] | 12.0 [11.5, 12.6] | 11.7 [11.6, 11.8] | — | — | — |
| rule+shield | 29.3 | 0.0 | 70.7 | 0.0 | 5.96 [5.70, 6.22] | 12.0 [11.5, 12.6] | 11.7 [11.6, 11.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 42.9 ± 0.2 | 18.9 ± 0.3 | 38.2 ± 0.4 | 0.0 ± 0.0 | 7.85 ± 0.04 | 15.7 ± 0.1 | 11.1 ± 0.0 | +3.7 ± 0.1 | -0.6 ± 0.0 | 4.3e-28 / 5.6e-09 |
| v2 PPO-Lag(mask shield + budget), eval shield | 33.1 ± 0.3 | 0.0 ± 0.0 | 66.9 ± 0.3 | 0.0 ± 0.0 | 6.20 ± 0.05 | 12.4 ± 0.1 | 11.6 ± 0.0 | +0.4 ± 0.1 | -0.1 ± 0.0 | 0.088 / 0.0036 |
| v2 PPO-Lag(no shield + budget), eval no shield | 40.8 ± 0.6 | 3.6 ± 1.1 | 55.7 ± 1.7 | 0.0 ± 0.0 | 7.08 ± 0.22 | 14.2 ± 0.4 | 11.5 ± 0.0 | +2.1 ± 0.4 | -0.2 ± 0.0 | 3.8e-08 / 0.057 |
| v2 PPO-Lag(no shield + budget), eval shield | 32.7 ± 0.5 | 0.0 ± 0.0 | 67.3 ± 0.5 | 0.0 ± 0.0 | 6.25 ± 0.08 | 12.5 ± 0.2 | 11.5 ± 0.0 | +0.5 ± 0.2 | -0.2 ± 0.0 | 0.32 / 0.1 |
| v1 PPO-Lag(train shield, post shield), eval shield | 30.8 ± 1.7 | 0.0 ± 0.0 | 52.1 ± 4.3 | 17.1 ± 3.3 | 6.21 ± 0.06 | 20.0 ± 0.8 | 11.6 ± 0.1 | +7.9 ± 0.8 | -0.1 ± 0.1 | 5.6e-25 / 0.73 |
| v1 PPO-Lag, eval no shield | 41.1 ± 0.6 | 17.2 ± 2.2 | 41.7 ± 1.5 | 0.0 ± 0.0 | 7.96 ± 0.12 | 15.9 ± 0.2 | 11.5 ± 0.1 | +3.9 ± 0.2 | -0.2 ± 0.1 | 1.2e-25 / 0.0024 |
| v1 PPO-Lag, eval shield | 31.2 ± 1.0 | 0.0 ± 0.0 | 41.8 ± 1.6 | 27.1 ± 2.7 | 6.11 ± 0.06 | 22.9 ± 0.9 | 11.5 ± 0.0 | +10.9 ± 0.9 | -0.1 ± 0.0 | 5.5e-32 / 0.0036 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v2 PPO-Lag(mask shield + budget), eval no shield | 0 | 43.0 | +13.7 | 89 | 7 | 3.26e-19 |
| v2 PPO-Lag(mask shield + budget), eval shield | 0 | 33.3 | +4.0 | 31 | 7 | 0.000116 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 1 | 42.7 | +13.3 | 85 | 5 | 7.53e-20 |
| v2 PPO-Lag(mask shield + budget), eval shield | 1 | 33.0 | +3.7 | 27 | 5 | 0.000113 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 2 | 43.0 | +13.7 | 90 | 8 | 1.09e-18 |
| v2 PPO-Lag(mask shield + budget), eval shield | 2 | 32.8 | +3.5 | 31 | 10 | 0.00145 |
| v2 PPO-Lag(no shield + budget), eval no shield | 0 | 41.3 | +12.0 | 79 | 7 | 1.52e-16 |
| v2 PPO-Lag(no shield + budget), eval shield | 0 | 32.7 | +3.3 | 30 | 10 | 0.00222 |
| v2 PPO-Lag(no shield + budget), eval no shield | 1 | 40.8 | +11.5 | 80 | 11 | 4.41e-14 |
| v2 PPO-Lag(no shield + budget), eval shield | 1 | 32.2 | +2.8 | 31 | 14 | 0.0161 |
| v2 PPO-Lag(no shield + budget), eval no shield | 2 | 40.2 | +10.8 | 74 | 9 | 7.71e-14 |
| v2 PPO-Lag(no shield + budget), eval shield | 2 | 33.2 | +3.8 | 34 | 11 | 0.000824 |
| v1 PPO-Lag(train shield, post shield), eval shield | 0 | 28.8 | -0.5 | 19 | 22 | 0.755 |
| v1 PPO-Lag(train shield, post shield), eval shield | 1 | 32.2 | +2.8 | 29 | 12 | 0.0115 |
| v1 PPO-Lag(train shield, post shield), eval shield | 2 | 31.3 | +2.0 | 25 | 13 | 0.073 |
| v1 PPO-Lag, eval no shield | 0 | 40.7 | +11.3 | 80 | 12 | 1.72e-13 |
| v1 PPO-Lag, eval shield | 0 | 30.3 | +1.0 | 20 | 14 | 0.392 |
| v1 PPO-Lag, eval no shield | 1 | 41.8 | +12.5 | 79 | 4 | 4e-19 |
| v1 PPO-Lag, eval shield | 1 | 32.3 | +3.0 | 23 | 5 | 0.000912 |
| v1 PPO-Lag, eval no shield | 2 | 40.8 | +11.5 | 79 | 10 | 1.87e-14 |
| v1 PPO-Lag, eval shield | 2 | 30.8 | +1.5 | 20 | 11 | 0.15 |
