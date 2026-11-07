# Report 3 deterministic evaluation (simulated)

### budget20 (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 50.5 | 0.0 | 49.5 | 0.0 | 7.82 [7.51, 8.15] | 16.3 [15.7, 16.9] | 17.9 [17.6, 18.2] | — | — | — |
| rule+shield | 50.5 | 0.0 | 49.5 | 0.0 | 7.82 [7.51, 8.13] | 16.3 [15.7, 16.9] | 17.9 [17.6, 18.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 62.2 ± 0.6 | 24.6 ± 1.0 | 13.2 ± 0.4 | 0.0 ± 0.0 | 9.88 ± 0.08 | 19.8 ± 0.1 | 15.5 ± 0.1 | +3.5 ± 0.1 | -2.4 ± 0.1 | 5.1e-22 / 4.3e-35 |
| v2 PPO-Lag(mask shield + budget), eval shield | 52.8 ± 0.3 | 0.0 ± 0.0 | 47.2 ± 0.3 | 0.0 ± 0.0 | 8.70 ± 0.18 | 17.4 ± 0.3 | 17.3 ± 0.0 | +1.1 ± 0.3 | -0.6 ± 0.0 | 6.2e-05 / 6.7e-07 |
| v2 PPO-Lag(no shield + budget), eval no shield | 60.8 ± 1.3 | 4.1 ± 0.8 | 35.2 ± 1.3 | 0.0 ± 0.0 | 9.55 ± 0.03 | 19.1 ± 0.1 | 17.2 ± 0.3 | +2.8 ± 0.1 | -0.8 ± 0.3 | 3.3e-19 / 0.0011 |
| v2 PPO-Lag(no shield + budget), eval shield | 53.0 ± 1.3 | 0.0 ± 0.0 | 47.0 ± 1.3 | 0.0 ± 0.0 | 8.67 ± 0.03 | 17.4 ± 0.1 | 17.3 ± 0.3 | +1.0 ± 0.1 | -0.6 ± 0.3 | 2.8e-10 / 0.09 |
| v1 PPO-Lag(train shield, post shield), eval shield | 51.8 ± 0.9 | 0.0 ± 0.0 | 35.0 ± 2.9 | 13.2 ± 2.1 | 8.75 ± 0.11 | 24.0 ± 0.3 | 17.5 ± 0.1 | +7.7 ± 0.3 | -0.4 ± 0.1 | 1.1e-25 / 0.047 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
