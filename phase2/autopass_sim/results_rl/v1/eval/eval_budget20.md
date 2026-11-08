# Report 3 deterministic evaluation (simulated)

### budget20 (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 50.5 | 0.0 | 49.5 | 0.0 | 7.82 [7.51, 8.15] | 16.3 [15.7, 16.9] | 17.9 [17.6, 18.2] | — | — | — |
| rule+shield | 50.5 | 0.0 | 49.5 | 0.0 | 7.82 [7.51, 8.13] | 16.3 [15.7, 16.9] | 17.9 [17.6, 18.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| PPO-Lag, eval no shield | 63.1 ± 0.2 | 21.2 ± 4.5 | 15.7 ± 4.6 | 0.0 ± 0.0 | 11.16 ± 0.40 | 22.3 ± 0.8 | 17.1 ± 0.1 | +6.0 ± 0.8 | -0.8 ± 0.1 | 1.9e-26 / 2.4e-09 |
| PPO-Lag, eval shield | 52.9 ± 0.7 | 0.0 ± 0.0 | 15.7 ± 4.6 | 31.3 ± 5.1 | 8.52 ± 0.05 | 29.3 ± 1.9 | 17.4 ± 0.0 | +13.0 ± 1.9 | -0.6 ± 0.0 | 3.7e-32 / 2.4e-06 |
| PPO-Lag(train shield), eval no shield | 61.8 ± 0.3 | 10.3 ± 0.1 | 27.9 ± 0.3 | 0.0 ± 0.0 | 10.48 ± 0.23 | 21.0 ± 0.5 | 17.4 ± 0.2 | +4.7 ± 0.5 | -0.5 ± 0.2 | 7.7e-23 / 0.0099 |
| PPO-Lag(train shield), eval shield | 51.8 ± 0.9 | 0.0 ± 0.0 | 35.0 ± 2.9 | 13.2 ± 2.1 | 8.75 ± 0.11 | 24.0 ± 0.3 | 17.5 ± 0.1 | +7.7 ± 0.3 | -0.4 ± 0.1 | 1.1e-25 / 0.047 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
