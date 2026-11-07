# Report 3 deterministic evaluation (simulated)

### budget12 (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 30.0 | 0.0 | 70.0 | 0.0 | 6.03 [5.77, 6.31] | 12.2 [11.7, 12.7] | 11.7 [11.6, 11.8] | — | — | — |
| rule+shield | 30.0 | 0.0 | 70.0 | 0.0 | 6.03 [5.77, 6.30] | 12.2 [11.7, 12.7] | 11.7 [11.5, 11.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| PPO-Lag, eval no shield | 41.4 ± 0.4 | 16.8 ± 2.4 | 41.8 ± 2.0 | 0.0 ± 0.0 | 7.96 ± 0.14 | 15.9 ± 0.3 | 11.5 ± 0.1 | +3.7 ± 0.3 | -0.2 ± 0.1 | 1.5e-23 / 0.0056 |
| PPO-Lag, eval shield | 30.9 ± 0.9 | 0.0 ± 0.0 | 41.8 ± 2.0 | 27.3 ± 2.9 | 6.10 ± 0.06 | 23.0 ± 1.0 | 11.5 ± 0.0 | +10.8 ± 1.0 | -0.1 ± 0.0 | 1.2e-30 / 0.0056 |
| PPO-Lag(train shield), eval no shield | 40.7 ± 1.3 | 11.1 ± 1.4 | 48.2 ± 2.4 | 0.0 ± 0.0 | 7.70 ± 0.08 | 15.4 ± 0.2 | 11.6 ± 0.1 | +3.2 ± 0.2 | -0.1 ± 0.1 | 1.5e-19 / 0.81 |
| PPO-Lag(train shield), eval shield | 30.4 ± 1.7 | 0.0 ± 0.0 | 52.8 ± 3.8 | 16.8 ± 2.7 | 6.19 ± 0.06 | 19.8 ± 0.7 | 11.6 ± 0.1 | +7.7 ± 0.7 | -0.1 ± 0.1 | 2.7e-23 / 0.81 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
