# Report 3 deterministic evaluation (simulated)

### uniform (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.7 | 0.0 | 2.3 | 0.0 | 12.18 [11.81, 12.57] | 27.6 [26.6, 28.6] | 32.6 [31.0, 34.3] | — | — | — |
| rule+shield | 97.7 | 0.0 | 2.3 | 0.0 | 12.18 [11.82, 12.54] | 27.6 [26.5, 28.6] | 32.6 [31.0, 34.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| PPO-Lag, eval no shield | 97.8 ± 0.1 | 1.6 ± 0.7 | 0.0 ± 0.0 | 0.6 ± 0.6 | 12.48 ± 0.49 | 25.0 ± 1.0 | 26.0 ± 2.3 | -2.6 ± 1.0 | -6.6 ± 2.3 | 5.4e-15 / 6.6e-25 |
| PPO-Lag, eval shield | 97.3 ± 0.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.7 ± 0.3 | 12.43 ± 0.46 | 26.0 ± 0.7 | 28.6 ± 1.5 | -1.6 ± 0.7 | -4.0 ± 1.5 | 8.1e-11 / 5.8e-17 |
| PPO-Lag(train shield), eval no shield | 98.3 ± 0.2 | 0.6 ± 0.3 | 0.7 ± 0.2 | 0.4 ± 0.3 | 12.96 ± 0.17 | 25.9 ± 0.3 | 27.8 ± 1.0 | -1.6 ± 0.3 | -4.8 ± 1.0 | 5.1e-06 / 1e-16 |
| PPO-Lag(train shield), eval shield | 98.2 ± 0.2 | 0.0 ± 0.0 | 0.9 ± 0.2 | 0.9 ± 0.3 | 12.94 ± 0.14 | 26.2 ± 0.2 | 28.6 ± 0.8 | -1.4 ± 0.2 | -4.0 ± 0.8 | 5e-05 / 6e-14 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
