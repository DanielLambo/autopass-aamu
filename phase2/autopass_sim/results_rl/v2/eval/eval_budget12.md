# Report 3 deterministic evaluation (simulated)

### budget12 (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 30.0 | 0.0 | 70.0 | 0.0 | 6.03 [5.77, 6.31] | 12.2 [11.7, 12.7] | 11.7 [11.6, 11.8] | — | — | — |
| rule+shield | 30.0 | 0.0 | 70.0 | 0.0 | 6.03 [5.77, 6.30] | 12.2 [11.7, 12.7] | 11.7 [11.5, 11.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 42.9 ± 0.3 | 18.8 ± 0.3 | 38.2 ± 0.5 | 0.0 ± 0.0 | 7.86 ± 0.05 | 15.7 ± 0.1 | 11.1 ± 0.0 | +3.5 ± 0.1 | -0.6 ± 0.0 | 3.6e-26 / 6.3e-09 |
| v2 PPO-Lag(mask shield + budget), eval shield | 32.4 ± 0.2 | 0.0 ± 0.0 | 67.6 ± 0.2 | 0.0 ± 0.0 | 6.16 ± 0.06 | 12.3 ± 0.1 | 11.6 ± 0.0 | +0.2 ± 0.1 | -0.1 ± 0.0 | 0.82 / 0.0056 |
| v2 PPO-Lag(no shield + budget), eval no shield | 40.6 ± 0.6 | 3.4 ± 1.1 | 56.1 ± 1.6 | 0.0 ± 0.0 | 7.06 ± 0.22 | 14.1 ± 0.4 | 11.5 ± 0.1 | +1.9 ± 0.4 | -0.2 ± 0.1 | 1.2e-06 / 0.081 |
| v2 PPO-Lag(no shield + budget), eval shield | 32.2 ± 0.8 | 0.0 ± 0.0 | 67.8 ± 0.8 | 0.0 ± 0.0 | 6.20 ± 0.07 | 12.4 ± 0.1 | 11.5 ± 0.0 | +0.2 ± 0.1 | -0.2 ± 0.0 | 0.99 / 0.15 |
| v1 PPO-Lag(train shield, post shield), eval shield | 30.4 ± 1.7 | 0.0 ± 0.0 | 52.8 ± 3.8 | 16.8 ± 2.7 | 6.19 ± 0.06 | 19.8 ± 0.7 | 11.6 ± 0.1 | +7.7 ± 0.7 | -0.1 ± 0.1 | 2.7e-23 / 0.81 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
