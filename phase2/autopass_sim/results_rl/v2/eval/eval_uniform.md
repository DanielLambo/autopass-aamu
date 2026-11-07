# Report 3 deterministic evaluation (simulated)

### uniform (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.7 | 0.0 | 2.3 | 0.0 | 12.18 [11.81, 12.57] | 27.6 [26.6, 28.6] | 32.6 [31.0, 34.3] | — | — | — |
| rule+shield | 97.7 | 0.0 | 2.3 | 0.0 | 12.18 [11.82, 12.54] | 27.6 [26.5, 28.6] | 32.6 [31.0, 34.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 78.7 ± 1.3 | 20.6 ± 2.4 | 0.0 ± 0.0 | 0.7 ± 1.1 | 10.62 ± 0.16 | 21.5 ± 0.4 | 19.0 ± 1.1 | -6.1 ± 0.4 | -13.7 ± 1.1 | 1e-31 / 2.5e-42 |
| v2 PPO-Lag(mask shield + budget), eval shield | 97.6 ± 1.1 | 0.0 ± 0.0 | 1.1 ± 0.3 | 1.3 ± 1.3 | 13.01 ± 0.57 | 26.5 ± 0.5 | 29.4 ± 0.9 | -1.1 ± 0.5 | -3.3 ± 0.9 | 5e-07 / 5.9e-13 |
| v2 PPO-Lag(no shield + budget), eval no shield | 96.3 ± 0.5 | 0.1 ± 0.1 | 0.1 ± 0.1 | 3.6 ± 0.4 | 15.18 ± 0.31 | 30.4 ± 0.7 | 37.7 ± 1.1 | +2.8 ± 0.7 | +5.0 ± 1.1 | 0.65 / 2e-07 |
| v2 PPO-Lag(no shield + budget), eval shield | 96.2 ± 0.5 | 0.0 ± 0.0 | 0.1 ± 0.2 | 3.7 ± 0.5 | 15.33 ± 0.40 | 30.7 ± 0.9 | 38.4 ± 1.5 | +3.2 ± 0.9 | +5.7 ± 1.5 | 0.81 / 1.4e-06 |
| v1 PPO-Lag(train shield, post shield), eval shield | 98.2 ± 0.2 | 0.0 ± 0.0 | 0.9 ± 0.2 | 0.9 ± 0.3 | 12.94 ± 0.14 | 26.2 ± 0.2 | 28.6 ± 0.8 | -1.4 ± 0.2 | -4.0 ± 0.8 | 5e-05 / 6e-14 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
