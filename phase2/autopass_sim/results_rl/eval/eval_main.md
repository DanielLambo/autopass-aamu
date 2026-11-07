# Report 3 deterministic evaluation (simulated)

### main (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.5 | 0.0 | 2.5 | 0.0 | 12.18 [11.82, 12.57] | 27.6 [26.6, 28.6] | 32.7 [31.1, 34.3] | — | — | — |
| rule+shield | 97.5 | 0.0 | 2.5 | 0.0 | 12.18 [11.82, 12.55] | 27.6 [26.6, 28.7] | 32.7 [31.1, 34.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| PPO-Lag, eval no shield | 98.3 ± 0.1 | 1.3 ± 0.5 | 0.0 ± 0.0 | 0.4 ± 0.5 | 12.37 ± 0.41 | 24.8 ± 0.8 | 25.5 ± 1.9 | -2.8 ± 0.8 | -7.2 ± 1.9 | 1.3e-16 / 5e-26 |
| PPO-Lag, eval shield | 97.7 ± 0.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.3 ± 0.2 | 12.34 ± 0.40 | 25.7 ± 0.4 | 27.8 ± 1.0 | -1.9 ± 0.4 | -4.9 ± 1.0 | 4.7e-11 / 1.2e-18 |
| PPO-Lag(train shield), eval no shield | 98.2 ± 0.2 | 0.8 ± 0.4 | 0.7 ± 0.3 | 0.3 ± 0.2 | 12.95 ± 0.21 | 25.9 ± 0.4 | 27.7 ± 1.1 | -1.7 ± 0.4 | -5.0 ± 1.1 | 1.4e-06 / 1.2e-16 |
| PPO-Lag(train shield), eval shield | 98.0 ± 0.3 | 0.0 ± 0.0 | 1.0 ± 0.2 | 1.0 ± 0.4 | 12.93 ± 0.19 | 26.2 ± 0.3 | 28.6 ± 0.8 | -1.4 ± 0.3 | -4.1 ± 0.8 | 1.8e-05 / 2.1e-14 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
