# Report 3 deterministic evaluation (simulated)

### main (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.5 | 0.0 | 2.5 | 0.0 | 12.18 [11.82, 12.57] | 27.6 [26.6, 28.6] | 32.7 [31.1, 34.3] | — | — | — |
| rule+shield | 97.5 | 0.0 | 2.5 | 0.0 | 12.18 [11.82, 12.55] | 27.6 [26.6, 28.7] | 32.7 [31.1, 34.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 78.5 ± 1.3 | 20.9 ± 2.1 | 0.0 ± 0.0 | 0.6 ± 0.9 | 10.60 ± 0.16 | 21.4 ± 0.3 | 18.8 ± 0.9 | -6.2 ± 0.3 | -13.9 ± 0.9 | 6e-32 / 1.4e-44 |
| v2 PPO-Lag(mask shield + budget), eval shield | 97.8 ± 1.0 | 0.0 ± 0.0 | 1.2 ± 0.2 | 1.1 ± 1.1 | 13.10 ± 0.59 | 26.6 ± 0.6 | 29.5 ± 1.2 | -1.0 ± 0.6 | -3.2 ± 1.2 | 3.4e-07 / 2.9e-14 |
| v2 PPO-Lag(no shield + budget), eval no shield | 96.2 ± 0.4 | 0.2 ± 0.2 | 0.0 ± 0.0 | 3.6 ± 0.5 | 15.17 ± 0.40 | 30.4 ± 0.8 | 37.5 ± 1.4 | +2.8 ± 0.8 | +4.8 ± 1.4 | 0.33 / 1.2e-07 |
| v2 PPO-Lag(no shield + budget), eval shield | 96.2 ± 0.4 | 0.0 ± 0.0 | 0.2 ± 0.2 | 3.7 ± 0.4 | 15.38 ± 0.37 | 30.8 ± 0.8 | 38.6 ± 1.3 | +3.2 ± 0.8 | +5.9 ± 1.3 | 0.54 / 3.5e-07 |
| v1 PPO-Lag(train shield, post shield), eval shield | 98.0 ± 0.3 | 0.0 ± 0.0 | 1.0 ± 0.2 | 1.0 ± 0.4 | 12.93 ± 0.19 | 26.2 ± 0.3 | 28.6 ± 0.8 | -1.4 ± 0.3 | -4.1 ± 0.8 | 1.8e-05 / 2.1e-14 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
