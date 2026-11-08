# Report 3 deterministic evaluation (simulated)

### main (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.5 | 0.0 | 2.5 | 0.0 | 12.19 [11.82, 12.57] | 27.6 [26.6, 28.6] | 32.7 [31.2, 34.4] | — | — | — |
| rule+shield | 97.5 | 0.0 | 2.5 | 0.0 | 12.19 [11.83, 12.55] | 27.6 [26.5, 28.6] | 32.7 [31.2, 34.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 78.6 ± 1.3 | 20.8 ± 2.2 | 0.0 ± 0.0 | 0.6 ± 0.9 | 10.59 ± 0.17 | 21.4 ± 0.3 | 18.8 ± 0.9 | -6.2 ± 0.3 | -14.0 ± 0.9 | 4.2e-32 / 1.4e-44 |
| v2 PPO-Lag(mask shield + budget), eval shield | 97.8 ± 1.0 | 0.0 ± 0.0 | 1.2 ± 0.2 | 1.0 ± 1.2 | 12.97 ± 0.51 | 26.3 ± 0.5 | 29.0 ± 0.9 | -1.2 ± 0.5 | -3.8 ± 0.9 | 1.3e-07 / 1.3e-15 |
| v2 PPO-Lag(no shield + budget), eval no shield | 96.3 ± 0.5 | 0.1 ± 0.1 | 0.1 ± 0.1 | 3.6 ± 0.3 | 15.22 ± 0.19 | 30.5 ± 0.4 | 37.7 ± 0.6 | +2.9 ± 0.4 | +5.0 ± 0.6 | 0.29 / 1.1e-06 |
| v2 PPO-Lag(no shield + budget), eval shield | 96.3 ± 0.5 | 0.0 ± 0.0 | 0.1 ± 0.1 | 3.6 ± 0.4 | 15.28 ± 0.22 | 30.6 ± 0.5 | 38.1 ± 0.7 | +3.0 ± 0.5 | +5.4 ± 0.7 | 0.47 / 1.1e-06 |
| v1 PPO-Lag(train shield, post shield), eval shield | 98.1 ± 0.1 | 0.0 ± 0.0 | 1.0 ± 0.2 | 0.9 ± 0.3 | 12.88 ± 0.23 | 26.1 ± 0.4 | 28.4 ± 0.9 | -1.5 ± 0.4 | -4.3 ± 0.9 | 5.8e-06 / 7.8e-15 |
| v1 PPO-Lag, eval no shield | 98.5 ± 0.2 | 1.1 ± 0.5 | 0.0 ± 0.0 | 0.4 ± 0.5 | 12.35 ± 0.37 | 24.7 ± 0.7 | 25.4 ± 1.6 | -2.8 ± 0.7 | -7.3 ± 1.6 | 3.8e-17 / 4.1e-28 |
| v1 PPO-Lag, eval shield | 97.9 ± 0.1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.1 ± 0.1 | 12.37 ± 0.40 | 25.8 ± 0.5 | 28.0 ± 1.1 | -1.8 ± 0.5 | -4.7 ± 1.1 | 1e-11 / 5.6e-19 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v2 PPO-Lag(mask shield + budget), eval no shield | 0 | 78.7 | -18.8 | 5 | 118 | 4.24e-29 |
| v2 PPO-Lag(mask shield + budget), eval shield | 0 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 1 | 79.8 | -17.7 | 3 | 109 | 9.02e-29 |
| v2 PPO-Lag(mask shield + budget), eval shield | 1 | 96.7 | -0.8 | 2 | 7 | 0.18 |
| v2 PPO-Lag(mask shield + budget), eval no shield | 2 | 77.2 | -20.3 | 6 | 128 | 6.91e-31 |
| v2 PPO-Lag(mask shield + budget), eval shield | 2 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v2 PPO-Lag(no shield + budget), eval no shield | 0 | 96.8 | -0.7 | 3 | 7 | 0.344 |
| v2 PPO-Lag(no shield + budget), eval shield | 0 | 96.8 | -0.7 | 3 | 7 | 0.344 |
| v2 PPO-Lag(no shield + budget), eval no shield | 1 | 95.8 | -1.7 | 3 | 13 | 0.0213 |
| v2 PPO-Lag(no shield + budget), eval shield | 1 | 95.8 | -1.7 | 3 | 13 | 0.0213 |
| v2 PPO-Lag(no shield + budget), eval no shield | 2 | 96.2 | -1.3 | 2 | 10 | 0.0386 |
| v2 PPO-Lag(no shield + budget), eval shield | 2 | 96.2 | -1.3 | 2 | 10 | 0.0386 |
| v1 PPO-Lag(train shield, post shield), eval shield | 0 | 98.2 | +0.7 | 5 | 1 | 0.219 |
| v1 PPO-Lag(train shield, post shield), eval shield | 1 | 98.0 | +0.5 | 4 | 1 | 0.375 |
| v1 PPO-Lag(train shield, post shield), eval shield | 2 | 98.0 | +0.5 | 4 | 1 | 0.375 |
| v1 PPO-Lag, eval no shield | 0 | 98.3 | +0.8 | 6 | 1 | 0.125 |
| v1 PPO-Lag, eval shield | 0 | 98.0 | +0.5 | 3 | 0 | 0.25 |
| v1 PPO-Lag, eval no shield | 1 | 98.5 | +1.0 | 7 | 1 | 0.0703 |
| v1 PPO-Lag, eval shield | 1 | 97.8 | +0.3 | 4 | 2 | 0.688 |
| v1 PPO-Lag, eval no shield | 2 | 98.7 | +1.2 | 7 | 0 | 0.0156 |
| v1 PPO-Lag, eval shield | 2 | 97.8 | +0.3 | 3 | 1 | 0.625 |
