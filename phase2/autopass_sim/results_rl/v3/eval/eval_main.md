# Report 3 deterministic evaluation (simulated)

### main (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.5 | 0.0 | 2.5 | 0.0 | 12.18 [11.82, 12.57] | 27.6 [26.6, 28.6] | 32.7 [31.1, 34.3] | — | — | — |
| rule+shield | 97.5 | 0.0 | 2.5 | 0.0 | 12.18 [11.82, 12.55] | 27.6 [26.6, 28.7] | 32.7 [31.1, 34.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 75.4 ± 4.4 | 24.0 ± 4.5 | 0.0 ± 0.0 | 0.6 ± 0.2 | 10.87 ± 0.41 | 21.7 ± 0.8 | 19.4 ± 1.8 | -5.9 ± 0.8 | -13.3 ± 1.8 | 5.7e-28 / 2.8e-43 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 97.8 ± 0.2 | 0.0 ± 0.0 | 1.3 ± 0.2 | 0.8 ± 0.0 | 13.40 ± 0.12 | 26.9 ± 0.2 | 30.1 ± 0.4 | -0.7 ± 0.2 | -2.5 ± 0.4 | 1.1e-06 / 1.5e-13 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 97.8 ± 1.0 | 0.0 ± 0.0 | 1.2 ± 0.2 | 1.1 ± 1.1 | 13.10 ± 0.59 | 26.6 ± 0.6 | 29.5 ± 1.2 | -1.0 ± 0.6 | -3.2 ± 1.2 | 3.4e-07 / 2.9e-14 |
| v1 PPO-Lag(train shield, post shield), eval shield | 98.0 ± 0.3 | 0.0 ± 0.0 | 1.0 ± 0.2 | 1.0 ± 0.4 | 12.93 ± 0.19 | 26.2 ± 0.3 | 28.6 ± 0.8 | -1.4 ± 0.3 | -4.1 ± 0.8 | 1.8e-05 / 2.1e-14 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 0 | 80.5 | -17.0 | 4 | 106 | 9.24e-27 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 0 | 97.8 | +0.3 | 3 | 1 | 0.625 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 1 | 72.5 | -25.0 | 5 | 155 | 1.16e-39 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 1 | 98.0 | +0.5 | 4 | 1 | 0.375 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 2 | 73.3 | -24.2 | 4 | 149 | 3.95e-39 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 2 | 97.7 | +0.2 | 2 | 1 | 1 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 0 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 1 | 96.7 | -0.8 | 2 | 7 | 0.18 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 2 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v1 PPO-Lag(train shield, post shield), eval shield | 0 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v1 PPO-Lag(train shield, post shield), eval shield | 1 | 97.8 | +0.3 | 4 | 2 | 0.688 |
| v1 PPO-Lag(train shield, post shield), eval shield | 2 | 97.8 | +0.3 | 4 | 2 | 0.688 |
