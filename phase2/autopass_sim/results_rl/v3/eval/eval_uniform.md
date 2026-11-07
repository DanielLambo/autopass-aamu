# Report 3 deterministic evaluation (simulated)

### uniform (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.7 | 0.0 | 2.3 | 0.0 | 12.18 [11.81, 12.57] | 27.6 [26.6, 28.6] | 32.6 [31.0, 34.3] | — | — | — |
| rule+shield | 97.7 | 0.0 | 2.3 | 0.0 | 12.18 [11.82, 12.54] | 27.6 [26.5, 28.6] | 32.6 [31.0, 34.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 75.9 ± 4.5 | 23.6 ± 4.6 | 0.0 ± 0.0 | 0.5 ± 0.2 | 10.86 ± 0.43 | 21.7 ± 0.9 | 19.3 ± 1.8 | -5.8 ± 0.9 | -13.3 ± 1.8 | 6.5e-27 / 3e-43 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 97.8 ± 0.2 | 0.0 ± 0.0 | 1.3 ± 0.3 | 0.8 ± 0.2 | 13.45 ± 0.14 | 27.0 ± 0.2 | 30.4 ± 0.4 | -0.6 ± 0.2 | -2.2 ± 0.4 | 1.1e-05 / 2.5e-12 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 97.6 ± 1.1 | 0.0 ± 0.0 | 1.1 ± 0.3 | 1.3 ± 1.3 | 13.01 ± 0.57 | 26.5 ± 0.5 | 29.4 ± 0.9 | -1.1 ± 0.5 | -3.3 ± 0.9 | 5e-07 / 5.9e-13 |
| v1 PPO-Lag(train shield, post shield), eval shield | 98.2 ± 0.2 | 0.0 ± 0.0 | 0.9 ± 0.2 | 0.9 ± 0.3 | 12.94 ± 0.14 | 26.2 ± 0.2 | 28.6 ± 0.8 | -1.4 ± 0.2 | -4.0 ± 0.8 | 5e-05 / 6e-14 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 0 | 81.0 | -16.7 | 5 | 105 | 1.98e-25 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 0 | 97.8 | +0.2 | 3 | 2 | 1 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 1 | 72.8 | -24.8 | 4 | 153 | 2.74e-40 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 1 | 98.0 | +0.3 | 3 | 1 | 0.625 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 2 | 73.8 | -23.8 | 4 | 147 | 1.5e-38 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 2 | 97.7 | +0.0 | 2 | 2 | 1 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 0 | 98.3 | +0.7 | 4 | 0 | 0.125 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 1 | 96.3 | -1.3 | 2 | 10 | 0.0386 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 2 | 98.2 | +0.5 | 4 | 1 | 0.375 |
| v1 PPO-Lag(train shield, post shield), eval shield | 0 | 98.3 | +0.7 | 4 | 0 | 0.125 |
| v1 PPO-Lag(train shield, post shield), eval shield | 1 | 98.0 | +0.3 | 3 | 1 | 0.625 |
| v1 PPO-Lag(train shield, post shield), eval shield | 2 | 98.2 | +0.5 | 3 | 0 | 0.25 |
