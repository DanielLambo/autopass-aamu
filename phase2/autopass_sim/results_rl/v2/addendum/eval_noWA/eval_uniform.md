# Report 3 deterministic evaluation (simulated)

### uniform (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.7 | 0.0 | 2.3 | 0.0 | 12.18 [11.81, 12.57] | 27.6 [26.6, 28.6] | 32.6 [31.0, 34.3] | — | — | — |
| rule+shield | 97.7 | 0.0 | 2.3 | 0.0 | 12.18 [11.82, 12.54] | 27.6 [26.5, 28.6] | 32.6 [31.0, 34.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2-A-noWA (w_A hidden), eval no shield | 74.6 ± 4.1 | 24.8 ± 3.6 | 0.0 ± 0.0 | 0.6 ± 0.7 | 10.48 ± 0.31 | 21.1 ± 0.4 | 18.1 ± 0.8 | -6.4 ± 0.4 | -14.5 ± 0.8 | 1e-32 / 1.1e-48 |
| v2-A-noWA (w_A hidden), eval shield | 97.9 ± 0.8 | 0.0 ± 0.0 | 1.2 ± 0.2 | 0.9 ± 0.9 | 13.23 ± 0.23 | 26.7 ± 0.1 | 29.7 ± 0.2 | -0.9 ± 0.1 | -2.9 ± 0.2 | 2e-06 / 1.7e-13 |
| v2-A, eval no shield | 78.7 ± 1.3 | 20.6 ± 2.4 | 0.0 ± 0.0 | 0.7 ± 1.1 | 10.62 ± 0.16 | 21.5 ± 0.4 | 19.0 ± 1.1 | -6.1 ± 0.4 | -13.7 ± 1.1 | 1e-31 / 2.5e-42 |
| v2-A, eval shield | 97.6 ± 1.1 | 0.0 ± 0.0 | 1.1 ± 0.3 | 1.3 ± 1.3 | 13.01 ± 0.57 | 26.5 ± 0.5 | 29.4 ± 0.9 | -1.1 ± 0.5 | -3.3 ± 0.9 | 5e-07 / 5.9e-13 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v2-A-noWA (w_A hidden), eval no shield | 0 | 71.0 | -26.7 | 2 | 162 | 1.16e-45 |
| v2-A-noWA (w_A hidden), eval shield | 0 | 97.0 | -0.7 | 1 | 5 | 0.219 |
| v2-A-noWA (w_A hidden), eval no shield | 1 | 73.8 | -23.8 | 4 | 147 | 1.5e-38 |
| v2-A-noWA (w_A hidden), eval shield | 1 | 98.3 | +0.7 | 4 | 0 | 0.125 |
| v2-A-noWA (w_A hidden), eval no shield | 2 | 79.0 | -18.7 | 4 | 116 | 1.28e-29 |
| v2-A-noWA (w_A hidden), eval shield | 2 | 98.3 | +0.7 | 4 | 0 | 0.125 |
| v2-A, eval no shield | 0 | 78.7 | -19.0 | 4 | 118 | 3.42e-30 |
| v2-A, eval shield | 0 | 98.3 | +0.7 | 4 | 0 | 0.125 |
| v2-A, eval no shield | 1 | 80.0 | -17.7 | 3 | 109 | 9.02e-29 |
| v2-A, eval shield | 1 | 96.3 | -1.3 | 2 | 10 | 0.0386 |
| v2-A, eval no shield | 2 | 77.3 | -20.3 | 5 | 127 | 1.18e-31 |
| v2-A, eval shield | 2 | 98.2 | +0.5 | 4 | 1 | 0.375 |
