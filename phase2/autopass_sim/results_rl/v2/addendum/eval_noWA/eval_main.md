# Report 3 deterministic evaluation (simulated)

### main (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 97.5 | 0.0 | 2.5 | 0.0 | 12.18 [11.82, 12.57] | 27.6 [26.6, 28.6] | 32.7 [31.1, 34.3] | — | — | — |
| rule+shield | 97.5 | 0.0 | 2.5 | 0.0 | 12.18 [11.82, 12.55] | 27.6 [26.6, 28.7] | 32.7 [31.1, 34.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v2-A-noWA (w_A hidden), eval no shield | 74.3 ± 3.5 | 25.2 ± 3.0 | 0.0 ± 0.0 | 0.6 ± 0.7 | 10.46 ± 0.30 | 21.1 ± 0.4 | 18.0 ± 0.7 | -6.5 ± 0.4 | -14.7 ± 0.7 | 2e-33 / 2.1e-49 |
| v2-A-noWA (w_A hidden), eval shield | 97.9 ± 0.8 | 0.0 ± 0.0 | 1.2 ± 0.1 | 0.9 ± 0.8 | 13.22 ± 0.22 | 26.6 ± 0.1 | 29.5 ± 0.1 | -1.0 ± 0.1 | -3.2 ± 0.1 | 5.1e-07 / 2e-14 |
| v2-A, eval no shield | 78.5 ± 1.3 | 20.9 ± 2.1 | 0.0 ± 0.0 | 0.6 ± 0.9 | 10.60 ± 0.16 | 21.4 ± 0.3 | 18.8 ± 0.9 | -6.2 ± 0.3 | -13.9 ± 0.9 | 6e-32 / 1.4e-44 |
| v2-A, eval shield | 97.8 ± 1.0 | 0.0 ± 0.0 | 1.2 ± 0.2 | 1.1 ± 1.1 | 13.10 ± 0.59 | 26.6 ± 0.6 | 29.5 ± 1.2 | -1.0 ± 0.6 | -3.2 ± 1.2 | 3.4e-07 / 2.9e-14 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v2-A-noWA (w_A hidden), eval no shield | 0 | 71.0 | -26.5 | 3 | 162 | 3.2e-44 |
| v2-A-noWA (w_A hidden), eval shield | 0 | 97.0 | -0.5 | 2 | 5 | 0.453 |
| v2-A-noWA (w_A hidden), eval no shield | 1 | 73.8 | -23.7 | 5 | 147 | 2.29e-37 |
| v2-A-noWA (w_A hidden), eval shield | 1 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v2-A-noWA (w_A hidden), eval no shield | 2 | 78.0 | -19.5 | 5 | 122 | 3.11e-30 |
| v2-A-noWA (w_A hidden), eval shield | 2 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v2-A, eval no shield | 0 | 78.7 | -18.8 | 5 | 118 | 4.24e-29 |
| v2-A, eval shield | 0 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
| v2-A, eval no shield | 1 | 79.7 | -17.8 | 3 | 110 | 4.63e-29 |
| v2-A, eval shield | 1 | 96.7 | -0.8 | 2 | 7 | 0.18 |
| v2-A, eval no shield | 2 | 77.2 | -20.3 | 6 | 128 | 6.91e-31 |
| v2-A, eval shield | 2 | 98.3 | +0.8 | 5 | 0 | 0.0625 |
