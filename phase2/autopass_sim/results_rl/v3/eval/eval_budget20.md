# Report 3 deterministic evaluation (simulated)

### budget20 (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 50.5 | 0.0 | 49.5 | 0.0 | 7.82 [7.51, 8.15] | 16.3 [15.7, 16.9] | 17.9 [17.6, 18.2] | — | — | — |
| rule+shield | 50.5 | 0.0 | 49.5 | 0.0 | 7.82 [7.51, 8.13] | 16.3 [15.7, 16.9] | 17.9 [17.6, 18.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 61.6 ± 0.3 | 25.1 ± 0.4 | 13.4 ± 0.1 | 0.0 ± 0.0 | 9.70 ± 0.42 | 19.4 ± 0.8 | 15.3 ± 0.3 | +3.1 ± 0.8 | -2.6 ± 0.3 | 5.6e-19 / 7.2e-33 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 52.6 ± 0.3 | 0.0 ± 0.0 | 47.4 ± 0.3 | 0.0 ± 0.0 | 8.73 ± 0.03 | 17.5 ± 0.1 | 17.4 ± 0.0 | +1.2 ± 0.1 | -0.6 ± 0.0 | 4.6e-07 / 3.3e-06 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 52.8 ± 0.3 | 0.0 ± 0.0 | 47.2 ± 0.3 | 0.0 ± 0.0 | 8.70 ± 0.18 | 17.4 ± 0.3 | 17.3 ± 0.0 | +1.1 ± 0.3 | -0.6 ± 0.0 | 6.2e-05 / 6.7e-07 |
| v1 PPO-Lag(train shield, post shield), eval shield | 51.8 ± 0.9 | 0.0 ± 0.0 | 35.0 ± 2.9 | 13.2 ± 2.1 | 8.75 ± 0.11 | 24.0 ± 0.3 | 17.5 ± 0.1 | +7.7 ± 0.3 | -0.4 ± 0.1 | 1.1e-25 / 0.047 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 0 | 61.8 | +11.3 | 91 | 23 | 9.29e-11 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 0 | 52.7 | +2.2 | 44 | 31 | 0.165 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 1 | 61.2 | +10.7 | 89 | 25 | 1.32e-09 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 1 | 52.3 | +1.8 | 43 | 32 | 0.248 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 2 | 61.7 | +11.2 | 89 | 22 | 9.63e-11 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 2 | 52.8 | +2.3 | 43 | 29 | 0.125 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 0 | 53.2 | +2.7 | 46 | 30 | 0.0846 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 1 | 52.5 | +2.0 | 38 | 26 | 0.169 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 2 | 52.8 | +2.3 | 45 | 31 | 0.135 |
| v1 PPO-Lag(train shield, post shield), eval shield | 0 | 50.8 | +0.3 | 38 | 36 | 0.908 |
| v1 PPO-Lag(train shield, post shield), eval shield | 1 | 52.0 | +1.5 | 39 | 30 | 0.336 |
| v1 PPO-Lag(train shield, post shield), eval shield | 2 | 52.5 | +2.0 | 40 | 28 | 0.182 |
