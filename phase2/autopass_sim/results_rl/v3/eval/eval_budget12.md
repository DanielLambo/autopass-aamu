# Report 3 deterministic evaluation (simulated)

### budget12 (600 episodes per policy, seeds 5000-5599, simulated)

| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |
|---|---|---|---|---|---|---|---|---|---|---|
| rule | 30.0 | 0.0 | 70.0 | 0.0 | 6.03 [5.77, 6.31] | 12.2 [11.7, 12.7] | 11.7 [11.6, 11.8] | — | — | — |
| rule+shield | 30.0 | 0.0 | 70.0 | 0.0 | 6.03 [5.77, 6.30] | 12.2 [11.7, 12.7] | 11.7 [11.5, 11.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 1 / 1 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 42.6 ± 0.5 | 19.2 ± 0.3 | 38.2 ± 0.3 | 0.0 ± 0.0 | 7.84 ± 0.06 | 15.7 ± 0.1 | 11.1 ± 0.1 | +3.5 ± 0.1 | -0.6 ± 0.1 | 1.5e-26 / 1e-08 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 31.9 ± 0.7 | 0.0 ± 0.0 | 68.1 ± 0.7 | 0.0 ± 0.0 | 6.14 ± 0.05 | 12.3 ± 0.1 | 11.6 ± 0.0 | +0.1 ± 0.1 | -0.1 ± 0.0 | 0.85 / 0.0056 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 32.4 ± 0.2 | 0.0 ± 0.0 | 67.6 ± 0.2 | 0.0 ± 0.0 | 6.16 ± 0.06 | 12.3 ± 0.1 | 11.6 ± 0.0 | +0.2 ± 0.1 | -0.1 ± 0.0 | 0.82 / 0.0056 |
| v1 PPO-Lag(train shield, post shield), eval shield | 30.4 ± 1.7 | 0.0 ± 0.0 | 52.8 ± 3.8 | 16.8 ± 2.7 | 6.19 ± 0.06 | 19.8 ± 0.7 | 11.6 ± 0.1 | +7.7 ± 0.7 | -0.1 ± 0.1 | 2.7e-23 / 0.81 |

Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.
Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, c = only rule succeeds):

| policy | seed | success % | Δ pts vs rule | b | c | p |
|---|---|---|---|---|---|---|
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 0 | 42.3 | +12.3 | 84 | 10 | 1.03e-15 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 0 | 31.8 | +1.8 | 27 | 16 | 0.126 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 1 | 42.2 | +12.2 | 83 | 10 | 1.85e-15 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 1 | 31.3 | +1.3 | 24 | 16 | 0.268 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval no shield | 2 | 43.2 | +13.2 | 86 | 7 | 2.08e-18 |
| v3 PPO-Lag(mask shield + budget + dwell cap 20), eval shield | 2 | 32.7 | +2.7 | 28 | 12 | 0.0166 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 0 | 32.7 | +2.7 | 28 | 12 | 0.0166 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 1 | 32.3 | +2.3 | 24 | 10 | 0.0243 |
| v2 PPO-Lag(mask shield + budget, no cap), eval shield | 2 | 32.3 | +2.3 | 28 | 14 | 0.0436 |
| v1 PPO-Lag(train shield, post shield), eval shield | 0 | 28.5 | -1.5 | 17 | 26 | 0.222 |
| v1 PPO-Lag(train shield, post shield), eval shield | 1 | 31.7 | +1.7 | 26 | 16 | 0.164 |
| v1 PPO-Lag(train shield, post shield), eval shield | 2 | 31.2 | +1.2 | 23 | 16 | 0.337 |
