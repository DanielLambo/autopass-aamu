# v2-A and rule on Report 2's no-repositioning worlds (simulated)

600 worlds (room + corridor, world seeds 1000-1299), wheelchair w_A = 81.3 cm. Mask shield on for v2-A. Time: mean [95% bootstrap CI] over all 600 missions; Δtime paired per world vs the RL-env rule. RL env and Report 2 use the same worlds, sensor model and z-test, but different executors (discrete viewpoints vs planned path with a 1.2 m commit point), so cross-source time differences are indicative only.

Report 2 rerun reproduces results_rev exactly: True.

| source | policy | n | success % | unsafe % | flag / no route % | timeout % | time_s | time_s (successes) | Δtime vs rule, no reposition (vp 0) |
|---|---|---|---|---|---|---|---|---|---|
| RL env | rule, no reposition (vp 0) | 600 | 98.5 | 0.0 | 1.5 | 0.0 | 27.9 [26.9, 28.9] | 27.8 | — |
| RL env | v2-A s0, no reposition (vp 0) | 600 | 98.7 | 0.0 | 1.3 | 0.0 | 25.4 [24.7, 26.2] | 25.0 | -2.5 [-3.3, -1.7] |
| RL env | v2-A s1, no reposition (vp 0) | 600 | 97.3 | 0.0 | 1.0 | 1.7 | 26.3 [25.2, 27.7] | 25.2 | -1.6 [-2.6, -0.3] |
| RL env | v2-A s2, no reposition (vp 0) | 600 | 98.7 | 0.0 | 1.2 | 0.2 | 25.6 [24.8, 26.4] | 25.1 | -2.3 [-3.2, -1.5] |
| RL env | v2-A s0, all 5 viewpoints | 600 | 99.0 | 0.0 | 0.8 | 0.2 | 25.9 [25.0, 26.8] | 25.4 | -2.0 [-2.9, -1.1] |
| RL env | v2-A s1, all 5 viewpoints | 600 | 97.7 | 0.0 | 0.7 | 1.7 | 26.6 [25.4, 28.0] | 25.3 | -1.3 [-2.4, +0.1] |
| RL env | v2-A s2, all 5 viewpoints | 600 | 98.8 | 0.0 | 1.2 | 0.0 | 26.3 [25.3, 27.4] | 25.2 | -1.6 [-2.7, -0.4] |
| Report 2 planner | B2-nr | 600 | 21.5 | 0.0 | 78.5 | 0.0 | 38.5 [37.5, 39.6] | 22.4 | +10.6 [+9.5, +11.7] |
| Report 2 planner | P-A*(4) | 600 | 99.7 | 0.0 | 0.3 | 0.0 | 28.1 [27.1, 29.1] | 28.1 | +0.2 [-0.5, +0.9] |
| Report 2 planner | P-A*(4)-nr | 600 | 98.3 | 0.0 | 1.7 | 0.0 | 29.0 [28.0, 30.1] | 28.8 | +1.1 [+0.5, +1.8] |
