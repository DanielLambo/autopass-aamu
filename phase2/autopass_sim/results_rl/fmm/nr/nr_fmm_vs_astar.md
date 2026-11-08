# No-repositioning worlds: FMM vs A* in the RL env (simulated)

Report 2's 600 worlds (room + corridor, world seeds 1000-1299), w_A = 81.3 cm. Counts out of 600. A* = results_rl/v2/addendum/nr_episodes.csv; FMM = nr/nr_episodes.csv. Report 2 planner rows are unchanged (they do not use the RL env) and are in nr/nr_table.md.

| policy | worlds | success astar → fmm | unsafe astar → fmm | flag / no route astar → fmm | timeout astar → fmm | time s astar → fmm | same outcome % |
|---|---|---|---|---|---|---|---|
| rule, no reposition (vp 0) | 600 | 591 → 591 | 0 → 0 | 9 → 9 | 0 → 0 | 27.97 → 27.89 | 100.0 |
| v2-A s0, no reposition (vp 0) | 600 | 592 → 592 | 0 → 0 | 8 → 8 | 0 → 0 | 25.44 → 25.42 | 100.0 |
| v2-A s1, no reposition (vp 0) | 600 | 584 → 584 | 0 → 0 | 6 → 6 | 10 → 10 | 26.49 → 26.32 | 100.0 |
| v2-A s2, no reposition (vp 0) | 600 | 591 → 592 | 0 → 0 | 8 → 7 | 1 → 1 | 25.76 → 25.55 | 99.8 |
| v2-A s0, all 5 viewpoints | 600 | 594 → 594 | 0 → 0 | 5 → 5 | 1 → 1 | 25.92 → 25.85 | 100.0 |
| v2-A s1, all 5 viewpoints | 600 | 585 → 586 | 0 → 0 | 4 → 4 | 11 → 10 | 26.63 → 26.56 | 99.8 |
| v2-A s2, all 5 viewpoints | 600 | 592 → 593 | 0 → 0 | 8 → 7 | 0 → 0 | 26.46 → 26.27 | 99.8 |
