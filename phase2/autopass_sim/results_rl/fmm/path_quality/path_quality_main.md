# A* vs FMM path quality, main setting (simulated)

Same motion queries: every GOTO and COMMIT executed in the A*-mode episodes of the rule and v2-A seeds 0-2 (mask shield), episode seeds 5000-5599. Each query is planned on the same inflated grid three ways. astar = what the env executes in astar mode (8-connected A* + line-of-sight shortcut); astar_raw = the 8-connected cell path before shortcutting. Turns: heading changes > 20° between consecutive 0.25 m segments. Door offset: where a COMMIT path crosses the wall face, distance from the door centre.

| metric | all | goto | commit |
|---|---|---|---|
| n | 6759 | 4414 | 2345 |
| mean length astar_raw (m) | 4.753 | 3.658 | 6.816 |
| mean length astar (m) | 4.57 | 3.562 | 6.467 |
| mean length fmm (m) | 4.572 | 3.569 | 6.461 |
| astar_raw over fmm: mean % | 3.74 | 2.9 | 5.34 |
| astar_raw over fmm: total % | 3.96 | 2.48 | 5.49 |
| astar over fmm: mean % | -0.23 | -0.38 | 0.06 |
| astar over fmm: total % | -0.06 | -0.21 | 0.1 |
| turns>20° per path astar_raw | 2.24 | 1.63 | 3.39 |
| turns>20° per path astar | 0.52 | 0.34 | 0.85 |
| turns>20° per path fmm | 0.55 | 0.36 | 0.91 |
| door offset>0.15 m astar_raw (count) |  |  | 341 |
| mean |door offset| astar_raw (m) |  |  | 0.096 |
| door offset>0.15 m astar (count) |  |  | 1303 |
| mean |door offset| astar (m) |  |  | 0.152 |
| door offset>0.15 m fmm (count) |  |  | 857 |
| mean |door offset| fmm (m) |  |  | 0.13 |
