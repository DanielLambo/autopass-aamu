# AutoPass Phase 2 — Report 2 video supplement (simulated)

All three videos come from the AutoPass simulator, not from the robot. Click a file, then **View raw**, to play it in the browser.

| File | Report figure | What it shows |
|---|---|---|
| [sweep_door.mp4](sweep_door.mp4) | Figure 1 | The Go2 walks an arc 1.6 m from a steel door frame. Past about 15° the frame corner hides the door stop and the opening reads wider; the error is a bias, so more scans from the same spot do not remove it. |
| [replay_room_1077.mp4](replay_room_1077.mp4) | Figure 7 | Furnished room, world 1077. Left: shortest path, deciding while walking, commits to a door 1.8 cm too narrow for a wheelchair. Right: observability-aware A* with the confidence rule goes to the wide door's trusted cone and passes. |
| [replay_corridor_1001.mp4](replay_corridor_1001.mp4) | Section 6 | Corridor, world 1001. The baseline commits to a door 0.15 cm too narrow; the proposed method flags the two doors within 1 cm of the requirement and passes the third, at the cost of a longer route. |

Regenerate: `python -m autopass_sim.render_sweep` and `python -m autopass_sim.render2d --layout room --seed 1077` / `--layout corridor --seed 1001`.
