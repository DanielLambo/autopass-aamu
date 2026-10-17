# Preliminary simulation results (v0, Oct 13 2026)

**Simulated, not experimentally validated.** Sensor noise, sensor height, the scan pattern
and the 1 cm error floor are assumptions until they are fitted on bag data (see README).

## Setup
- 200 generated worlds: 100 `room` (open room with furniture) and 100 `corridor` (2.4 m
  hallway). Each world has 2–3 doors: one clearly passable, the rest within ±3 cm of the
  wheelchair requirement (81.3 cm). 60% steel frames, 40% wood stops.
- All methods run on the same worlds. A* methods: 100 worlds per layout. RRT*: 30 per layout.
- Raw-scan sensor mode (L1 at 0.5 s per scan, one scan per 0.25 m at 0.5 m/s). Commit
  point 1.2 m before the door. z = 2.
- Reproduce: `python -m autopass_sim.experiments --worlds 100 --rrt-worlds 30 --seed0 1000`

## Findings (raw-scan mode)

| | Unsafe missions (committed to a too-narrow door) | Wrong door decisions | Mean mission time |
|---|---|---|---|
| B1 shortest path + naive rule | 14 / 200 (7.0%) | 4.6–6.5% | 21.0–21.6 s |
| P-A*(4)n observability path + naive rule | 4 / 200 (2.0%) | 0.7–2.3% | 24.6–25.0 s |
| B2 shortest path + trusted rule | 0 / 200 | 0% | 26.8 s |
| P-A*(4) observability path + trusted rule | 0 / 200 | 0% | 27.3–27.8 s |

1. **Measure-while-walking is unsafe near the threshold.** B1 committed to a too-narrow
   door in 7% of missions. Every one of those was a false PASS caused by oblique or
   occluded views reading wide.
2. **The trusted decision rule removes those errors.** B2 went from 14 unsafe missions to 0
   (paired McNemar exact p < 0.001), at a cost of about 5.4 s per mission (paired).
3. **The path on its own also helps.** With the same naive rule, the observability-aware
   path cut unsafe missions from 14 to 4 (p = 0.006, 11 vs 1 discordant pairs), for about
   3.4 s more per mission.
4. **Combined, sensing quality goes up at little cost.** At λ = 4:
   - Trusted evidence at the first decision rose from 5.1 to 6.6 /cm² (+29%) in the
     corridor and from 7.7 to 11.6 /cm² (+51%) in the room.
   - Width error at decision dropped from 0.21 to 0.15–0.16 cm.
   - Extra time was +0.5 s (95% CI −0.0 to 1.1) in the corridor and +1.1 s (−0.8 to 2.9)
     in the room.
   - The decision error rate was already 0 with B2, so the gain shows up as sensing
     quality, not as fewer wrong calls.
5. **RRT\* matches A\* on decisions but costs about 8× the CPU** (3.7–4.0 s vs 0.35–0.47 s
   per mission, Python). A* is the better choice at this map scale.
6. **About 14% of decisions stay FLAG for every safe method.** These are doors within about
   ±1 cm of the requirement. At this noise level they cannot be decided, so the robot
   routes to another door and success stays at 99–100%.

## Voxel-map mode (what the Phase 1 bags actually contain)
With the Go2's 5 cm voxel map instead of raw scans (60 worlds per layout):
- FLAG rate rose to 35–52%.
- Mission success fell to 62–85%.
- Width error at decision was about 2.6–3.0 cm.

**Doors within about ±3 cm of a requirement cannot be decided from the voxel map**,
whatever the path. Phase 2 robot experiments should record the raw L1 point cloud.

## Known limitations / next steps
- All parameters marked "assumed" in the README must be fitted on bag data.
- The planner does not use a prior on each door's width. In a few worlds it went to a
  well-observable but tight door before the wide one (e.g. room seed 1014, +29 s). Next
  step: a check-or-detour rule that uses far-range observations as a prior.
- 2D extruded geometry, a simplified edge estimator, and geometric planning only (no Go2
  gait limits yet).
- The number of worlds is modest. Rerun with `--worlds 300` on the TensorBook for tighter
  CIs.
