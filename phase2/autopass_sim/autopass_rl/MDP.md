# Report 3, Section 3: Problem formulation (constrained MDP)

Everything here is simulation only. The environment is the Report 2 simulator
(`autopass_sim`, commit 07061aa) with the same L1 ray-cast sensor, density-aware
noise model and trusted-view rule. Code: `autopass_rl/env.py`.

## 3.1 Assumptions

- The floor plan is known: walls, furniture, door positions and frame types.
  Door clear widths are unknown and must be measured.
- The robot is a 0.20 m disc moving on the inflated occupancy grid (as in Report 2).
  No gait dynamics. Each motion action is executed by A* plus a straight-line walk,
  with one 0.5 s LiDAR sweep every 0.25 m.
- One agent profile per episode, given as the required clear width w_A.

## 3.2 Agent profile

w_A is an input to the policy, not a training constant. Default sampling is
wheelchair (81.3 cm, ADA §404.2.3) or walker (71.1 cm), 50/50. `agents="uniform"`
samples w_A ~ U(66, 92) cm to test generalization to profiles never seen in training.
Door widths are generated around w_A as in Report 2: one door clearly passable
(w_A + 7 to 13 cm), the rest within ±3 cm of w_A.

## 3.3 State (observation vector, 49 values)

Global (7): robot position (x/W, y/H), goal offset (/20 m), (w_A − 76)/10,
step fraction t/T_max, elapsed simulated time / 300 s.

Per door, padded to N_max = 3 (14 each):

| Feature | Meaning |
|---|---|
| exists, rejected | padding flag; door already rejected |
| Δx, Δy (/10 m) | door centre relative to robot |
| d/5, ψ/90 | current distance and near-jamb sightline angle |
| trusted_now | current view passes ψ ≤ 12.5°, d ≤ 4 m, both jambs visible |
| n/20 | number of trusted sweeps fused so far |
| m̂/10, σ_m/5, b | fused margin (cm), its std, fused bias bound (cm) |
| z | (|m̂| − b)/σ_m with the sign of m̂, clipped to ±5 |
| L/40 | known-map route length robot → door → goal |
| z-test | +1 PASS, −1 FAIL, 0 undecided |

m̂, σ_m and b are computed exactly as in Report 2: inverse-variance fusion over trusted
sweeps, q = 1/(σ_scan² + σ_floor² + B²). The belief is the per-door (m̂, σ_m) pair,
so the policy never sees the true width.

## 3.4 Actions (discrete, 22)

| Index | Action | Effect |
|---|---|---|
| 0–14 | GOTO(i, k) | walk to viewpoint k of door i, scanning on the way |
| 15 | DWELL | one sweep in place (0.5 s) |
| 16–18 | COMMIT(i) | drive through door i to the goal; episode ends |
| 19–21 | REJECT(i) | mark door i blocked (FLAG, logged) |

Viewpoints, offset from the door centre (lateral, distance back from the wall):
k0 (0, 1.2 m) is the Report 2 commit point, k1 (0, 2.0), k2 (0, 3.0), k3 (−0.9, 1.6),
k4 (+0.9, 1.6). The oblique viewpoints are often untrusted, so the policy has to learn
when they are worth it. Viewpoints inside furniture snap to the nearest free cell
within 0.3 m, otherwise they are masked.

Action mask: GOTO is invalid for rejected doors, unreachable viewpoints and the current
position. COMMIT(i) is invalid if the door is rejected, has no route to the goal, or has
zero trusted sweeps. REJECT(i) is invalid once rejected. DWELL is always valid. A masked
action sent anyway is executed as DWELL, which keeps the env usable by trainers without
masking.

## 3.5 Reward

r_t = −c_t · Δt_sim + R_goal · 1[COMMIT], with c_t = 0.05 per simulated second
(walking 1 m = 2 s, one dwell sweep = 0.5 s) and R_goal = 10.

The reward does not depend on whether the commit was safe. Safety enters only through
the cost below. If the reward also penalized unsafe commits, the safety trade-off would
be hidden inside a hand-tuned penalty weight, which is exactly what the CMDP is meant
to avoid.

## 3.6 Constraint cost and budget

c_t = 1 if the action is COMMIT(i) and the true width w_i < w_A, else 0.
Constraint: J_c(π) = E[Σ c_t] ≤ d, the expected number of unsafe commits per episode.
Default d = 0.01.

This is an explicit cost constraint (CMDP). PPO-Lagrangian bounds the expected cost
only. It is not a per-episode guarantee, and early in training the policy violates it
(see 3.8).

## 3.7 Shield (option)

With `shield=True`, COMMIT(i) only executes if the Report 2 z-test returns PASS
(m̂ − b ≥ z·σ_m, z = 2). Otherwise the env executes one DWELL instead and counts a
shield block. We report every result shielded and unshielded. Shielded training answers
"can RL be more efficient than Report 2 at the same safety". Unshielded training answers
"does the Lagrangian alone learn to be safe".

## 3.8 Episode end

- Terminated: any executed COMMIT (outcome success or unsafe), or every door rejected
  (outcome flag: robot stops and logs, same as Report 2's FLAG).
- Truncated: 80 decisions or 300 s of simulated time (outcome timeout).

## 3.9 Metrics (matched to Report 2)

Success, unsafe commits and flags (counts out of N missions); path length (m); simulated
time (s); sweeps used; shield blocks. Since Report 2 already reaches 0/600 unsafe, the
claim to test is efficiency at matched safety: shorter paths, fewer sweeps, and higher
success in the no-repositioning setting. More safety is not the claim.

## 3.10 Baselines

- In-env rule policy (`autopass_rl/policies.py::RulePolicy`): go to k0, dwell until the
  z-test decides (max 20 sweeps), PASS → commit, else reject. Same decision rule as B2.
- Report 2 planners, run unchanged via `autopass_sim.experiments`: B1, B2, P-A*(4), plus
  the -nr (no repositioning) variants.

Sanity check (60 episodes each, seeds 5000–5059, 40-step cap; the cap is now 80):

| Policy | Success | Unsafe | Flag | Timeout | Path (m) | Sweeps |
|---|---|---|---|---|---|---|
| random (masked) | 31 | 18 | 11 | 0 | 28.5 | 88.8 |
| random + shield | 35 | 0 | 25 | 0 | 32.4 | 116.7 |
| rule | 58 | 0 | 1 | 1 | 11.7 | 31.4 |
| rule + shield | 58 | 0 | 1 | 1 | 11.7 | 31.4 |

The random policy shows the cost signal matters: 18/60 unsafe without the shield.
The rule policy reproduces Report 2's 0 unsafe.

## 3.11 Learner

OmniSafe's on-policy algorithms do not support discrete actions (support is an open PR,
PKU-Alignment/omnisafe#286) and have no action masking. We use a single-file masked
PPO-Lagrangian (`autopass_rl/ppo_lag.py`) with separate reward and cost critics:
A = (Â_r − λÂ_c)/(1 + λ), λ ← max(0, λ + η_λ(Ĵ_c − d)).
`SafeAutoPassEnv` returns the safety-gymnasium 6-tuple, so the env can move to OmniSafe
later if #286 lands or if we switch to a continuous waypoint action.

## 3.12 Changes in v2 (reported method)

- Mask shield (`shield_mode="mask"`): COMMIT(i) is removed from the action mask unless the z-test
  returns PASS, instead of being replaced by a DWELL after the fact. With it, an unsafe commit is
  impossible by construction, so the cost and λ stay at 0 during training.
- Scan budget in the state: two features appended last (budget present, remaining sweeps / 40), 51
  values in total. Training samples the budget from {none 0.4, 30, 20, 12 sweeps 0.2 each}. v1 agents
  read the first 49 values.
- Evaluation settings: `main`, `uniform` (unseen w_A), `budget20`, `budget12`, on seeds 5000-5599.
