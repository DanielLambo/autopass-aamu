"""AutoPassEnv: Report 2 simulator wrapped as a constrained MDP (Gymnasium API).

High-level decisions only. The robot moves between discrete viewpoints in front of
each door; low-level walking is the geometric A* walk from mission.py (scans every 0.25 m).
Known floor plan, unknown door widths.

step() returns the standard Gymnasium 5-tuple. The CMDP cost is NOT in the reward:
    info["cost"]         1.0 on an unsafe commit (true width < w_A), else 0.0
    info["action_mask"]  bool array, also available as env.action_masks()
SafeAutoPassEnv (below) returns the safety-gymnasium 6-tuple (obs, r, cost, term, trunc, info).

Action index layout (N_MAX doors, K viewpoints per door):
    [0, N*K)           GOTO(door i, viewpoint k)   a = i*K + k
    N*K                DWELL (one 0.5 s scan in place)
    N*K + 1 + i        COMMIT(door i)   drive through door i to the goal, episode ends
    N*K + 1 + N + i    REJECT(door i)   mark door i blocked
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from autopass_sim.world import make_world
from autopass_sim.observability import ObsModel
from autopass_sim.mission import Config, _observe
from autopass_sim.planners import astar, free_with_blocked, path_length, resample
from autopass_sim.sensor import view_geometry, jambs_visible
from autopass_rl import fmm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def distance_field(free, res, goal):
    """Grid distance (m) from every free cell to the goal cell: same 8-neighbour moves,
    unit cost and no-corner-cutting rule as planners.astar. inf where unreachable/blocked."""
    ny, nx = free.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, wts = [], [], []
    for dx, dy, w in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        ys = slice(max(0, -dy), ny - max(0, dy))
        xs = slice(0, nx - dx)
        ys2 = slice(max(0, dy), ny + min(0, dy) if dy < 0 else ny)
        a, b = free[ys, xs], free[ys2, dx:]
        ok = a & b
        if dx and dy:                                  # diagonal: both orthogonal cells must be free
            ok &= free[ys, dx:] & free[ys2, xs]
        rows.append(idx[ys, xs][ok]); cols.append(idx[ys2, dx:][ok]); wts.append(np.full(ok.sum(), w * res))
    g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2).tocsr()
    gx, gy = int(goal[0] / res), int(goal[1] / res)
    if not free[gy, gx]:
        return np.full((ny, nx), np.inf)
    return dijkstra(g, directed=False, indices=gy * nx + gx).reshape(ny, nx)

_ROOT = os.path.join(os.path.dirname(__file__), "..")
N_MAX = 3
# viewpoint offsets (dx, dy) from the door centre, dy measured away from the wall (m)
VIEWPOINTS = np.array([[0.0, 1.2],    # 0 square-on at the commit point
                       [0.0, 2.0],    # 1 square-on, mid
                       [0.0, 3.0],    # 2 square-on, far
                       [-0.9, 1.6],   # 3 oblique left  (psi ~ 10-20 deg, often untrusted)
                       [0.9, 1.6]])   # 4 oblique right
K = len(VIEWPOINTS)
AGENTS = {"wheelchair": 81.3, "walker": 71.1}
OBS_GLOBAL, OBS_DOOR = 7, 14
OBS_BUDGET = 2       # appended last: has_budget, remaining_sweeps / 40 (v1 agents use obs[:-OBS_BUDGET])
BUDGET_DIST_V2 = ((None, 0.4), (30, 0.2), (20, 0.2), (12, 0.2))


class AutoPassEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, layouts=("room", "corridor"), agents="profiles", shield=False,
                 time_cost=0.05, success_reward=10.0, max_steps=80, max_time=300.0,
                 commit_needs_trusted=True, calib=None, rho_calib=None, sensor_mode="raw",
                 scan_budget=None, shield_mode="mask", budget_dist=None, hide_wA=False, viewpoints=None,
                 path_mode="astar"):
        self.layouts = list(layouts)
        self.scan_budget = scan_budget        # max total sweeps per episode (None = unlimited); stress setting
        self._fixed_budget = scan_budget
        # budget_dist: ((budget or None, prob), ...) sampled at each reset (training); overrides scan_budget
        self.budget_dist = tuple(budget_dist) if budget_dist else None
        assert shield_mode in ("post", "mask")
        # "mask": COMMIT(i) is only a valid action when ztest(i) == PASS (preemptive shield).
        # "post": COMMIT is always valid; a non-PASS COMMIT is blocked and costs a DWELL (Report 3 v1).
        self.shield_mode = shield_mode
        # hide_wA (ablation): the explicit w_A feature (obs[4]) is set to 0. The per-door margin features
        # (m = width - w_A, z-score, z-test result) are unchanged, since the decision rule is defined on them.
        self.hide_wA = hide_wA
        # viewpoints: allowed GOTO viewpoint indices (None = all K). (0,) = only the square-on commit point,
        # i.e. no repositioning to a better view (Report 2 "-nr" scenario).
        self.viewpoints = None if viewpoints is None else set(viewpoints)
        # path_mode: "astar" = 8-connected A* + line-of-sight shortcut, route features from a grid Dijkstra field
        # (Report 2/3 default). "fmm" = fast-marching travel-time field per target on the same inflated grid;
        # paths follow its gradient and route features read the same field.
        assert path_mode in ("astar", "fmm")
        self.path_mode = path_mode
        self.agents = agents                  # "profiles" (wheelchair/walker) | "uniform" (66-92 cm)
        self.shield = shield
        self.time_cost, self.success_reward = time_cost, success_reward
        self.max_steps, self.max_time = max_steps, max_time
        self.commit_needs_trusted = commit_needs_trusted
        self.model = ObsModel(calib or os.path.join(_ROOT, "calib.json"),
                              rho_calib=rho_calib or os.path.join(_ROOT, "calib_rho.json"))
        self.cfg = Config(protocol="trusted", sensor_mode=sensor_mode)
        self.n_actions = N_MAX * K + 1 + 2 * N_MAX
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(-10.0, 10.0, (OBS_GLOBAL + N_MAX * OBS_DOOR + OBS_BUDGET,), np.float32)
        self.A_DWELL = N_MAX * K
        self.A_COMMIT = N_MAX * K + 1
        self.A_REJECT = N_MAX * K + 1 + N_MAX

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # world and sensor streams both derive from the Gymnasium np_random -> seeded runs repeat
        wseed, sseed = (int(x) for x in self.np_random.integers(0, 2**31 - 1, 2))
        if "world_seed" in options:           # rebuild a Report 2 world: make_world(default_rng(seed), layout)
            wseed = int(options["world_seed"]); sseed = wseed * 101 + 7
        layout = options.get("layout") or self.layouts[int(self.np_random.integers(len(self.layouts)))]
        if "w_A" in options:
            self.w_A = float(options["w_A"])
        elif self.agents == "uniform":
            self.w_A = float(self.np_random.uniform(66.0, 92.0))
        else:
            self.w_A = float(list(AGENTS.values())[int(self.np_random.integers(len(AGENTS)))])
        if self.budget_dist:                  # drawn last, so worlds/agents match the no-budget envs per seed
            vals = [b for b, _ in self.budget_dist]
            self.scan_budget = vals[int(self.np_random.choice(len(vals), p=[p for _, p in self.budget_dist]))]
        else:
            self.scan_budget = self._fixed_budget
        self.cfg.agent_w = self.w_A
        self.world = make_world(np.random.default_rng(wseed), layout, agent_w=self.w_A / 100)
        self.srng = np.random.default_rng(sseed)
        self.doors = sorted(self.world.doors, key=lambda d: d.xc)[:N_MAX]
        self.pos = np.asarray(self.world.start, float)
        self.memory = {d.id: [] for d in self.world.doors}
        self.rejected = set()
        self.t, self.sim_time, self.path_len, self.scans = 0, 0.0, 0.0, 0
        self.shield_blocks = 0
        self.free_near = free_with_blocked(self.world, {d.id for d in self.world.doors})  # start side only
        self.vp = {}                      # (i, k) -> reachable viewpoint position or None
        for i, d in enumerate(self.doors):
            for k in range(K):
                self.vp[i, k] = self._snap(np.array([d.xc + VIEWPOINTS[k, 0], d.yw - VIEWPOINTS[k, 1]]))
        # one goal-rooted distance field per door (other doors blocked); None once the door is rejected
        self.field = {}
        self.fc = fmm.FieldCache(self.world.res) if self.path_mode == "fmm" else None
        for i, d in enumerate(self.doors):
            others = {o.id for o in self.world.doors if o.id != d.id}
            fr = free_with_blocked(self.world, others)
            if self.path_mode == "fmm":
                self.field[i] = (self.fc.get(("goal", i), fr, self.world.goal)[0], fr)
                continue
            gc = (np.floor(self.world.goal / self.world.res) + 0.5) * self.world.res
            self.field[i] = (distance_field(fr, self.world.res, self.world.goal) +
                             float(np.linalg.norm(self.world.goal - gc)), fr)
        self.goto_cache = {}              # (start x, y rounded to 1 cm, door i, viewpoint k) -> path
        self.route = {i: self._route_len(i) for i in range(len(self.doors))}
        _observe(self.world, self.pos, self.srng, self.cfg, self.model, self.memory, None)
        self.scans += 1
        return self._obs(), self._info(outcome=None, cost=0.0)

    def _snap(self, p):
        """Nearest free start-side cell within 0.3 m of p, or None."""
        best, bd = None, np.inf
        for dx in np.arange(-0.3, 0.31, 0.1):
            for dy in np.arange(-0.3, 0.31, 0.1):
                q = p + [dx, dy]
                if q[1] < self.world.wall_y - 0.3 and self._free(q) and np.hypot(dx, dy) < bd:
                    best, bd = q, np.hypot(dx, dy)
        return best

    def _free(self, p):
        i, j = int(p[0] / self.world.res), int(p[1] / self.world.res)
        h, w = self.free_near.shape
        return 0 <= i < w and 0 <= j < h and bool(self.free_near[j, i])

    def _route_len(self, i):
        """Known-map route length from the current position through door i to the goal, read from
        door i's distance field: field at the robot's cell + distance to that cell's centre.
        Grid-path length (no shortcutting), so slightly longer than the A* route; the exact A*
        path is only computed when a COMMIT executes. Rejected doors keep their last value."""
        if self.field.get(i) is None:
            return self.route[i]
        D = self.field[i][0]
        res = self.world.res
        if self.path_mode == "fmm":
            v = fmm.value_at(D, self.pos, res)
            return (None, v) if np.isfinite(v) else (None, np.inf)
        cx, cy = int(self.pos[0] / res), int(self.pos[1] / res)
        if not (0 <= cy < D.shape[0] and 0 <= cx < D.shape[1]) or not np.isfinite(D[cy, cx]):
            return (None, np.inf)
        centre = np.array([(cx + 0.5) * res, (cy + 0.5) * res])
        return (None, float(D[cy, cx] + np.linalg.norm(self.pos - centre)))

    def _commit_path(self, i):
        """Exact A* path through door i (other doors blocked), for the executed COMMIT."""
        fr = self.field[i][1] if self.field.get(i) is not None else \
            free_with_blocked(self.world, {d.id for d in self.world.doors if d.id != self.doors[i].id})
        if self.path_mode == "fmm":
            p = self.fc.path(("goal", i), fr, self.pos, self.world.goal)
            return (p, path_length(p)) if p is not None else self.route[i]
        p = astar(self.world, self.pos, self.world.goal, np.ones_like(self.world.free, float), free=fr)
        return (p, path_length(p)) if p is not None else self.route[i]

    # ------------------------------------------------------------------ belief
    def _fused(self, d):
        """Inverse-variance fusion over trusted sweeps: (m, sigma_m, b, n). Same rule as Report 2."""
        use = [o for o in self.memory[d.id] if o.trusted]
        if not use:
            return 0.0, np.inf, 0.0, 0
        q = np.array([1 / o.sig ** 2 for o in use])
        m = float((q * [o.m for o in use]).sum() / q.sum())
        b = float((q * [o.b for o in use]).sum() / q.sum())
        return m, float(1 / np.sqrt(q.sum())), b, len(use)

    def ztest(self, i):
        m, s, b, n = self._fused(self.doors[i])
        if n == 0:
            return "FLAG"
        if m - b >= self.cfg.z * s:
            return "PASS"
        if m + b <= -self.cfg.z * s:
            return "FAIL"
        return "FLAG"

    # ------------------------------------------------------------------ mask
    def action_masks(self):
        mask = np.zeros(self.n_actions, bool)
        can_scan = not self.budget_spent()
        mask[self.A_DWELL] = can_scan               # budget spent -> only COMMIT / REJECT remain
        for i, d in enumerate(self.doors):
            if d.id in self.rejected:
                continue
            for k in range(K):
                p = self.vp[i, k]
                mask[i * K + k] = can_scan and p is not None and np.linalg.norm(p - self.pos) > 0.05 \
                    and (self.viewpoints is None or k in self.viewpoints)
            n_trusted = self._fused(d)[3]
            ok = np.isfinite(self.route[i][1]) and (n_trusted > 0 or not self.commit_needs_trusted)
            if self.shield and self.shield_mode == "mask":
                ok = ok and self.ztest(i) == "PASS"
            mask[self.A_COMMIT + i] = ok
            mask[self.A_REJECT + i] = True
        return mask

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = int(action)
        mask = self.action_masks()
        if not mask[a]:
            a = self.A_DWELL                       # invalid -> DWELL (for frameworks without masking)
        self.t += 1
        t0 = self.sim_time
        cost, terminated, outcome = 0.0, False, None
        reward = 0.0

        if a < self.A_DWELL:                        # GOTO
            i, k = divmod(a, K)
            key = (round(float(self.pos[0]), 2), round(float(self.pos[1]), 2), i, k)
            if self.path_mode == "fmm":
                self.goto_cache[key] = None if self.vp[i, k] is None else \
                    self.fc.path(("vp", i, k), self.free_near, self.pos, self.vp[i, k])
            elif key not in self.goto_cache:
                self.goto_cache[key] = astar(self.world, self.pos, self.vp[i, k],
                                             np.ones_like(self.world.free, float), free=self.free_near)
            path = self.goto_cache[key]
            if path is not None:
                path = path.copy()
                path[0] = self.pos                  # start from the exact position (cached start is within 1 cm)
            if path is not None and len(path) > 1:
                pts, s = resample(path, self.cfg.step)
                for p in pts[1:]:
                    if not self.budget_spent():         # walk continues, sweeps stop at the budget
                        _observe(self.world, p, self.srng, self.cfg, self.model, self.memory, None)
                        self.scans += 1
                self.path_len += float(s[-1])
                self.sim_time += float(s[-1]) / self.cfg.speed
                self.pos = pts[-1]
            else:
                self._dwell()
        elif a == self.A_DWELL:
            self._dwell()
        elif a < self.A_REJECT:                     # COMMIT
            i = a - self.A_COMMIT
            if self.shield and self.shield_mode == "post" and self.ztest(i) != "PASS":
                self.shield_blocks += 1
                self._dwell()                       # shield: COMMIT only executes on a z-test PASS
            else:
                path, L = self._commit_path(i)
                self.path_len += L
                self.sim_time += L / self.cfg.speed
                terminated = True
                safe = self.doors[i].w * 100 >= self.w_A
                outcome = "success" if safe else "unsafe"
                cost = 0.0 if safe else 1.0
                reward += self.success_reward       # reward does not see safety; the cost does
                self.committed = i
        else:                                       # REJECT
            i = a - self.A_REJECT
            self.rejected.add(self.doors[i].id)
            self.field[i] = None                    # field no longer used; route[i] keeps its last value
            if all(d.id in self.rejected for d in self.doors):
                terminated, outcome = True, "flag"   # all doors blocked: FLAG, logged, robot stops

        if not terminated and a < self.A_COMMIT:
            self.route = {i: self._route_len(i) for i in range(len(self.doors))}
        reward -= self.time_cost * (self.sim_time - t0)
        truncated = not terminated and (self.t >= self.max_steps or self.sim_time >= self.max_time)
        if truncated:
            outcome = "timeout"
        return self._obs(), float(reward), terminated, truncated, self._info(outcome, cost)

    def budget_spent(self):
        return self.scan_budget is not None and self.scans >= self.scan_budget

    def _dwell(self):
        if not self.budget_spent():                 # e.g. a shield-blocked COMMIT after the budget: time only
            _observe(self.world, self.pos, self.srng, self.cfg, self.model, self.memory, None)
            self.scans += 1
        self.sim_time += 0.5

    # ------------------------------------------------------------------ observation
    def _obs(self):
        W, H = self.world.W, self.world.H
        g = self.world.goal - self.pos
        o = [self.pos[0] / W, self.pos[1] / H, g[0] / 20, g[1] / 20,
             0.0 if self.hide_wA else (self.w_A - 76.0) / 10, self.t / self.max_steps, self.sim_time / self.max_time]
        for i in range(N_MAX):
            if i >= len(self.doors):
                o += [0.0] * OBS_DOOR
                continue
            d = self.doors[i]
            dist, psi = view_geometry(d, self.pos)
            m, s, b, n = self._fused(d)
            trusted_now = float(self.model.trusted(dist, psi) and jambs_visible(self.world, d, self.pos))
            z = np.clip((abs(m) - b) / s, -5, 5) * np.sign(m) if n else 0.0
            L = self.route[i][1]
            o += [1.0, float(d.id in self.rejected), (d.xc - self.pos[0]) / 10, (d.yw - self.pos[1]) / 10,
                  dist / 5, psi / 90, trusted_now, n / 20,
                  m / 10 if n else 0.0, min(s, 5.0) / 5 if n else 1.0, b, z / 5,
                  min(L, 40.0) / 40 if np.isfinite(L) else 1.0,
                  {"PASS": 1.0, "FAIL": -1.0, "FLAG": 0.0}[self.ztest(i)]]
        if self.scan_budget is None:
            o += [0.0, 1.0]
        else:
            o += [1.0, max(self.scan_budget - self.scans, 0) / 40]
        return np.clip(np.asarray(o, np.float32), -10.0, 10.0)

    def _info(self, outcome, cost):
        return dict(cost=float(cost), action_mask=self.action_masks(), outcome=outcome, w_A=self.w_A,
                    path_len=self.path_len, sim_time=self.sim_time, scans=self.scans,
                    shield_blocks=self.shield_blocks, n_doors=len(self.doors))


class SafeAutoPassEnv(AutoPassEnv):
    """safety-gymnasium style: step -> (obs, reward, cost, terminated, truncated, info)."""

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return obs, r, info["cost"], term, trunc, info
