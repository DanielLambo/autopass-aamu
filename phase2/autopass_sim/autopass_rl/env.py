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


class AutoPassEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, layouts=("room", "corridor"), agents="profiles", shield=False,
                 time_cost=0.05, success_reward=10.0, max_steps=80, max_time=300.0,
                 commit_needs_trusted=True, calib=None, rho_calib=None, sensor_mode="raw"):
        self.layouts = list(layouts)
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
        self.observation_space = spaces.Box(-10.0, 10.0, (OBS_GLOBAL + N_MAX * OBS_DOOR,), np.float32)
        self.A_DWELL = N_MAX * K
        self.A_COMMIT = N_MAX * K + 1
        self.A_REJECT = N_MAX * K + 1 + N_MAX

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # world and sensor streams both derive from the Gymnasium np_random -> seeded runs repeat
        wseed, sseed = (int(x) for x in self.np_random.integers(0, 2**31 - 1, 2))
        layout = options.get("layout") or self.layouts[int(self.np_random.integers(len(self.layouts)))]
        if "w_A" in options:
            self.w_A = float(options["w_A"])
        elif self.agents == "uniform":
            self.w_A = float(self.np_random.uniform(66.0, 92.0))
        else:
            self.w_A = float(list(AGENTS.values())[int(self.np_random.integers(len(AGENTS)))])
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
        """Known-map path length from the current position through door i to the goal."""
        others = {d.id for d in self.world.doors if d.id != self.doors[i].id}
        p = astar(self.world, self.pos, self.world.goal, np.ones_like(self.world.free, float),
                  free=free_with_blocked(self.world, others))
        return (p, path_length(p)) if p is not None else (None, np.inf)

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
        mask[self.A_DWELL] = True
        for i, d in enumerate(self.doors):
            if d.id in self.rejected:
                continue
            for k in range(K):
                p = self.vp[i, k]
                mask[i * K + k] = p is not None and np.linalg.norm(p - self.pos) > 0.05
            n_trusted = self._fused(d)[3]
            ok = np.isfinite(self.route[i][1]) and (n_trusted > 0 or not self.commit_needs_trusted)
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
            path = astar(self.world, self.pos, self.vp[i, k], np.ones_like(self.world.free, float),
                         free=self.free_near)
            if path is not None and len(path) > 1:
                pts, s = resample(path, self.cfg.step)
                for p in pts[1:]:
                    _observe(self.world, p, self.srng, self.cfg, self.model, self.memory, None)
                self.scans += len(pts) - 1
                self.path_len += float(s[-1])
                self.sim_time += float(s[-1]) / self.cfg.speed
                self.pos = pts[-1]
            else:
                self._dwell()
        elif a == self.A_DWELL:
            self._dwell()
        elif a < self.A_REJECT:                     # COMMIT
            i = a - self.A_COMMIT
            if self.shield and self.ztest(i) != "PASS":
                self.shield_blocks += 1
                self._dwell()                       # shield: COMMIT only executes on a z-test PASS
            else:
                path, L = self.route[i]
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
            if all(d.id in self.rejected for d in self.doors):
                terminated, outcome = True, "flag"   # all doors blocked: FLAG, logged, robot stops

        if not terminated and a < self.A_COMMIT:
            self.route = {i: self._route_len(i) for i in range(len(self.doors))}
        reward -= self.time_cost * (self.sim_time - t0)
        truncated = not terminated and (self.t >= self.max_steps or self.sim_time >= self.max_time)
        if truncated:
            outcome = "timeout"
        return self._obs(), float(reward), terminated, truncated, self._info(outcome, cost)

    def _dwell(self):
        _observe(self.world, self.pos, self.srng, self.cfg, self.model, self.memory, None)
        self.scans += 1
        self.sim_time += 0.5

    # ------------------------------------------------------------------ observation
    def _obs(self):
        W, H = self.world.W, self.world.H
        g = self.world.goal - self.pos
        o = [self.pos[0] / W, self.pos[1] / H, g[0] / 20, g[1] / 20,
             (self.w_A - 76.0) / 10, self.t / self.max_steps, self.sim_time / self.max_time]
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
