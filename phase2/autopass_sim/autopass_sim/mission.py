"""Mission executor: plan, drive, observe, decide at the commit point, replan on FAIL/FLAG.

Every planner is run under the same protocol, so differences in outcome come from the
path (where the robot looked from), not from different decision logic.

Decision protocols (margin m = measured width - agent requirement, in cm)
  naive    fuse every observation within d_max, no bias allowance:
             PASS if m >= z*sigma, FAIL if m <= -z*sigma, else FLAG
  trusted  fuse only trusted views (psi <= psi_safe), bias allowance b:
             PASS if m - b >= z*sigma, FAIL if m + b <= -z*sigma, else FLAG
On FLAG the robot stops at the commit point and keeps scanning (up to `dwell_max` scans).
Still FLAG -> door left unresolved (treated as blocked, robot replans).
"""
from dataclasses import dataclass, field
import numpy as np
from .sensor import scan, estimate_width, view_geometry, VoxelMap, jambs_visible, edge_density
from .planners import astar, rrt_star, free_with_blocked, resample, path_length
from .observability import cost_field


@dataclass
class Config:
    planner: str = "astar"          # astar | rrt
    lam: float = 0.0
    protocol: str = "trusted"       # trusted | naive
    sensor_mode: str = "raw"        # raw | voxelmap
    agent_w: float = 81.3           # cm
    z: float = 2.0
    d_commit: float = 1.2           # m: decide before getting closer than this
    dwell_max: int = 20             # scans (0.5 s each)
    step: float = 0.25              # m between scans (0.5 m/s, 0.5 s per scan)
    speed: float = 0.5              # m/s
    max_replans: int = 4
    rrt_iters: int = 2500
    reposition: bool = True        # trusted protocol: move to a trusted viewpoint when undecided
    rho_source: str = "measured"    # density model: per-sweep measured rho, or 'geometry' = rho_hat(d, psi)
    tracer: object = field(default=None, repr=False)   # optional Tracer for rendering


@dataclass
class Obs:
    m: float      # margin estimate, cm (calibrated)
    sig: float    # cm
    d: float
    psi: float
    trusted: bool
    b: float = 0.5   # per-sweep bias bound, cm (trusted protocol)


class Tracer:
    """Records every scan (robot pose, door hits, running estimate) for the renderers."""

    def __init__(self, keep_pts=400):
        self.frames, self.events, self.keep = [], [], keep_pts

    def scan(self, world, pos, P, memory, cfg, model):
        near = np.zeros(len(P), bool)
        for d in world.doors:
            near |= (np.abs(P[:, 0] - d.xc) < d.hf + 0.6) & (P[:, 1] > d.yw - 0.05) & (P[:, 1] < d.yw + d.D + 0.05) \
                    & (P[:, 2] > 0.3) & (P[:, 2] < 1.9)
        Q = P[near]
        if len(Q) > self.keep:
            Q = Q[np.random.default_rng(len(self.frames)).choice(len(Q), self.keep, replace=False)]
        est = {}
        for d in world.doors:
            if memory[d.id]:
                dec, m, O = _decide(memory[d.id], cfg, model)
                use = [o for o in memory[d.id] if o.trusted] if cfg.protocol == "trusted" else memory[d.id]
                sig = 1 / np.sqrt(sum(1 / o.sig ** 2 for o in use)) if use else np.nan
                est[d.id] = (m, sig, O, memory[d.id][-1].trusted)
        self.frames.append(dict(pos=np.array(pos, float), pts=Q, est=est))

    def event(self, door_id, decision):
        self.events.append((len(self.frames) - 1, door_id, decision))


@dataclass
class Result:
    outcome: str = ""                # success | unsafe | no_route | plan_fail
    path_len: float = 0.0
    time: float = 0.0
    dwell_scans: int = 0
    repositions: int = 0
    t_first_decision: float = -1.0
    false_pass: int = 0
    false_fail: int = 0
    unresolved: int = 0
    decisions: list = field(default_factory=list)   # (door id, true margin, decision, est margin, O_trusted)
    paths: list = field(default_factory=list)


def _decide(obs, cfg, model):
    use = [o for o in obs if o.trusted] if cfg.protocol == "trusted" else [o for o in obs if o.d <= model.d_max]
    if not use:
        return "FLAG", np.nan, 0.0
    q = np.array([1 / o.sig ** 2 for o in use])
    m = float((q * np.array([o.m for o in use])).sum() / q.sum())
    sig = float(1 / np.sqrt(q.sum()))
    if cfg.protocol == "trusted":
        b = float((q * np.array([o.b for o in use])).sum() / q.sum())   # bound on the fused bias
    else:
        b = 0.0
    O = float(sum(1 / (o.sig ** 2 + (o.b ** 2 if model.density else 0.0)) for o in obs if o.trusted))
    if m - b >= cfg.z * sig:
        return "PASS", m, O
    if m + b <= -cfg.z * sig:
        return "FAIL", m, O
    return "FLAG", m, O


def _observe(world, pos, rng, cfg, model, memory, vmap):
    doors = [d for d in world.doors if pos[1] < d.yw - 0.05 and view_geometry(d, pos)[0] <= model.d_max]
    if not doors:
        if cfg.tracer is not None:
            cfg.tracer.scan(world, pos, np.zeros((0, 3)), memory, cfg, model)
        return
    P = scan(world, pos, rng, doors=doors)
    if cfg.sensor_mode == "voxelmap":
        vmap.add(P)
        P = vmap.points()
    for d in doors:
        dist, psi = view_geometry(d, pos)
        w = estimate_width(P, d)
        if not np.isfinite(w):
            continue
        if cfg.sensor_mode == "voxelmap":
            # the map already accumulates: keep only the latest estimate (no fusion)
            est = w * 100
            sig = float(np.hypot(model.sigma(dist), 2.04))     # + 5 cm voxel quantisation
            memory[d.id] = []
            b = model.b_res
        elif model.density:
            est = w * 100 - model.b0(dist)
            rho = min(edge_density(P, d)) if cfg.rho_source == "measured" else float(model.rho_hat(dist, psi))
            sig = float(model.sigma_rho(rho, psi))
            b = float(model.bias_rho(rho))
        else:
            est = w * 100 - model.b0(dist)
            sig = model.sigma(dist)
            b = model.b_res
        trusted = model.trusted(dist, psi) and jambs_visible(world, d, pos)
        memory[d.id].append(Obs(est - cfg.agent_w, sig, dist, psi, trusted, b))
    if cfg.tracer is not None:
        cfg.tracer.scan(world, pos, P, memory, cfg, model)


def _plan(world, pos, cfg, model, blocked, rng):
    free = free_with_blocked(world, blocked)
    active = {d.id for d in world.doors if d.id not in blocked}
    cost = cost_field(world, model, cfg.lam, active=active)
    if cfg.planner == "rrt":
        return rrt_star(world, pos, world.goal, cost, rng, free=free, iters=cfg.rrt_iters)
    return astar(world, pos, world.goal, cost, free=free)


def _viewpoint(world, door, pos, model, free, k_best=5):
    """Nearest reachable trusted viewpoint in front of the door (psi below psi_safe - 2 deg)."""
    cands = []
    for dy in np.arange(0.9, 2.51, 0.1):
        for dx in np.arange(-0.6, 0.61, 0.05):
            p = np.array([door.xc + dx, door.yw - dy])
            d, psi = view_geometry(door, p)
            i, j = int(p[0] / world.res), int(p[1] / world.res)
            if psi <= model.psi_safe - 2 and 0 <= i < free.shape[1] and 0 <= j < free.shape[0] and free[j, i] \
                    and jambs_visible(world, door, p):
                cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: np.linalg.norm(p - pos))
    best, best_len = None, np.inf
    ones = np.ones_like(free, float)
    for p in cands[:k_best]:
        path = astar(world, pos, p, ones, free=free)
        if path is not None and path_length(path) < best_len:
            best, best_len = path, path_length(path)
    return best


def _walk(world, path, cfg, model, rng, memory, vmap, res):
    pts, s = resample(path, cfg.step)
    for p in pts[1:]:
        _observe(world, p, rng, cfg, model, memory, vmap)
    res.paths.append(pts)
    res.path_len += float(s[-1])
    return pts[-1]


def run_mission(world, cfg, model, rng):
    res = Result()
    memory = {d.id: [] for d in world.doors}
    vmap = VoxelMap()
    blocked, pos = set(), np.asarray(world.start, float)
    for _ in range(cfg.max_replans + 1):
        path = _plan(world, pos, cfg, model, blocked, rng)
        if path is None:
            res.outcome = "no_route" if blocked else "plan_fail"
            res.time = res.path_len / cfg.speed + res.dwell_scans * 0.5
            return res
        door = world.door_crossed(path)
        if door is None:
            res.outcome = "plan_fail"
            return res
        pts, s = resample(path, cfg.step)
        dist = np.hypot(pts[:, 0] - door.xc, pts[:, 1] - door.yw)
        before = pts[:, 1] < door.yw - 0.05
        idx = np.where(before & (dist <= cfg.d_commit))[0]
        k = int(idx[0]) if len(idx) else int(np.where(before)[0][-1])
        for p in pts[:k + 1]:
            _observe(world, p, rng, cfg, model, memory, vmap)
        res.paths.append(pts[:k + 1])
        res.path_len += float(s[k])
        pos = pts[k]
        dec, m_est, O = _decide(memory[door.id], cfg, model)
        if dec == "FLAG" and cfg.protocol == "trusted" and cfg.reposition:
            vp = _viewpoint(world, door, pos, model, free_with_blocked(world, blocked))
            if vp is not None and len(vp) > 1:
                res.repositions += 1
                pos = _walk(world, vp, cfg, model, rng, memory, vmap, res)
                dec, m_est, O = _decide(memory[door.id], cfg, model)
        n = 0
        while dec == "FLAG" and n < cfg.dwell_max:
            _observe(world, pos, rng, cfg, model, memory, vmap)
            n += 1
            dec, m_est, O = _decide(memory[door.id], cfg, model)
        res.dwell_scans += n
        if res.t_first_decision < 0:
            res.t_first_decision = res.path_len / cfg.speed + res.dwell_scans * 0.5
        true_m = door.w * 100 - cfg.agent_w
        if cfg.tracer is not None:
            cfg.tracer.event(door.id, dec)
        res.decisions.append((door.id, round(true_m, 2), dec, None if not np.isfinite(m_est) else round(m_est, 2), round(O, 3)))
        if dec == "PASS":
            others = {d.id for d in world.doors if d.id != door.id}
            rest = astar(world, pos, world.goal, np.ones_like(world.free, float), free=free_with_blocked(world, others))
            if rest is not None:
                res.paths.append(rest)
                res.path_len += path_length(rest)
            res.time = res.path_len / cfg.speed + res.dwell_scans * 0.5
            if true_m >= 0:
                res.outcome = "success"
            else:
                res.false_pass += 1
                res.outcome = "unsafe"
            return res
        if dec == "FAIL" and true_m >= 0:
            res.false_fail += 1
        if dec == "FLAG":
            res.unresolved += 1
        blocked.add(door.id)
    res.outcome = "no_route"
    res.time = res.path_len / cfg.speed + res.dwell_scans * 0.5
    return res
