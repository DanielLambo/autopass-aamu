"""Geometric planners on the inflated occupancy grid.

All planners minimise the same path cost
    J(path) = integral of c(x) ds,   c(x) = 1 + lambda * g(x)  (see observability.py)
lambda = 0 gives the shortest path. Paths are post-processed with a cost-aware
shortcut, so smoothing never trades observability away for length.
Geometric only: the Go2 is treated as a disc of radius `inflate`; gait dynamics are
out of scope at this stage (per Dr. Fu: geometric planning first).
"""
import heapq
import numpy as np
from .geometry import segment_hits_grid

SQ2 = np.sqrt(2.0)
NBRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


def free_with_blocked(world, blocked):
    """Copy of the free grid with the openings of `blocked` door ids closed."""
    free = world.free.copy()
    if not blocked:
        return free
    ny, nx = free.shape
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * world.res, (np.arange(ny) + 0.5) * world.res)
    for d in world.doors:
        if d.id in blocked:
            m = (np.abs(X - d.xc) <= d.hf + world.inflate + world.res) & \
                (Y >= d.yw - world.inflate) & (Y <= d.yw + d.D + world.inflate)
            free[m] = False
    return free


def _cell(p, res):
    return int(p[0] / res), int(p[1] / res)


def _center(c, res):
    return np.array([(c[0] + 0.5) * res, (c[1] + 0.5) * res])


def seg_cost(p0, p1, cost, res, step=0.05):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L == 0:
        return 0.0
    n = max(2, int(np.ceil(L / step)) + 1)
    pts = p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]
    ij = np.clip(np.floor(pts / res).astype(int), 0, [cost.shape[1] - 1, cost.shape[0] - 1])
    return L * float(cost[ij[:, 1], ij[:, 0]].mean())


def path_cost(path, cost, res):
    return sum(seg_cost(a, b, cost, res) for a, b in zip(path[:-1], path[1:]))


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0


def shortcut(path, free, cost, res):
    """Greedy cost-aware shortcutting: replace a run of waypoints with a straight segment
    only if it is collision-free and not more expensive."""
    if len(path) < 3:
        return path
    out, i = [path[0]], 0
    pre = [0.0]
    for a, b in zip(path[:-1], path[1:]):
        pre.append(pre[-1] + seg_cost(a, b, cost, res))
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if segment_hits_grid(path[i], path[j], free, res) and \
                    seg_cost(path[i], path[j], cost, res) <= pre[j] - pre[i] + 1e-9:
                break
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


def astar(world, start, goal, cost, free=None):
    free = world.free if free is None else free
    res = world.res
    ny, nx = free.shape
    s, g = _cell(start, res), _cell(goal, res)
    if not (free[s[1], s[0]] and free[g[1], g[0]]):
        return None
    gx, gy = g
    dist = np.full((ny, nx), np.inf)
    parent = -np.ones((ny, nx), dtype=np.int64)
    dist[s[1], s[0]] = 0.0
    openq = [(0.0, s[0], s[1])]
    while openq:
        f, x, y = heapq.heappop(openq)
        if (x, y) == g:
            break
        dxy = dist[y, x]
        if f - np.hypot(x - gx, y - gy) * res > dxy + 1e-9:
            continue
        for ddx, ddy, w in NBRS:
            xx, yy = x + ddx, y + ddy
            if 0 <= xx < nx and 0 <= yy < ny and free[yy, xx]:
                if ddx and ddy and not (free[y, xx] and free[yy, x]):
                    continue                           # no corner cutting
                nd = dxy + w * res * 0.5 * (cost[y, x] + cost[yy, xx])
                if nd < dist[yy, xx]:
                    dist[yy, xx] = nd
                    parent[yy, xx] = y * nx + x
                    heapq.heappush(openq, (nd + np.hypot(xx - gx, yy - gy) * res, xx, yy))
    if not np.isfinite(dist[gy, gx]):
        return None
    cells, k = [], gy * nx + gx
    while k != -1:
        cells.append((k % nx, k // nx))
        k = parent[k // nx, k % nx]
    path = [np.asarray(start, float)] + [_center(c, res) for c in reversed(cells[1:-1])] + [np.asarray(goal, float)]
    return shortcut(np.array(path), free, cost, res)


def rrt_star(world, start, goal, cost, rng, free=None, iters=2500, step=0.6, goal_tol=0.3,
             gamma=6.0, goal_bias=0.08):
    free = world.free if free is None else free
    res = world.res
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    V = np.zeros((iters + 1, 2)); V[0] = start
    C = np.zeros(iters + 1)
    P = -np.ones(iters + 1, dtype=int)
    n = 1
    best, best_c = -1, np.inf
    for _ in range(iters):
        x = goal if rng.random() < goal_bias else np.array([rng.uniform(0, world.W), rng.uniform(0, world.H)])
        d = np.linalg.norm(V[:n] - x, axis=1)
        i_near = int(d.argmin())
        v = V[i_near] + (x - V[i_near]) * min(1.0, step / max(d[i_near], 1e-9))
        if not world.is_free(v) or not free[int(v[1] / res), int(v[0] / res)]:
            continue
        if not segment_hits_grid(V[i_near], v, free, res):
            continue
        r = min(1.5, gamma * np.sqrt(np.log(n + 1) / (n + 1)))
        nb = np.where(np.linalg.norm(V[:n] - v, axis=1) <= r)[0]
        par, cmin = i_near, C[i_near] + seg_cost(V[i_near], v, cost, res)
        edge = {}
        for j in nb:
            if segment_hits_grid(V[j], v, free, res):
                edge[j] = seg_cost(V[j], v, cost, res)
                if C[j] + edge[j] < cmin:
                    par, cmin = j, C[j] + edge[j]
        V[n], C[n], P[n] = v, cmin, par
        for j, e in edge.items():                    # rewire
            if cmin + e < C[j]:
                C[j], P[j] = cmin + e, n
        if np.linalg.norm(v - goal) <= goal_tol and segment_hits_grid(v, goal, free, res):
            cg = cmin + seg_cost(v, goal, cost, res)
            if cg < best_c:
                best, best_c = n, cg
        n += 1
    if best < 0:
        return None
    pts, k = [goal], best
    while k != -1:
        pts.append(V[k]); k = P[k]
    return shortcut(np.array(pts[::-1]), free, cost, res)


def resample(path, ds=0.25):
    """Points every ds metres along a polyline (the robot's scan positions)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], ds)
    if len(t) == 0 or t[-1] < s[-1]:
        t = np.append(t, s[-1])
    return np.stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])], 1), t
