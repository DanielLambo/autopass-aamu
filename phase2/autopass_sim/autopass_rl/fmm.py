"""Fast-marching paths on the inflated occupancy grid (path_mode="fmm").

travel_field() solves the Eikonal equation |grad T| = 1 (unit speed, so T is geodesic distance in m)
on the free cells of the same 0.1 m grid A* uses, seeded at one target point. Blocked cells and
cells not connected to the target are inf. descend() follows -grad T from a start point to the
target. The result is a smooth path that is not tied to the 8 grid directions.
"""
import numpy as np
import skfmm

STEP = 0.05          # m, gradient-descent step
GOAL_TOL = 0.12      # m, switch to a straight final segment inside this radius


def travel_field(free, res, target):
    """Geodesic distance (m) from every free cell centre to `target`, inf where unreachable."""
    ny, nx = free.shape
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res)
    tx, ty = int(target[0] / res), int(target[1] / res)
    if not (0 <= tx < nx and 0 <= ty < ny and free[ty, tx]):
        return np.full((ny, nx), np.inf)
    r = np.hypot(X - target[0], Y - target[1])
    r0 = res                                            # seed disc: at least the target's own cell
    phi = np.ma.MaskedArray(r - r0, mask=~free)
    d = skfmm.distance(phi, dx=res)
    T = np.where(np.ma.getmaskarray(d), np.inf, np.ma.getdata(d) + r0)
    T[(r <= r0) & free] = r[(r <= r0) & free]
    T[~free] = np.inf
    return T


def _grad(T, free, res):
    """Cell-centred gradient of T; one-sided next to blocked or unreachable cells."""
    F = np.where(np.isfinite(T), T, np.nan)
    gx, gy = np.zeros_like(T), np.zeros_like(T)
    for axis, g in ((1, gx), (0, gy)):
        fwd = np.roll(F, -1, axis)
        bwd = np.roll(F, 1, axis)
        if axis == 1:
            fwd[:, -1] = np.nan; bwd[:, 0] = np.nan
        else:
            fwd[-1, :] = np.nan; bwd[0, :] = np.nan
        both = np.isfinite(fwd) & np.isfinite(bwd)
        g[both] = (fwd - bwd)[both] / (2 * res)
        only_f = np.isfinite(fwd) & ~np.isfinite(bwd)
        g[only_f] = (fwd - F)[only_f] / res
        only_b = np.isfinite(bwd) & ~np.isfinite(fwd)
        g[only_b] = (F - bwd)[only_b] / res
    gx[~np.isfinite(F)] = 0.0
    gy[~np.isfinite(F)] = 0.0
    return gx, gy


def _bilinear(A, p, res):
    ny, nx = A.shape
    x, y = p[0] / res - 0.5, p[1] / res - 0.5
    i0, j0 = int(np.clip(np.floor(x), 0, nx - 2)), int(np.clip(np.floor(y), 0, ny - 2))
    fx, fy = np.clip(x - i0, 0, 1), np.clip(y - j0, 0, 1)
    return ((1 - fx) * (1 - fy) * A[j0, i0] + fx * (1 - fy) * A[j0, i0 + 1] +
            (1 - fx) * fy * A[j0 + 1, i0] + fx * fy * A[j0 + 1, i0 + 1])


def value_at(T, p, res):
    """Field value at a continuous point: bilinear over finite neighbours, else the cell value."""
    ny, nx = T.shape
    x, y = p[0] / res - 0.5, p[1] / res - 0.5
    i0, j0 = int(np.clip(np.floor(x), 0, nx - 2)), int(np.clip(np.floor(y), 0, ny - 2))
    block = T[j0:j0 + 2, i0:i0 + 2]
    if np.all(np.isfinite(block)):
        return float(_bilinear(T, p, res))
    ci, cj = int(p[0] / res), int(p[1] / res)
    if not (0 <= ci < nx and 0 <= cj < ny) or not np.isfinite(T[cj, ci]):
        return np.inf
    centre = np.array([(ci + 0.5) * res, (cj + 0.5) * res])
    return float(T[cj, ci] + np.linalg.norm(np.asarray(p) - centre))


def descend(T, free, res, start, target, grad=None, max_len=200.0):
    """Path from start to target following -grad T. None if the start cell cannot reach the target."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    ny, nx = T.shape
    ci, cj = int(start[0] / res), int(start[1] / res)
    if not (0 <= ci < nx and 0 <= cj < ny) or not np.isfinite(T[cj, ci]):
        return None
    gx, gy = grad if grad is not None else _grad(T, free, res)
    pts, p = [start], start.copy()
    n_max = int(max_len / STEP)
    for _ in range(n_max):
        if np.linalg.norm(p - target) <= GOAL_TOL:
            break
        g = np.array([_bilinear(gx, p, res), _bilinear(gy, p, res)])
        ng = np.linalg.norm(g)
        q = p - STEP * g / ng if ng > 1e-9 else None
        if q is None or not _is_free(free, q, res) or value_at(T, q, res) >= value_at(T, p, res):
            q = _cell_step(T, p, res)              # fallback: steepest descent over the 8 neighbour cells
            if q is None:
                return None
        pts.append(q)
        p = q
    else:
        return None
    pts.append(target)
    return np.array(pts)


def _is_free(free, p, res):
    i, j = int(p[0] / res), int(p[1] / res)
    return 0 <= i < free.shape[1] and 0 <= j < free.shape[0] and bool(free[j, i])


def _cell_step(T, p, res):
    ci, cj = int(p[0] / res), int(p[1] / res)
    best, bv = None, T[cj, ci]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            i, j = ci + dx, cj + dy
            if (dx or dy) and 0 <= j < T.shape[0] and 0 <= i < T.shape[1] and T[j, i] < bv:
                if dx and dy and not (np.isfinite(T[cj, i]) and np.isfinite(T[j, ci])):
                    continue                       # no corner cutting, as in A*
                best, bv = np.array([(i + 0.5) * res, (j + 0.5) * res]), T[j, i]
    return best


class FieldCache:
    """Per-episode cache: one travel-time field (and its gradient) per (free-grid key, target)."""

    def __init__(self, res):
        self.res, self._f = res, {}

    def get(self, key, free, target):
        if key not in self._f:
            T = travel_field(free, self.res, target)
            self._f[key] = (T, _grad(T, free, self.res))
        return self._f[key]

    def path(self, key, free, start, target):
        T, g = self.get(key, free, target)
        return descend(T, free, self.res, start, target, grad=g)
