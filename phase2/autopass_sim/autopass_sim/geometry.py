"""2D segment geometry and vectorized ray casting.

Every structure in the world is a vertical extrusion of a 2D segment, so a 3D LiDAR
ray can be cast in 2D (horizontal distance) and lifted back to 3D with its elevation.
"""
import numpy as np


def box_segments(x0, y0, x1, y1):
    """Four segments of an axis-aligned box (closed polygon)."""
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def raycast(origin, dirs, segs):
    """Horizontal hit distance of each ray against a set of segments.

    origin: (2,) ray start. dirs: (N, 2) unit horizontal directions.
    segs: (M, 4) segments as x0, y0, x1, y1.
    Returns (N,) distance to the first hit (inf where nothing is hit).
    """
    ox, oy = origin
    px, py = segs[:, 0], segs[:, 1]
    ex, ey = segs[:, 2] - px, segs[:, 3] - py
    dx, dy = dirs[:, 0:1], dirs[:, 1:2]
    denom = dx * ey - dy * ex                       # (N, M)
    qx, qy = px - ox, py - oy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qx * ey - qy * ex) / denom             # distance along ray
        u = (qx * dy - qy * dx) / denom             # position along segment
    ok = (np.abs(denom) > 1e-12) & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
    t = np.where(ok, t, np.inf)
    return t.min(axis=1)


def segment_hits_grid(p0, p1, free, res, step=0.05):
    """True if the straight segment p0->p1 stays in free grid cells."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
    pts = p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]
    ij = np.floor(pts / res).astype(int)
    h, w = free.shape
    if (ij[:, 0] < 0).any() or (ij[:, 1] < 0).any() or (ij[:, 0] >= w).any() or (ij[:, 1] >= h).any():
        return False
    return bool(free[ij[:, 1], ij[:, 0]].all())
