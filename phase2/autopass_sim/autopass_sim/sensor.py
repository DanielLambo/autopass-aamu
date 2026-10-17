"""LiDAR model based on the Unitree L1 and a GAIT-style doorway width estimator.

From the L1 spec sheet: 21,600 effective points/s, 360 deg x 90 deg field of view,
+/-2 cm range accuracy, 8 mm range resolution.
Assumed (to be calibrated on the Phase 1 bags): sensor height 0.30 m, elevation band
-15..75 deg, 1 cm range noise (1 sigma), uniform random scan pattern.

Two output modes:
  raw       every observation is a fresh 0.5 s scan (independent scans)
  voxelmap  points are snapped to a 5 cm grid and accumulated into one map, which is
            what the Go2's WebRTC /point_cloud2 topic delivers (consecutive
            observations are NOT independent)
"""
import numpy as np
from .geometry import raycast
from .world import DOOR_HEIGHT

PTS_PER_SCAN = 10800     # 0.5 s at 21,600 pts/s
SENSOR_Z = 0.30
SIG_R = 0.010
RES_R = 0.008
VOXEL = 0.05


def scan(world, pos, rng, n=PTS_PER_SCAN, doors=None, margin=0.8):
    """One simulated scan from planar position pos. Returns (K, 3) points.

    If `doors` is given, only rays aimed within `margin` metres of those doors are
    cast. The ray density is unchanged, so door points are statistically identical to
    a full scan; this only skips rays that can never reach a doorway.
    """
    az = rng.uniform(0, 2 * np.pi, n)
    el = np.radians(rng.uniform(-15, 75, n))
    if doors is not None:
        keep = np.zeros(n, bool)
        for d in doors:
            dx, dy = d.xc - pos[0], d.yw + d.D / 2 - pos[1]
            dist = max(np.hypot(dx, dy), 1e-6)
            half = np.arctan2(d.hf + margin, dist) if dist > d.hf + margin else np.pi
            diff = np.angle(np.exp(1j * (az - np.arctan2(dy, dx))))
            keep |= np.abs(diff) <= half
        az, el = az[keep], el[keep]
    dirs = np.stack([np.cos(az), np.sin(az)], 1)
    th = raycast(np.asarray(pos, float), dirs, world.segs)          # horizontal distance
    z = SENSOR_Z + th * np.tan(el)
    hit = np.isfinite(th) & (z >= 0.0) & (z <= 3.0)
    # rays through an opening above the door head would hit the header; those points sit
    # above the estimator's 0.3-1.9 m band, so the missing header does not affect widths
    r = th[hit] / np.cos(el[hit])
    r = np.round((r + rng.normal(0, SIG_R, r.size)) / RES_R) * RES_R
    d3 = np.stack([np.cos(el[hit]) * dirs[hit, 0], np.cos(el[hit]) * dirs[hit, 1], np.sin(el[hit])], 1)
    o = np.array([pos[0], pos[1], SENSOR_Z])
    return o + r[:, None] * d3


def voxelize(P, size=VOXEL):
    """Snap points to voxel centres and drop duplicates (Go2 voxel-map behaviour)."""
    if len(P) == 0:
        return P
    k = np.unique(np.floor(P / size).astype(np.int64), axis=0)
    return (k + 0.5) * size


class VoxelMap:
    """Accumulating 5 cm voxel map in the world frame."""

    def __init__(self, size=VOXEL):
        self.size = size
        self.keys = set()

    def add(self, P):
        for k in map(tuple, np.floor(P / self.size).astype(np.int64)):
            self.keys.add(k)

    def points(self):
        if not self.keys:
            return np.zeros((0, 3))
        return (np.array(list(self.keys), float) + 0.5) * self.size


def estimate_width(P, door, k=2, min_pts=6):
    """Innermost jamb points inside the frame slab, per side (simplified GAIT edge step).

    Returns the width estimate in metres, or nan if either jamb is not seen.
    """
    xc, yw, hf, D = door.xc, door.yw, door.hf, door.D
    m = ((P[:, 1] > yw - 0.03) & (P[:, 1] < yw + D + 0.03) &
         (P[:, 2] > 0.3) & (P[:, 2] < min(1.9, DOOR_HEIGHT)) &
         (np.abs(P[:, 0] - xc) < hf + 0.6))
    x = P[m, 0]
    L, R = np.sort(x[x < xc]), np.sort(x[x > xc])
    if len(L) < min_pts or len(R) < min_pts:
        return np.nan
    return float(R[k - 1] - L[-k])


def edge_density(P, door, win=0.05):
    """Edge-support density rho: returns within `win` of each stop tip, inside the frame slab
    and the 0.3-1.9 m estimator band. Returns (left, right) counts for one sweep."""
    xc, yw, D = door.xc, door.yw, door.D
    m = ((P[:, 1] > yw - 0.03) & (P[:, 1] < yw + D + 0.03) &
         (P[:, 2] > 0.3) & (P[:, 2] < min(1.9, DOOR_HEIGHT)))
    x = P[m, 0]
    nL = int(((x < xc) & (np.abs(x - (xc - door.w / 2)) <= win)).sum())
    nR = int(((x > xc) & (np.abs(x - (xc + door.w / 2)) <= win)).sum())
    return nL, nR


def view_geometry(door, pos):
    """Distance to the door centre and the near-jamb sightline angle psi (deg)."""
    dx, dy = pos[0] - door.xc, door.yw - pos[1]
    d = float(np.hypot(dx, dy))
    if dy <= 0:
        return d, 90.0
    psi = np.degrees(np.arctan(max(0.0, abs(dx) - door.w / 2) / (door.r + dy)))
    return d, float(psi)


def jambs_visible(world, door, pos):
    """True if the lines of sight to both stop tips reach the door wall unobstructed.

    A jamb hidden behind furniture leaves a shadow next to the opening that looks like
    part of the gap, so an occluded view can over-read the width by tens of cm.
    """
    pos = np.asarray(pos, float)
    tips = np.array([[door.xc - door.w / 2, door.yw + door.r], [door.xc + door.w / 2, door.yw + door.r]])
    v = tips - pos
    dist = np.linalg.norm(v, axis=1)
    t = raycast(pos, v / dist[:, None], world.segs)
    hit_y = pos[1] + t * v[:, 1] / dist
    return bool(np.all(hit_y >= door.yw - 0.02))
