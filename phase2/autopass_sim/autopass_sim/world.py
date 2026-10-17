"""Procedural floor plans: a dividing wall with several doors and alternative routes.

Layouts
  room      open room on the start side, furniture boxes, 2-3 doors into the goal room
  corridor  2.4 m corridor on the start side (robot travels parallel to the door wall)

Door frames use a double-rabbet steel-frame profile: a stop of depth s set back r
from each face of a frame of depth D. Clear width w is stop-to-stop.
Defaults follow SDI frames (5/8 in stop, 1-15/16 in rabbet) and wood stops (3/8 in).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from .geometry import box_segments

FRAME_DEPTH = 0.146   # m (5-3/4 in frame)
DOOR_HEIGHT = 2.03    # m (80 in)


@dataclass
class Door:
    id: int
    xc: float        # opening centre (x)
    yw: float        # robot-side wall face (y)
    w: float         # true clear width, stop to stop (m)
    s: float         # stop depth (m)
    r: float         # rabbet = stop setback from each face (m)
    D: float = FRAME_DEPTH

    @property
    def hf(self):            # half width between soffits
        return self.w / 2 + self.s

    @property
    def center(self):
        return np.array([self.xc, self.yw])

    def segments(self):
        """Soffits and both stops (the wall faces are added by the world)."""
        xc, yw, hf, s, r, D = self.xc, self.yw, self.hf, self.s, self.r, self.D
        segs = [(xc - hf, yw, xc - hf, yw + D), (xc + hf, yw, xc + hf, yw + D)]
        ws = D - 2 * r
        segs += box_segments(xc - hf, yw + r, xc - hf + s, yw + r + ws)
        segs += box_segments(xc + hf - s, yw + r, xc + hf, yw + r + ws)
        return segs


@dataclass
class World:
    layout: str
    W: float
    H: float
    wall_y: float
    doors: list
    boxes: list                      # (x0, y0, x1, y1)
    start: np.ndarray
    goal: np.ndarray
    res: float = 0.1
    inflate: float = 0.20
    segs: np.ndarray = field(default=None, repr=False)
    free: np.ndarray = field(default=None, repr=False)

    def build(self):
        segs = box_segments(0, 0, self.W, self.H)
        y0, y1 = self.wall_y, self.wall_y + FRAME_DEPTH
        edges = [0.0]
        for d in sorted(self.doors, key=lambda d: d.xc):
            edges += [d.xc - d.hf, d.xc + d.hf]
            segs += d.segments()
        edges.append(self.W)
        for a, b in zip(edges[0::2], edges[1::2]):      # wall faces between openings
            segs += [(a, y0, b, y0), (a, y1, b, y1)]
        for bx in self.boxes:
            segs += box_segments(*bx)
        self.segs = np.array(segs, float)
        self.free = self._grid()
        return self

    def _grid(self):
        nx, ny = int(np.ceil(self.W / self.res)), int(np.ceil(self.H / self.res))
        xs = (np.arange(nx) + 0.5) * self.res
        ys = (np.arange(ny) + 0.5) * self.res
        X, Y = np.meshgrid(xs, ys)
        occ = np.zeros((ny, nx), bool)
        slab = (Y >= self.wall_y) & (Y <= self.wall_y + FRAME_DEPTH)
        opening = np.zeros_like(slab)
        for d in self.doors:
            opening |= np.abs(X - d.xc) < d.w / 2
        occ |= slab & ~opening
        for x0, y0, x1, y1 in self.boxes:
            occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        k = int(np.ceil(self.inflate / self.res))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = (xx ** 2 + yy ** 2) * self.res ** 2 <= self.inflate ** 2 + 1e-9
        return ~ndimage.binary_dilation(occ, structure=disk)

    def is_free(self, p):
        i, j = int(p[0] / self.res), int(p[1] / self.res)
        h, w = self.free.shape
        return 0 <= i < w and 0 <= j < h and bool(self.free[j, i])

    def door_crossed(self, path):
        """Door whose opening the path passes through (first one), or None."""
        for a, b in zip(path[:-1], path[1:]):
            if (a[1] - self.wall_y) * (b[1] - self.wall_y) <= 0 and a[1] != b[1]:
                t = (self.wall_y - a[1]) / (b[1] - a[1])
                x = a[0] + t * (b[0] - a[0])
                for d in self.doors:
                    if abs(x - d.xc) <= d.hf + 0.05:
                        return d
        return None


def _reachable(free, a, b, res):
    lab, _ = ndimage.label(free)
    ia = lab[int(a[1] / res), int(a[0] / res)]
    ib = lab[int(b[1] / res), int(b[0] / res)]
    return ia != 0 and ia == ib


def make_world(rng, layout="room", agent_w=0.813, n_doors=None, max_tries=200):
    for _ in range(max_tries):
        w = _sample(rng, layout, agent_w, n_doors).build()
        if not (w.is_free(w.start) and w.is_free(w.goal)):
            continue
        # every door must be usable: test with only that door open
        ok = True
        for d in w.doors:
            single = World(w.layout, w.W, w.H, w.wall_y, [d], w.boxes, w.start, w.goal).build()
            if not _reachable(single.free, w.start, w.goal, w.res):
                ok = False
                break
        if ok:
            return w
    raise RuntimeError("could not generate a valid world")


def _door_widths(rng, n, agent_w):
    """One clearly passable door, the rest within +/-3 cm of the agent requirement."""
    ws = [agent_w + rng.uniform(0.07, 0.13)]
    ws += list(agent_w + rng.uniform(-0.03, 0.03, n - 1))
    rng.shuffle(ws)
    return ws


def _frame(rng):
    if rng.random() < 0.6:   # steel frame (SDI): 5/8 in stop, 1-15/16 in rabbet
        return 0.0159, rng.uniform(0.045, 0.0492)
    return 0.0095, rng.uniform(0.033, 0.036)   # wood stop 3/8 in on a 1-3/8 in door


def _sample(rng, layout, agent_w, n_doors):
    n = n_doors or int(rng.integers(2, 4))
    if layout == "corridor":
        W, H = rng.uniform(12, 16), rng.uniform(8, 10)
        wall_y = 2.4
        xs = _spread(rng, n, 3.0, W - 1.2, 2.5)
        start = np.array([rng.uniform(0.6, 1.0), 1.2])
        boxes = []
        for _ in range(int(rng.integers(0, 3))):       # bins/benches against the far wall
            x = rng.uniform(2.0, W - 1.5)
            boxes.append((x, 0.1, x + rng.uniform(0.4, 0.9), 0.1 + rng.uniform(0.3, 0.5)))
    else:
        W, H = rng.uniform(10, 14), rng.uniform(9, 12)
        wall_y = rng.uniform(0.45, 0.55) * H
        xs = _spread(rng, n, 1.2, W - 1.2, 2.5)
        start = np.array([rng.uniform(1.0, W - 1.0), rng.uniform(0.7, 1.4)])
        boxes = []
        for _ in range(int(rng.integers(2, 6))):       # furniture on the start side
            bw, bh = rng.uniform(0.4, 1.4), rng.uniform(0.4, 1.0)
            x = rng.uniform(0.3, W - 0.3 - bw)
            y = rng.uniform(1.8, wall_y - 0.5 - bh)
            if y > 1.8:
                boxes.append((x, y, x + bw, y + bh))
    goal = np.array([rng.uniform(1.0, W - 1.0), rng.uniform(wall_y + 1.5, H - 0.7)])
    for _ in range(int(rng.integers(0, 3))):           # furniture on the goal side
        bw, bh = rng.uniform(0.4, 1.2), rng.uniform(0.4, 1.0)
        x = rng.uniform(0.3, W - 0.3 - bw)
        y = rng.uniform(wall_y + 1.0, H - 0.3 - bh)
        if np.hypot(x + bw / 2 - goal[0], y + bh / 2 - goal[1]) > 1.2:
            boxes.append((x, y, x + bw, y + bh))
    doors = []
    for i, (x, w) in enumerate(zip(xs, _door_widths(rng, n, agent_w))):
        s, r = _frame(rng)
        doors.append(Door(i, float(x), float(wall_y), float(w), s, r))
    return World(layout, float(W), float(H), float(wall_y), doors, boxes, start, goal)


def _spread(rng, n, lo, hi, gap):
    for _ in range(1000):
        xs = np.sort(rng.uniform(lo, hi, n))
        if n == 1 or np.diff(xs).min() >= gap:
            return xs
    return np.linspace(lo, hi, n)
