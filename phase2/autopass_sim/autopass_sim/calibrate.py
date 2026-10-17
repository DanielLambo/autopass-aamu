"""Calibration sweep: per-observation width error vs distance and approach angle.

Produces calib.json with
  b0(d)   distance-dependent bias at a square-on view (removed by calibration, as one
          would calibrate GAIT against tape-measured doors)
  sigma(d) per-observation noise at a square-on view, fitted as a + c * d^p
  excess(psi) extra bias from the hidden stop, binned by near-jamb sightline angle
The same table drives the planner's observability field and the decision rule.
Run: python -m autopass_sim.calibrate --trials 100
"""
import argparse, json
import numpy as np
from .world import World, Door
from .sensor import scan, estimate_width, view_geometry


def sweep(trials=100, seed=11):
    rng = np.random.default_rng(seed)
    door = Door(0, 4.0, 5.0, 0.837, 0.0159, 0.0492)
    world = World("cal", 8, 8, 5.0, [door], [], np.array([4, 1.0]), np.array([4, 7.0])).build()
    ds = np.round(np.arange(0.75, 4.01, 0.25), 2)
    phis = np.arange(0, 61, 10)
    rows = []
    for d in ds:
        for ph in phis:
            pos = (door.xc - d * np.sin(np.radians(ph)), door.yw - d * np.cos(np.radians(ph)))
            _, psi = view_geometry(door, pos)
            e = [estimate_width(scan(world, pos, rng, doors=[door]), door) for _ in range(trials)]
            e = (np.array([v for v in e if np.isfinite(v)]) - door.w) * 100
            rows.append(dict(d=float(d), phi=float(ph), psi=psi, n=int(e.size),
                             bias=float(e.mean()) if e.size else None,
                             sd=float(e.std(ddof=1)) if e.size > 2 else None))
    return rows


def fit(rows):
    ds = sorted({r["d"] for r in rows})
    sq = {r["d"]: r for r in rows if r["phi"] == 0}
    b0 = [sq[d]["bias"] for d in ds]
    sd = np.array([sq[d]["sd"] for d in ds])
    x = np.array(ds)
    best = None
    for p in np.arange(1.0, 4.01, 0.25):
        A = np.vstack([np.ones_like(x), x ** p]).T
        c, *_ = np.linalg.lstsq(A, sd, rcond=None)
        err = float(((A @ c - sd) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, float(p), [float(v) for v in c])
    bins = [0, 0.01, 5, 10, 12.5, 15, 17.5, 20, 25, 90]
    excess = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        v = [r["bias"] - sq[r["d"]]["bias"] for r in rows if lo <= r["psi"] < hi and r["bias"] is not None]
        if v:
            excess.append(dict(lo=lo, hi=hi, mean=float(np.mean(v)), max=float(np.max(v)), n=len(v)))
    return dict(d=ds, b0=b0, sigma_fit=dict(a=best[2][0], c=best[2][1], p=best[1]),
                sigma_points=[float(v) for v in sd], excess=excess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", default="calib.json")
    a = ap.parse_args()
    rows = sweep(a.trials)
    cal = fit(rows)
    cal["rows"] = rows
    json.dump(cal, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in cal.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
