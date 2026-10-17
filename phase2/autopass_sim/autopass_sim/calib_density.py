"""Density / angle calibration sweep (Report 2 revision).

For a steel frame (5/8 in stop, 1-15/16 in rabbet) and a wood frame (3/8 in stop, 1-7/16 in
rabbet; the earliest-occluding frames in the simulator), the robot is placed at depth dy in
front of the frame face and laterally offset so the near-jamb sightline angle is psi.
Each pose gets `trials` independent 0.5 s L1 sweeps. Per sweep we record

  err    width estimate - true width (cm)
  rho    edge-support density: L1 returns within 5 cm of each stop tip, inside the frame slab
         and the 0.3-1.9 m estimator band; rho = min(left, right)  (points per sweep)

Outputs results/calib_density.csv (one row per sweep) and results/calib_density.json (fits).
Run: python -m autopass_sim.calib_density --trials 50
"""
import argparse, json, time
import numpy as np
from .world import World, Door, DOOR_HEIGHT
from .sensor import scan, estimate_width, view_geometry, edge_density

FRAMES = {"steel": (0.0159, 0.0492), "wood": (0.0095, 0.036)}


def sweep(trials, seed=21):
    rng = np.random.default_rng(seed)
    rows = []
    for fname, (s, r) in FRAMES.items():
        door = Door(0, 4.0, 5.0, 0.837, s, r)
        world = World("cal", 8, 8, 5.0, [door], [], np.array([4, 1.0]), np.array([4, 7.0])).build()
        for dy in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            for psi_t in range(0, 25, 2):
                dx = door.w / 2 + np.tan(np.radians(psi_t)) * (door.r + dy) if psi_t > 0 else 0.0
                pos = (door.xc - dx, door.yw - dy)
                d, psi = view_geometry(door, pos)
                for t in range(trials):
                    P = scan(world, pos, rng, doors=[door])
                    w = estimate_width(P, door)
                    nL, nR = edge_density(P, door)
                    rows.append((fname, dy, round(d, 3), round(psi, 2), t,
                                 (w - door.w) * 100 if np.isfinite(w) else np.nan, nL, nR))
    return rows


def fit(csv_path, calib_path="calib.json", psi_gate=12.5):
    """Fit the density-aware observation model from the sweep.

    sigma_scan(rho, psi) = exp(s0) * rho^s_rho * exp(s_psi * psi)     (cm, per sweep)
    rho_hat(d, psi)      = exp(r0) * d^r_d * exp(r_psi * psi)          (expected density)
    B(rho)               = max(b_min, kB / sqrt(rho))                  (bias bound, cm)
    kB is the envelope over every pose in the trusted zone (both frame types) of
    |mean residual bias after the square-on b0(d) correction| * sqrt(mean rho).
    """
    import pandas as pd
    from .observability import ObsModel
    M = ObsModel(calib_path)
    d = pd.read_csv(csv_path)
    d["rho"] = np.minimum(d.rhoL, d.rhoR)
    d["res"] = d.err_cm - d.d.map(M.b0)
    d["psi_t"] = (d.psi / 2).round() * 2
    g = d.groupby(["frame", "dy", "psi_t"]).agg(bias=("res", "mean"), sd=("err_cm", "std"),
                                                rho=("rho", "mean"), d=("d", "mean"),
                                                excess=("err_cm", "mean")).reset_index()
    base = g[g.psi_t == 0].set_index(["frame", "dy"]).excess
    g["excess"] = [r.excess - base[(r.frame, r.dy)] for r in g.itertuples()]
    t = g[g.psi_t <= psi_gate]
    X = np.c_[np.ones(len(t)), np.log(t.rho), t.psi_t]
    y = np.log(t.sd)
    cs, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = lambda yy, pp: float(1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    Xd = np.c_[np.ones(len(t)), np.log(t.d), t.psi_t]
    cd, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    Xr = np.c_[np.ones(len(g)), np.log(g.d), g.psi_t]
    cr, *_ = np.linalg.lstsq(Xr, np.log(g.rho), rcond=None)
    kB = float((t.bias.abs() * np.sqrt(t.rho)).max())
    onset = {}
    for f in g.frame.unique():
        dense = g[(g.frame == f) & (g.dy == g.dy.min())].sort_values("psi_t")
        onset[f] = dict(psi=dense.psi_t.tolist(), excess=[round(v, 3) for v in dense.excess])
    return dict(sigma=dict(s0=float(cs[0]), s_rho=float(cs[1]), s_psi=float(cs[2]), r2=r2(y, X @ cs),
                           r2_distance_model=r2(y, Xd @ cd)),
                rho=dict(r0=float(cr[0]), r_d=float(cr[1]), r_psi=float(cr[2]),
                         r2=r2(np.log(g.rho), Xr @ cr)),
                bias=dict(kB=kB, b_min=0.5, psi_gate=psi_gate),
                dense_view_excess=onset,
                poses=g.round(4).to_dict(orient="records"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", default="results/calib_density")
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    if a.fit_only:
        res = fit(a.out + ".csv")
        json.dump(res, open("calib_rho.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k not in ("poses",)}, indent=1))
        return
    t0 = time.time()
    rows = sweep(a.trials)
    import csv
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "dy", "d", "psi", "trial", "err_cm", "rhoL", "rhoR"])
        w.writerows(rows)
    print("wrote", a.out + ".csv", len(rows), "sweeps in", round(time.time() - t0), "s")
    res = fit(a.out + ".csv")
    json.dump(res, open("calib_rho.json", "w"), indent=1)
    print("wrote calib_rho.json")


if __name__ == "__main__":
    main()
