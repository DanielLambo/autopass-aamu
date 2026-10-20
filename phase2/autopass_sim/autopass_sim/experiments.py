"""Batch experiments: every method runs on the same generated worlds.

Methods
  B1  shortest path (A*, lambda=0) + naive decision           (measure-while-walking)
  B2  shortest path (A*, lambda=0) + trusted rule + reposition (reactive fix)
  P-A*(l)   observability-aware A* + trusted rule + reposition (proposed), lambda sweep
  P-A*(4)n  observability-aware A* + naive rule              (isolates the path's effect)
  P-RRT*    observability-aware RRT* + trusted rule + reposition

Run:  python -m autopass_sim.experiments --worlds 100 --rrt-worlds 30 --out results
"""
import argparse, csv, json, os, time
from multiprocessing import Pool
import numpy as np
from .world import make_world
from .observability import ObsModel
from .mission import Config, run_mission

METHODS = {
    "B1": dict(planner="astar", lam=0.0, protocol="naive"),
    "B2": dict(planner="astar", lam=0.0, protocol="trusted"),
    "P-A*(1)": dict(planner="astar", lam=1.0, protocol="trusted"),
    "P-A*(2)": dict(planner="astar", lam=2.0, protocol="trusted"),
    "P-A*(4)": dict(planner="astar", lam=4.0, protocol="trusted"),
    "P-A*(8)": dict(planner="astar", lam=8.0, protocol="trusted"),
    "P-A*(4)n": dict(planner="astar", lam=4.0, protocol="naive"),
    "P-RRT*(4)": dict(planner="rrt", lam=4.0, protocol="trusted"),
    # harder scenarios (Report 2 rev.): no repositioning, and no repositioning + short dwell budget
    "B2-nr": dict(planner="astar", lam=0.0, protocol="trusted", reposition=False),
    "P-A*(4)-nr": dict(planner="astar", lam=4.0, protocol="trusted", reposition=False),
    "B2-b4": dict(planner="astar", lam=0.0, protocol="trusted", reposition=False, dwell_max=4),
    "P-A*(4)-b4": dict(planner="astar", lam=4.0, protocol="trusted", reposition=False, dwell_max=4),
    # ablation: density model driven by expected rho_hat(d, psi) instead of the measured per-sweep rho
    "B1-rhohat": dict(planner="astar", lam=0.0, protocol="naive", rho_source="geometry"),
}
DEFAULT = ["B1", "B2", "P-A*(1)", "P-A*(2)", "P-A*(4)", "P-A*(8)", "P-A*(4)n", "P-RRT*(4)"]

_MODEL = None


def _init(calib, floor, rho_calib=None):
    global _MODEL
    _MODEL = ObsModel(calib, floor=floor, rho_calib=rho_calib or None)


def _job(args):
    layout, wseed, method, sensor_mode = args
    world = make_world(np.random.default_rng(wseed), layout)
    cfg = Config(sensor_mode=sensor_mode, **METHODS[method])
    t0 = time.time()
    r = run_mission(world, cfg, _MODEL, np.random.default_rng(wseed * 101 + 7))
    first = r.decisions[0] if r.decisions else (None, np.nan, "NONE", None, 0.0)
    return dict(layout=layout, world=wseed, method=method, sensor=sensor_mode, outcome=r.outcome,
                path_len=round(r.path_len, 3), time=round(r.time, 2), dwell_scans=r.dwell_scans,
                repositions=r.repositions, t_first_decision=round(r.t_first_decision, 2),
                false_pass=r.false_pass, false_fail=r.false_fail, unresolved=r.unresolved,
                n_decisions=len(r.decisions),
                first_true_m=first[1], first_decision=first[2],
                first_est_err=(abs(first[3] - first[1]) if first[3] is not None else ""),
                first_O=first[4], cpu_s=round(time.time() - t0, 2),
                decisions=json.dumps(r.decisions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=100)
    ap.add_argument("--rrt-worlds", type=int, default=30)
    ap.add_argument("--layouts", default="room,corridor")
    ap.add_argument("--sensor", default="raw")
    ap.add_argument("--floor", type=float, default=1.0)
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--rho-calib", default="calib_rho.json", help="density-aware model; '' for the distance-only model")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--methods", default=",".join(DEFAULT) if False else "", help="comma list; default = the Report 2 method set")
    ap.add_argument("--tag", default="", help="suffix for the output CSV name")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for layout in a.layouts.split(","):
        for i in range(a.worlds):
            for m in (a.methods.split(",") if a.methods else DEFAULT):
                if m.startswith("P-RRT") and i >= a.rrt_worlds:
                    continue
                jobs.append((layout, a.seed0 + i, m, a.sensor))
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.calib, a.floor, a.rho_calib)) as pool:
        rows = []
        for k, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            rows.append(row)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(jobs)} missions, {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["layout"], r["world"], r["method"]))
    path = os.path.join(a.out, f"missions_{a.sensor}{a.tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump(vars(a), open(os.path.join(a.out, f"run_config_{a.sensor}.json"), "w"), indent=1)
    print("wrote", path, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
