"""A* vs FMM path quality on the same motion queries (simulated).

Replays the main-setting episodes in the default A* mode (rule + v2-A seeds 0-2, mask shield, seeds
5000-5599) and records every executed motion: GOTO (robot position -> viewpoint, start-side grid) and
COMMIT (robot position -> goal through the chosen door, other doors blocked). Each query is planned three ways
on the same inflated grid:
    astar_raw   8-connected A* cell path (no shortcutting)
    astar       8-connected A* + line-of-sight shortcut (what the env executes in astar mode)
    fmm         fast-marching travel-time field + gradient descent (path_mode="fmm")
Metrics: length; excess of each A* variant over FMM (%); heading changes > 20 deg between consecutive
0.25 m segments (the scan spacing); for COMMIT paths, wall-crossing offset from the door centre > 0.15 m.

    python tools/path_quality.py --out results_rl/fmm/path_quality
"""
import argparse, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch

from autopass_rl.env import AutoPassEnv, K
from autopass_rl.evaluate import load_agent
from autopass_rl.policies import RulePolicy
from autopass_rl import fmm
from autopass_sim.planners import astar, path_length, resample

TURN_DEG, OFFSET_M = 20.0, 0.15


def heading_changes(path):
    if path is None or len(path) < 2 or path_length(path) < 1e-6:
        return 0
    pts, _ = resample(np.asarray(path, float), 0.25)
    d = np.diff(pts, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-6]
    if len(d) < 2:
        return 0
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    return int((dh > TURN_DEG).sum())


def crossing_offset(path, door):
    """Signed x offset from the door centre where the path crosses the wall's robot-side face."""
    y0 = door.yw
    for a, b in zip(path[:-1], path[1:]):
        if (a[1] - y0) * (b[1] - y0) <= 0 and a[1] != b[1]:
            t = (y0 - a[1]) / (b[1] - a[1])
            return float(a[0] + t * (b[0] - a[0]) - door.xc)
    return np.nan


def queries(args):
    pol, agent_path, seeds = args
    torch.set_num_threads(1)
    env = AutoPassEnv(shield=agent_path is not None, shield_mode="mask")
    agent = load_agent(agent_path) if agent_path else None
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        rule = RulePolicy(); rule._dwell = 0
        fc = fmm.FieldCache(env.world.res)
        while True:
            if agent is None:
                a = rule(env, info)
            else:
                with torch.no_grad():
                    lg = agent.pi(torch.as_tensor(obs[None, :agent.n_obs]))[0]
                    a = int(lg.masked_fill(~torch.as_tensor(info["action_mask"]), -torch.inf).argmax())
            q = None
            if a < env.A_DWELL:
                i, k = divmod(a, K)
                if env.vp[i, k] is not None:
                    q = ("goto", i, env.pos.copy(), env.vp[i, k].copy(), env.free_near, ("vp", i, k))
            elif env.A_COMMIT <= a < env.A_REJECT:
                i = a - env.A_COMMIT
                q = ("commit", i, env.pos.copy(), env.world.goal.copy(), env.field[i][1], ("goal", i))
            if q is not None:
                kind, i, s, t, free, key = q
                ones = np.ones_like(env.world.free, float)
                P = {"astar_raw": astar(env.world, s, t, ones, free=free, smooth=False),
                     "astar": astar(env.world, s, t, ones, free=free),
                     "fmm": fc.path(key, free, s, t)}
                if all(p is not None for p in P.values()) and path_length(P["astar"]) > 1e-6:
                    r = dict(policy=pol, seed=seed, kind=kind, door=i)
                    for m, p in P.items():
                        r[f"len_{m}"] = path_length(p)
                        r[f"turns_{m}"] = heading_changes(p)
                        if kind == "commit":
                            r[f"offset_{m}"] = crossing_offset(p, env.doors[i])
                    rows.append(r)
            obs, rew, term, trunc, info = env.step(a)
            if term or trunc:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_rl/fmm/path_quality")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(5000, 5000 + a.episodes))
    chunks = [seeds[i::20] for i in range(20)]
    pols = [("rule", None)] + [(f"v2-A s{s}", f"results_rl/v2/A_mask_budget_s{s}/agent.pt") for s in range(3)]
    with Pool(a.procs) as pool:
        df = pd.DataFrame([r for rs in pool.map(queries, [(p, ap_, c) for p, ap_ in pols for c in chunks]) for r in rs])
    df.to_csv(os.path.join(a.out, "path_queries_main.csv"), index=False)

    out = []
    for (kind), g in [("all", df), ("goto", df[df.kind == "goto"]), ("commit", df[df.kind == "commit"])]:
        row = {"queries": kind, "n": len(g)}
        for m in ("astar_raw", "astar", "fmm"):
            row[f"mean length {m} (m)"] = round(g[f"len_{m}"].mean(), 3)
        for m in ("astar_raw", "astar"):
            ex = 100 * (g[f"len_{m}"] / g.len_fmm - 1)
            row[f"{m} over fmm: mean %"] = round(ex.mean(), 2)
            row[f"{m} over fmm: total %"] = round(100 * (g[f"len_{m}"].sum() / g.len_fmm.sum() - 1), 2)
        for m in ("astar_raw", "astar", "fmm"):
            row[f"turns>20° per path {m}"] = round(g[f"turns_{m}"].mean(), 2)
        if kind == "commit":
            for m in ("astar_raw", "astar", "fmm"):
                off = g[f"offset_{m}"].abs()
                row[f"door offset>0.15 m {m} (count)"] = int((off > OFFSET_M).sum())
                row[f"mean |door offset| {m} (m)"] = round(off.mean(), 3)
        out.append(row)
    tab = pd.DataFrame(out).set_index("queries").T
    tab.to_csv(os.path.join(a.out, "path_quality_main.csv"))
    md = ["# A* vs FMM path quality, main setting (simulated)", "",
          "Same motion queries: every GOTO and COMMIT executed in the A*-mode episodes of the rule and v2-A seeds 0-2 "
          "(mask shield), episode seeds 5000-5599. Each query is planned on the same inflated grid three ways. "
          "astar = what the env executes in astar mode (8-connected A* + line-of-sight shortcut); astar_raw = the "
          "8-connected cell path before shortcutting. Turns: heading changes > 20° between consecutive 0.25 m "
          "segments. Door offset: where a COMMIT path crosses the wall face, distance from the door centre.", "",
          "| metric | " + " | ".join(tab.columns) + " |", "|---|" + "---|" * len(tab.columns)]
    fmt = lambda v: "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() and abs(v) >= 1 else str(v))
    md += ["| " + idx + " | " + " | ".join(fmt(v) for v in r) + " |" for idx, r in tab.iterrows()]
    open(os.path.join(a.out, "path_quality_main.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
