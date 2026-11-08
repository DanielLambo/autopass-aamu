"""v2-A and the rule on Report 2's no-repositioning worlds, next to Report 2's planners (simulated).

Worlds: Report 2's harder-scenario set, room + corridor x world seeds 1000-1299 (600 worlds), wheelchair
w_A = 81.3 cm, built with the same make_world(default_rng(seed), layout) call. Same sensor model,
calibration and z-test rule in both simulators.

RL env, no repositioning: GOTO is limited to viewpoint 0 (square-on commit point, 1.2 m), so the agent
cannot move to a better view; it can still choose door order, dwell, COMMIT (mask shield) and REJECT.
v2-A is also run with all 5 viewpoints for context. Report 2 rows come from autopass_sim.experiments
(mission executor: drive the planned path, decide 1.2 m before the door, dwell <= 20, replan).

    python tools/nr_compare.py --out results_rl/v2/addendum
"""
import argparse, json, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch

from autopass_rl.env import AutoPassEnv
from autopass_rl.evaluate import load_agent
from autopass_rl.policies import RulePolicy

WORLDS = [(lay, s) for lay in ("corridor", "room") for s in range(1000, 1300)]
W_A = 81.3


def run(args):
    name, agent_path, vps, worlds, path_mode = args
    torch.set_num_threads(1)
    env = AutoPassEnv(shield=agent_path is not None, shield_mode="mask", viewpoints=vps, path_mode=path_mode)
    agent = load_agent(agent_path) if agent_path else None
    rows = []
    for lay, ws in worlds:
        obs, info = env.reset(seed=ws, options=dict(world_seed=ws, layout=lay, w_A=W_A))
        rule = RulePolicy(); rule._dwell = 0
        while True:
            if agent is None:
                a = rule(env, info)
            else:
                with torch.no_grad():
                    lg = agent.pi(torch.as_tensor(obs[None, :agent.n_obs]))[0]
                    a = int(lg.masked_fill(~torch.as_tensor(info["action_mask"]), -torch.inf).argmax())
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rows.append(dict(source="RL env", policy=name, layout=lay, world=ws, outcome=info["outcome"],
                         time=info["sim_time"], path_len=info["path_len"], scans=info["scans"]))
    return rows


def md_table(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def summarize(df, ref_name):
    rng = np.random.default_rng(0)
    ref = df[df.policy == ref_name].set_index(["layout", "world"]).sort_index()
    out = []
    for (src, pol), g in df.groupby(["source", "policy"], sort=False):
        g = g.set_index(["layout", "world"]).sort_index()
        n = len(g); t = g.time.values
        bs = t[rng.integers(0, n, (2000, n))].mean(1)
        d = t - ref.time.values
        dbs = d[rng.integers(0, n, (2000, n))].mean(1)
        out.append({"source": src, "policy": pol, "n": n,
                    "success %": round(100 * (g.outcome == "success").mean(), 1),
                    "unsafe %": round(100 * (g.outcome == "unsafe").mean(), 1),
                    "flag / no route %": round(100 * g.outcome.isin(["flag", "no_route"]).mean(), 1),
                    "timeout %": round(100 * (g.outcome == "timeout").mean(), 1),
                    "time_s": f"{t.mean():.1f} [{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}]",
                    "time_s (successes)": round(g.time[g.outcome == "success"].mean(), 1),
                    f"Δtime vs {ref_name}": "—" if pol == ref_name else
                    f"{d.mean():+.1f} [{np.percentile(dbs, 2.5):+.1f}, {np.percentile(dbs, 97.5):+.1f}]"})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_rl/v2/addendum")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="re-tabulate saved episodes without rerunning")
    ap.add_argument("--path-mode", default="astar", choices=["astar", "fmm"], help="RL-env path planner")
    ap.add_argument("--report2-dir", default=None, help="reuse a Report 2 rerun folder instead of rerunning it")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Report 2 planners on the same worlds (same code as phase2-sim), checked against results_rev
    r2dir = a.report2_dir or os.path.join(a.out, "report2_rerun")
    if not a.reuse and not a.report2_dir:
        subprocess.run([sys.executable, "-m", "autopass_sim.experiments", "--worlds", "300", "--rrt-worlds", "0",
                        "--methods", "B2-nr,P-A*(4)-nr,P-A*(4)", "--procs", str(a.procs), "--out", r2dir,
                        "--tag", "_nr"], check=True, stdout=subprocess.DEVNULL)
    r2 = pd.read_csv(os.path.join(r2dir, "missions_raw_nr.csv"))
    rev = pd.concat([pd.read_csv("results_rev/missions_raw_scen.csv"), pd.read_csv("results_rev/missions_raw.csv")])
    key = ["layout", "world", "method"]
    m = r2.merge(rev[key + ["outcome", "time"]].drop_duplicates(key), on=key, suffixes=("", "_rev"))
    repro = bool(len(m) == len(r2) and (m.outcome == m.outcome_rev).all() and np.allclose(m.time, m.time_rev))
    print(f"Report 2 rerun matches results_rev on {len(m)}/{len(r2)} missions: {repro}")
    r2 = r2.assign(source="Report 2 planner", policy=r2.method)[
        ["source", "policy", "layout", "world", "outcome", "time", "path_len"]]

    if a.reuse:
        df = pd.read_csv(os.path.join(a.out, "nr_episodes.csv"))
    else:
        chunks = [WORLDS[i::a.procs * 2] for i in range(a.procs * 2)]
        agent = "results_rl/v2/A_mask_budget_s{}/agent.pt"
        pols = [("rule, no reposition (vp 0)", None, (0,))] + \
               [(f"v2-A s{s}, no reposition (vp 0)", agent.format(s), (0,)) for s in range(3)] + \
               [(f"v2-A s{s}, all 5 viewpoints", agent.format(s), None) for s in range(3)]
        jobs = [(name, p, vps, c, a.path_mode) for name, p, vps in pols for c in chunks]
        with Pool(a.procs) as pool:
            env_rows = [r for rs in pool.map(run, jobs) for r in rs]
        df = pd.concat([pd.DataFrame(env_rows), r2], ignore_index=True)
        df.to_csv(os.path.join(a.out, "nr_episodes.csv"), index=False)
    tab = summarize(df, "rule, no reposition (vp 0)")
    tab.to_csv(os.path.join(a.out, "nr_table.csv"), index=False)
    md = ["# v2-A and rule on Report 2's no-repositioning worlds (simulated)", "",
          "600 worlds (room + corridor, world seeds 1000-1299), wheelchair w_A = 81.3 cm. Mask shield on for v2-A. "
          "Time: mean [95% bootstrap CI] over all 600 missions; Δtime paired per world vs the RL-env rule. "
          "RL env and Report 2 use the same worlds, sensor model and z-test, but different executors "
          "(discrete viewpoints vs planned path with a 1.2 m commit point), so cross-source time differences "
          "are indicative only.", "",
          f"Report 2 rerun reproduces results_rev exactly: {repro}.", "",
          md_table(tab), ""]
    open(os.path.join(a.out, "nr_table.md"), "w").write("\n".join(md))
    print("\n".join(md))
    json.dump(dict(worlds="room+corridor 1000-1299", w_A=W_A, report2_repro=repro, path_mode=a.path_mode), open(os.path.join(a.out, "nr_args.json"), "w"))


if __name__ == "__main__":
    main()
