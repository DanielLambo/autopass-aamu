"""Env sanity check: random vs rule policy, shielded and unshielded, plus a determinism check.

Run (TensorBook, from phase2/autopass_sim):
    python tools/sanity_env.py --episodes 200 --out results_rl/v1/sanity.json
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from autopass_rl.env import AutoPassEnv
from autopass_rl.policies import RulePolicy, random_policy, rollout

SETTINGS = {
    "random":          (dict(shield=False), "random"),
    "random+shield":   (dict(shield=True), "random"),
    "rule":            (dict(shield=False), "rule"),
    "rule+shield":     (dict(shield=True), "rule"),
}


def _job(args):
    name, seed = args
    kw, pol = SETTINGS[name]
    env = AutoPassEnv(**kw)
    r = rollout(env, RulePolicy() if pol == "rule" else random_policy, seed=seed)
    r.pop("final_obs")
    traj = r.pop("traj")
    r["traj_hash"] = hashlib.md5(json.dumps(traj).encode()).hexdigest()
    return name, seed, r


def summarize(rows):
    out = {}
    for name in SETTINGS:
        rs = [r for n, _, r in rows if n == name]
        if not rs:
            continue
        oc = [r["outcome"] for r in rs]
        out[name] = dict(n=len(rs),
                         success=oc.count("success"), unsafe=oc.count("unsafe"),
                         flag=oc.count("flag"), timeout=oc.count("timeout"),
                         mean_return=float(np.mean([r["ret"] for r in rs])),
                         mean_cost=float(np.mean([r["cost"] for r in rs])),
                         mean_path_m=float(np.mean([r["path_len"] for r in rs])),
                         mean_time_s=float(np.mean([r["sim_time"] for r in rs])),
                         mean_scans=float(np.mean([r["scans"] for r in rs])),
                         mean_steps=float(np.mean([r["steps"] for r in rs])),
                         shield_blocks=int(sum(r["shield_blocks"] for r in rs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results_rl/v1/sanity.json")
    a = ap.parse_args()
    jobs = [(n, a.seed0 + i) for n in SETTINGS for i in range(a.episodes)]
    t0 = time.time()
    with Pool(a.procs) as p:
        rows = p.map(_job, jobs, chunksize=2)
        rows2 = p.map(_job, jobs[: min(40, len(jobs))], chunksize=2)      # determinism re-run
    h1 = {(n, s): r["traj_hash"] for n, s, r in rows}
    repeat_ok = all(h1[(n, s)] == r["traj_hash"] for n, s, r in rows2)
    summ = summarize(rows)
    print(f"{'setting':15s} {'n':>4s} {'succ':>5s} {'unsafe':>6s} {'flag':>5s} {'tmo':>4s} "
          f"{'return':>7s} {'path_m':>7s} {'time_s':>7s} {'scans':>6s} {'shield':>6s}")
    for k, v in summ.items():
        print(f"{k:15s} {v['n']:4d} {v['success']:5d} {v['unsafe']:6d} {v['flag']:5d} {v['timeout']:4d} "
              f"{v['mean_return']:7.2f} {v['mean_path_m']:7.2f} {v['mean_time_s']:7.1f} "
              f"{v['mean_scans']:6.1f} {v['shield_blocks']:6d}")
    print("seeded re-run identical:", repeat_ok, f"({len(rows2)} episodes)  wall {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(summary=summ, repeat_ok=repeat_ok, args=vars(a)), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
