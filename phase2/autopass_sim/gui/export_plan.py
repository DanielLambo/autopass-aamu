"""Export one RL-env world + v2-A's greedy action sequence as a waypoint plan for the Isaac Lab GUI run.

Runs v2-A (mask shield) deterministically on the given main-setting seed (same as the 2D replay) and records,
for every step, the exact A* path the env walks (GOTO), dwell time (0.5 s per DWELL sweep), REJECT events and the
COMMIT path to the goal. Coordinates are the env's world frame (m): x along the door wall, y away from start.

    python gui/export_plan.py --seed 5081 --out results_rl/gui/plan_5081.json
"""
import argparse, json
import numpy as np
import torch

from autopass_sim.world import FRAME_DEPTH, DOOR_HEIGHT
from autopass_rl.env import AutoPassEnv, K
from autopass_rl.evaluate import load_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5081)
    ap.add_argument("--agent", default="results_rl/v2/A_mask_budget_s0/agent.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = AutoPassEnv(shield=True, shield_mode="mask")
    obs, info = env.reset(seed=a.seed)
    agent = load_agent(a.agent)
    w = env.world
    plan, pos = [], env.pos.copy()
    while True:
        with torch.no_grad():
            lg = agent.pi(torch.as_tensor(obs[None, :agent.n_obs]))[0]
            act = int(lg.masked_fill(~torch.as_tensor(info["action_mask"]), -torch.inf).argmax())
        t0, p0 = env.sim_time, env.pos.copy()
        if act < env.A_DWELL:
            i, k = divmod(act, K)
            key = (round(float(p0[0]), 2), round(float(p0[1]), 2), i, k)
        elif env.A_COMMIT <= act < env.A_REJECT:
            i = act - env.A_COMMIT
            cpath, L = env._commit_path(i)
            m, s, b, n = env._fused(env.doors[i])
            zt = env.ztest(i)
        obs, r, term, trunc, info = env.step(act)
        if act < env.A_DWELL:
            path = env.goto_cache.get(key)
            if path is not None and len(path) > 1 and np.linalg.norm(env.pos - p0) > 1e-6:
                path = path.copy(); path[0] = p0
                plan.append(dict(kind="walk", door=int(env.doors[i].id), viewpoint=int(k),
                                 pts=np.round(path, 4).tolist(), t_env=round(env.sim_time - t0, 3)))
            else:
                plan.append(dict(kind="dwell", door=None, seconds=round(env.sim_time - t0, 3), sweeps=1))
        elif act == env.A_DWELL:
            if plan and plan[-1]["kind"] == "dwell":
                plan[-1]["seconds"] = round(plan[-1]["seconds"] + env.sim_time - t0, 3); plan[-1]["sweeps"] += 1
            else:
                plan.append(dict(kind="dwell", seconds=round(env.sim_time - t0, 3), sweeps=1))
        elif act < env.A_REJECT:
            plan.append(dict(kind="commit", door=int(env.doors[i].id), ztest=zt, fused_margin_cm=round(m, 2),
                             sigma_cm=round(s, 2), n_trusted=int(n), pts=np.round(np.asarray(cpath), 4).tolist(),
                             t_env=round(env.sim_time - t0, 3)))
        else:
            plan.append(dict(kind="reject", door=int(env.doors[act - env.A_REJECT].id)))
        if term or trunc:
            break
    out = dict(
        label="simulated", source="AutoPass RL env (Report 3), main setting, v2-A training seed 0, mask shield",
        episode_seed=a.seed, agent=a.agent, outcome=info["outcome"], env_time_s=round(info["sim_time"], 2),
        env_sweeps=info["scans"], env_path_m=round(info["path_len"], 3), w_A_cm=env.w_A,
        world=dict(layout=w.layout, W=w.W, H=w.H, wall_y=w.wall_y, wall_depth=FRAME_DEPTH, door_height=DOOR_HEIGHT,
                   start=w.start.tolist(), goal=w.goal.tolist(), boxes=[list(map(float, b)) for b in w.boxes],
                   doors=[dict(id=d.id, xc=d.xc, yw=d.yw, w=d.w, s=d.s, r=d.r, D=d.D, hf=d.hf,
                               margin_cm=round(d.w * 100 - env.w_A, 2)) for d in w.doors]),
        plan=plan)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("plan",)}, indent=1))
    for st in plan:
        print({k: (f"{len(v)} pts {v[0]}->{v[-1]}" if k == "pts" else v) for k, v in st.items()})


if __name__ == "__main__":
    main()
