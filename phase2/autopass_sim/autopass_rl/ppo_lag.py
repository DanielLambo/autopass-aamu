"""Masked PPO-Lagrangian for AutoPassEnv (single file, CleanRL style).

Why not OmniSafe: OmniSafe's on-policy algorithms assume Box actions; discrete support is
an open, unmerged PR (PKU-Alignment/omnisafe#286), and it has no action masking.

CMDP:  max J_r(pi)  s.t.  J_c(pi) <= d,   J_c = E[# unsafe commits per episode]
Lagrangian: policy uses A = (A_r - lam * A_c) / (1 + lam);  lam <- max(0, lam + lr_lam (J_c - d))
This bounds the EXPECTED cost only. Use --shield to make COMMIT require a z-test PASS.

Run (TensorBook, from phase2/autopass_sim):
    python -m autopass_rl.ppo_lag --total-steps 500000 --envs 18 --out results_rl/ppolag_s0 --seed 0
    python -m autopass_rl.ppo_lag --total-steps 500000 --envs 18 --out results_rl/ppolag_shield_s0 --seed 0 --shield
"""
import os

# one BLAS/OpenMP thread per process: the env workers are separate processes, so extra threads oversubscribe
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, csv, json, time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------- vectorised env (subprocesses)
def _worker(remote, kw, seed):
    from autopass_rl.env import AutoPassEnv
    env = AutoPassEnv(**kw)
    ep = 0
    obs, info = env.reset(seed=seed)
    R = C = 0.0
    while True:
        cmd, a = remote.recv()
        if cmd == "step":
            obs, r, term, trunc, info = env.step(a)
            c = info["cost"]                      # read before a possible reset overwrites info
            R += r; C += c
            done, stats = term or trunc, None
            if done:
                stats = dict(ret=R, cost=C, outcome=info["outcome"], path=info["path_len"],
                             time=info["sim_time"], scans=info["scans"])
                ep += 1
                obs, info = env.reset(seed=seed + 100000 * ep)
                R = C = 0.0
            remote.send((obs, r, c, term, info["action_mask"], stats))
        elif cmd == "init":
            remote.send((obs, info["action_mask"]))
        elif cmd == "close":
            return


class VecEnv:
    def __init__(self, n, kw, seed):
        ctx = mp.get_context("fork")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, kw, seed * 1000 + i), daemon=True)
            p.start(); self.remotes.append(a); self.procs.append(p)

    def init(self):
        for r in self.remotes: r.send(("init", None))
        o, m = zip(*[r.recv() for r in self.remotes])
        return np.stack(o), np.stack(m)

    def step(self, acts):
        for r, a in zip(self.remotes, acts): r.send(("step", int(a)))
        o, rew, c, term, m, st = zip(*[r.recv() for r in self.remotes])
        return np.stack(o), np.array(rew), np.array(c, float), np.array(term), np.stack(m), st

    def close(self):
        for r in self.remotes: r.send(("close", None))


# ---------------------------------------------------------------- network
def layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o); nn.init.orthogonal_(l.weight, std); nn.init.zeros_(l.bias); return l


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, h=128):
        super().__init__()
        body = lambda: nn.Sequential(layer(n_obs, h), nn.Tanh(), layer(h, h), nn.Tanh())
        self.pi = nn.Sequential(body(), layer(h, n_act, 0.01))
        self.vr = nn.Sequential(body(), layer(h, 1, 1.0))
        self.vc = nn.Sequential(body(), layer(h, 1, 1.0))

    def dist(self, x, mask):
        logits = self.pi(x).masked_fill(~mask, -1e9)
        return Categorical(logits=logits)


# ---------------------------------------------------------------- training
def gae(rew, val, done_next, last_val, gamma, lam):
    T = len(rew); adv = np.zeros_like(rew); g = 0
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done_next[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--envs", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shield", action="store_true")
    ap.add_argument("--agents", default="profiles", choices=["profiles", "uniform"])
    ap.add_argument("--cost-limit", type=float, default=0.01, help="d: allowed expected unsafe commits/episode")
    ap.add_argument("--lam-lr", type=float, default=0.05)
    ap.add_argument("--lam-init", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--out", default="results_rl/ppolag")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(1)

    venv = VecEnv(a.envs, dict(shield=a.shield, agents=a.agents), a.seed)
    obs, mask = venv.init()
    agent = Agent(obs.shape[1], mask.shape[1])
    opt = torch.optim.Adam(agent.parameters(), lr=a.lr, eps=1e-5)
    lam = a.lam_init
    N, T = a.envs, a.horizon
    log = open(os.path.join(a.out, "train.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["iter", "steps", "episodes", "ret", "cost", "success", "unsafe", "flag", "timeout",
                 "path_m", "time_s", "scans", "lambda", "entropy", "wall_s"])
    t0, step, it = time.time(), 0, 0
    while step < a.total_steps:
        B = dict(o=np.zeros((T, N, obs.shape[1]), np.float32), m=np.zeros((T, N, mask.shape[1]), bool),
                 a=np.zeros((T, N), np.int64), lp=np.zeros((T, N), np.float32),
                 r=np.zeros((T, N), np.float32), c=np.zeros((T, N), np.float32),
                 d=np.zeros((T, N), np.float32), vr=np.zeros((T, N), np.float32), vc=np.zeros((T, N), np.float32))
        eps = []
        for t in range(T):
            with torch.no_grad():
                x, mk = torch.as_tensor(obs), torch.as_tensor(mask)
                dist = agent.dist(x, mk)
                act = dist.sample()
                B["lp"][t] = dist.log_prob(act).numpy()
                B["vr"][t] = agent.vr(x).squeeze(-1).numpy()
                B["vc"][t] = agent.vc(x).squeeze(-1).numpy()
            B["o"][t], B["m"][t], B["a"][t] = obs, mask, act.numpy()
            obs, r, c, term, mask, st = venv.step(act.numpy())
            done = np.array([s is not None for s in st])
            B["r"][t], B["c"][t], B["d"][t] = r, c, done
            eps += [s for s in st if s is not None]
        step += T * N; it += 1
        with torch.no_grad():
            x = torch.as_tensor(obs)
            lvr, lvc = agent.vr(x).squeeze(-1).numpy(), agent.vc(x).squeeze(-1).numpy()
        Ar, Rr = gae(B["r"], B["vr"], B["d"], lvr, a.gamma, a.gae)
        Ac, Rc = gae(B["c"], B["vc"], B["d"], lvc, a.gamma, a.gae)
        # Lagrange multiplier update from this batch's finished episodes
        Jc = float(np.mean([e["cost"] for e in eps])) if eps else 0.0
        lam = max(0.0, lam + a.lam_lr * (Jc - a.cost_limit))
        Ar = (Ar - Ar.mean()) / (Ar.std() + 1e-8)
        Ac = Ac - Ac.mean()
        A = (Ar - lam * Ac) / (1 + lam)
        f = lambda k: torch.as_tensor(B[k].reshape(T * N, *B[k].shape[2:]))
        o, m, ac, lp0 = f("o"), f("m"), f("a"), f("lp")
        At = torch.as_tensor(A.reshape(-1)); Rrt = torch.as_tensor(Rr.reshape(-1)); Rct = torch.as_tensor(Rc.reshape(-1))
        idx = np.arange(T * N); mb = T * N // a.minibatches
        for _ in range(a.epochs):
            np.random.shuffle(idx)
            for s in range(0, T * N, mb):
                j = idx[s:s + mb]
                dist = agent.dist(o[j], m[j])
                ratio = (dist.log_prob(ac[j]) - lp0[j]).exp()
                pg = -torch.min(ratio * At[j], ratio.clamp(1 - a.clip, 1 + a.clip) * At[j]).mean()
                v = ((agent.vr(o[j]).squeeze(-1) - Rrt[j]) ** 2).mean() + ((agent.vc(o[j]).squeeze(-1) - Rct[j]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + 0.5 * v - a.ent * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step()
        if eps:
            oc = [e["outcome"] for e in eps]
            row = [it, step, len(eps), np.mean([e["ret"] for e in eps]), Jc,
                   oc.count("success"), oc.count("unsafe"), oc.count("flag"), oc.count("timeout"),
                   np.mean([e["path"] for e in eps]), np.mean([e["time"] for e in eps]),
                   np.mean([e["scans"] for e in eps]), lam, float(ent.detach()), time.time() - t0]
            wr.writerow([round(v, 4) if isinstance(v, float) else v for v in row]); log.flush()
            print(f"it {it:4d} steps {step:8d} eps {len(eps):3d} ret {row[3]:6.2f} cost {Jc:.3f} "
                  f"succ {row[5]:3d} unsafe {row[6]:2d} flag {row[7]:2d} path {row[9]:5.1f} lam {lam:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        torch.save(agent.state_dict(), os.path.join(a.out, "agent.pt"))
    venv.close()


if __name__ == "__main__":
    main()
