"""Sweeps per door vs normalized margin |m_hat| / sigma_m at decision time: v2-A vs rule (simulated).

m_hat is the fused margin estimate (measured width - w_A, cm) and sigma_m its fused std, both over the
trusted sweeps the z-test uses. A "decision" is a COMMIT or REJECT of a door; the sweep count is the
number of trusted sweeps of that door fused at that moment. Doors decided with 0 trusted sweeps
(sigma = inf) are counted separately and left out of the bins.

    python tools/fig_margin.py --out results_rl/v2/addendum
"""
import argparse, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch

from autopass_rl.env import AutoPassEnv
from autopass_rl.evaluate import load_agent, SETTINGS
from autopass_rl.policies import RulePolicy

BINS = [0, 1, 2, 3, 4, 6, 10, np.inf]


def decisions(args):
    pol, agent_path, seeds, setting = args
    torch.set_num_threads(1)
    env = AutoPassEnv(shield=agent_path is not None, shield_mode="mask", **SETTINGS[setting])
    agent = load_agent(agent_path) if agent_path else None
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        rule = RulePolicy(); rule._dwell = 0
        while True:
            if agent is None:
                a = rule(env, info)
            else:
                with torch.no_grad():
                    lg = agent.pi(torch.as_tensor(obs[None, :agent.n_obs]))[0]
                    a = int(lg.masked_fill(~torch.as_tensor(info["action_mask"]), -torch.inf).argmax())
            if env.A_COMMIT <= a < env.A_REJECT + len(env.doors) and a != env.A_DWELL:
                kind = "commit" if a < env.A_REJECT else "reject"
                i = a - (env.A_COMMIT if kind == "commit" else env.A_REJECT)
                d = env.doors[i]
                m, s, b, n = env._fused(d)
                rows.append(dict(policy=pol, seed=seed, door=i, kind=kind, n_trusted=n,
                                 n_all=len(env.memory[d.id]), m=m, sigma=s, ztest=env.ztest(i),
                                 true_margin=d.w * 100 - env.w_A))
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_rl/v2/addendum")
    ap.add_argument("--setting", default="main")
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    chunks = [seeds[i::a.procs * 2] for i in range(a.procs * 2)]
    pols = [("rule", None)] + [(f"v2-A s{s}", f"results_rl/v2/A_mask_budget_s{s}/agent.pt") for s in range(3)]
    with Pool(a.procs) as pool:
        rows = [r for rs in pool.map(decisions, [(p, ap_, c, a.setting) for p, ap_ in pols for c in chunks]) for r in rs]
    df = pd.DataFrame(rows)
    df["zabs"] = np.where(df.n_trusted > 0, df.m.abs() / df.sigma.replace(np.inf, np.nan), np.nan)
    df.to_csv(os.path.join(a.out, f"margin_decisions_{a.setting}.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["0-1", "1-2", "2-3", "3-4", "4-6", "6-10", "≥10"]
    df["bin"] = pd.cut(df.zabs, BINS, right=False, labels=labels)
    df["group"] = np.where(df.policy == "rule", "Rule (z-test)", "PPO-Lag v2-A + mask shield (3 seeds)")
    tab = []
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    x = np.arange(len(labels))
    colors = {"Rule (z-test)": "#4C72B0", "PPO-Lag v2-A + mask shield (3 seeds)": "#DD8452"}
    rng = np.random.default_rng(0)
    for k, (g, col) in enumerate(colors.items()):
        sub = df[(df.group == g) & df.n_trusted.gt(0)]
        mean, lo, hi, cnt = [], [], [], []
        for lab in labels:
            v = sub[sub.bin == lab].n_trusted.values.astype(float)
            cnt.append(len(v))
            if len(v) == 0:
                mean.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
            mean.append(v.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
            tab.append(dict(group=g, bin=lab, decisions=len(v), mean_trusted_sweeps=round(v.mean(), 2),
                            ci_lo=round(np.percentile(bs, 2.5), 2), ci_hi=round(np.percentile(bs, 97.5), 2),
                            commit_frac=round((sub[sub.bin == lab].kind == "commit").mean(), 3)))
        mean, lo, hi = map(np.array, (mean, lo, hi))
        ax.errorbar(x + (k - 0.5) * 0.12, mean, yerr=[mean - lo, hi - mean], fmt="o-", color=col, capsize=3, label=g)
        n_runs = sub.policy.nunique()
        ax2.bar(x + (k - 0.5) * 0.38, np.array(cnt) / n_runs, width=0.38, color=col)
    ax.axvspan(1.5, 2.5, color="0.9", zorder=0)
    ax.text(2, ax.get_ylim()[1] * 0.97, "z-test boundary\n(2 + b/σ)", ha="center", va="top", fontsize=8, color="0.35")
    ax.set_ylabel("trusted sweeps of the door\nat decision (mean, 95% CI)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(f"Sweeps per door vs normalized margin at decision time ({a.setting} setting, simulated)", fontsize=11)
    ax2.set_ylabel("decisions per\n600 episodes")
    ax2.set_xticks(x, labels)
    ax2.set_xlabel("|m̂| / σ_m at decision   (m̂ = fused measured width − w_A)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    n0 = df[df.n_trusted == 0].groupby("group").size()
    fig.text(0.5, 0.005, f"Simulated. {a.episodes} episodes (seeds {a.seed0}-{a.seed0 + a.episodes - 1}); PPO pooled over "
             f"3 training seeds. Excluded (0 trusted sweeps): rule {n0.get('Rule (z-test)', 0)}, "
             f"PPO {n0.get('PPO-Lag v2-A + mask shield (3 seeds)', 0)}.", ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(a.out, "fig_margin.png"), dpi=150)
    pd.DataFrame(tab).to_csv(os.path.join(a.out, "fig_margin_table.csv"), index=False)
    print(pd.DataFrame(tab).to_string(index=False))
    print("0-trusted-sweep decisions:", n0.to_dict())


if __name__ == "__main__":
    main()
