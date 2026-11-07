"""Deterministic evaluation of trained PPO-Lagrangian agents vs the rule policy.

Every policy runs on the same fixed seeds (default 5000-5599), so differences vs the rule policy
are paired per seed. A trained agent acts greedily: argmax over its masked logits.

Settings:
    main       AutoPassEnv()                      (training distribution: wheelchair/walker widths)
    uniform    AutoPassEnv(agents="uniform")      (w_A ~ U(66, 92) cm, unseen widths)
    budget20   AutoPassEnv(scan_budget=20)        (stress: at most 20 sweeps, no retraining)
    budget12   AutoPassEnv(scan_budget=12)

Policies per setting: rule, rule+shield, and each trained run with the shield off and on at eval
time (so unshielded-trained agents are also tested with the shield added at deployment).

Run (TensorBook, from phase2/autopass_sim):
    python -m autopass_rl.evaluate --settings main uniform budget20 budget12
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, json, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, binomtest

from autopass_rl.env import AutoPassEnv
from autopass_rl.policies import RulePolicy
from autopass_rl.ppo_lag import Agent

SETTINGS = {
    "main": dict(),
    "uniform": dict(agents="uniform"),
    "budget20": dict(scan_budget=20),
    "budget12": dict(scan_budget=12),
}
# Policy sets. Each trained run: (run name, agent dir under results_rl, group label, eval variants).
# Eval variant tag -> (shield on, shield_mode). v1 used the post-hoc shield; v2 uses the action mask.
EVALS = {"eval_noshield": (False, "mask"), "eval_shield": (True, "mask"), "eval_postshield": (True, "post")}
EVAL_LABEL = {"eval_noshield": "eval no shield", "eval_shield": "eval shield", "eval_postshield": "eval shield"}
V1_EVALS = ("eval_noshield", "eval_postshield")
POLICY_SETS = {
    "v1": [(f"ppolag_s{s}", f"ppolag_s{s}", "PPO-Lag", V1_EVALS) for s in range(3)] +
          [(f"ppolag_shield_s{s}", f"ppolag_shield_s{s}", "PPO-Lag(train shield)", V1_EVALS) for s in range(3)],
    "v2": [(f"A_mask_budget_s{s}", f"v2/A_mask_budget_s{s}", "v2 PPO-Lag(mask shield + budget)",
            ("eval_noshield", "eval_shield")) for s in range(3)] +
          [(f"B_noshield_budget_s{s}", f"v2/B_noshield_budget_s{s}", "v2 PPO-Lag(no shield + budget)",
            ("eval_noshield", "eval_shield")) for s in range(3)] +
          [(f"v1_ppolag_shield_s{s}", f"ppolag_shield_s{s}", "v1 PPO-Lag(train shield, post shield)",
            ("eval_postshield",)) for s in range(3)],
}
GROUP = {}   # run name -> group label, filled from the chosen policy set
METRICS = ("path_m", "time_s", "scans")
OUTCOMES = ("success", "unsafe", "flag", "timeout")

_agents = {}


def load_agent(path):
    """Agent sized from its own weights; v1 agents (no budget features) see obs[:n_obs]."""
    if path not in _agents:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        n_obs, n_act = sd["pi.0.0.weight"].shape[1], sd["pi.1.weight"].shape[0]
        a = Agent(n_obs, n_act)
        a.load_state_dict(sd)
        a.eval()
        a.n_obs = n_obs
        _agents[path] = a
    return _agents[path]


def run_episode(env, policy, seed):
    """policy: RulePolicy instance, or an Agent (greedy over masked logits)."""
    obs, info = env.reset(seed=seed)
    if isinstance(policy, RulePolicy):
        policy._dwell = 0
    while True:
        if isinstance(policy, RulePolicy):
            a = policy(env, info)
        else:
            with torch.no_grad():
                logits = policy.pi(torch.as_tensor(obs[None, :policy.n_obs]))[0]
                logits = logits.masked_fill(~torch.as_tensor(info["action_mask"]), -torch.inf)
                a = int(torch.argmax(logits))
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            return dict(seed=seed, outcome=info["outcome"], path_m=info["path_len"],
                        time_s=info["sim_time"], scans=info["scans"], shield_blocks=info["shield_blocks"])


def _job(args):
    torch.set_num_threads(1)
    setting, pol, eval_shield, shield_mode, agent_path, seeds = args
    env = AutoPassEnv(shield=eval_shield, shield_mode=shield_mode, **SETTINGS[setting])
    policy = RulePolicy() if agent_path is None else load_agent(agent_path)
    return [dict(setting=setting, policy=pol, eval_shield=eval_shield, **run_episode(env, policy, s))
            for s in seeds]


def boot_ci(x, rng, n=2000):
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcox_p(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def summarize(ep, rng):
    """One row per (setting, policy): counts, mean + 95% bootstrap CI, paired diff vs rule."""
    rows = []
    for (setting, pol), g in ep.groupby(["setting", "policy"], sort=False):
        g = g.sort_values("seed")
        ref = ep[(ep.setting == setting) & (ep.policy == "rule")].sort_values("seed")
        assert list(g.seed) == list(ref.seed)
        row = dict(setting=setting, policy=pol, kind="run", n=len(g))
        for o in OUTCOMES:
            row[o] = int((g.outcome == o).sum())
        for o in ("success", "unsafe"):
            d = (g.outcome == o).astype(float).values - (ref.outcome == o).astype(float).values
            row[f"{o}_rate_diff"] = float(d.mean())
            row[f"{o}_rate_diff_lo"], row[f"{o}_rate_diff_hi"] = boot_ci(d, rng)
        for m in METRICS:
            x = g[m].values
            row[m] = float(x.mean())
            row[f"{m}_lo"], row[f"{m}_hi"] = boot_ci(x, rng)
            d = x - ref[m].values
            row[f"{m}_diff"] = float(d.mean())
            row[f"{m}_diff_lo"], row[f"{m}_diff_hi"] = boot_ci(d, rng)
            row[f"{m}_diff_p"] = wilcox_p(d)
        row["shield_blocks"] = int(g.shield_blocks.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def group_name(pol):
    """'ppolag_shield_s1|eval_postshield' -> 'PPO-Lag(train shield), eval shield'."""
    run, ev = pol.split("|")
    return f"{GROUP[run]}, {EVAL_LABEL[ev]}"


def aggregate(runs):
    """Mean +- std across the 3 training seeds for each trained-policy group."""
    out = []
    tr = runs[runs.policy.str.contains(r"\|")].copy()
    tr["group"] = tr.policy.map(group_name)
    num = [c for c in runs.columns if c not in ("setting", "policy", "kind", "group")]
    for (setting, grp), g in tr.groupby(["setting", "group"], sort=False):
        row = dict(setting=setting, policy=grp, kind="mean_over_seeds", n_seeds=len(g))
        for c in num:
            if c.endswith("_p"):
                row[c] = float(g[c].max())            # conservative: largest p across seeds
            else:
                row[c] = float(g[c].mean())
                row[f"{c}_sd"] = float(g[c].std(ddof=1))
        out.append(row)
    return pd.DataFrame(out)


def fmt_md(setting, runs, agg):
    rule = runs[(runs.setting == setting) & runs.policy.isin(["rule", "rule+shield"])]
    ag = agg[agg.setting == setting]
    n = int(rule.n.iloc[0])
    pct = lambda v, s=None: f"{100 * v / n:.1f}" + (f" ± {100 * s / n:.1f}" if s is not None else "")
    lines = [f"### {setting} ({n} episodes per policy, seeds {args_seed0}-{args_seed0 + n - 1}, simulated)", "",
             "| policy | success % | unsafe % | flag % | timeout % | path_m | time_s | scans | Δtime_s vs rule | Δscans vs rule | p (time / scans) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in rule.iterrows():
        d = "—" if r.policy == "rule" else f"{r.time_s_diff:+.1f} [{r.time_s_diff_lo:+.1f}, {r.time_s_diff_hi:+.1f}]"
        ds = "—" if r.policy == "rule" else f"{r.scans_diff:+.1f} [{r.scans_diff_lo:+.1f}, {r.scans_diff_hi:+.1f}]"
        p = "—" if r.policy == "rule" else f"{r.time_s_diff_p:.2g} / {r.scans_diff_p:.2g}"
        lines.append(f"| {r.policy} | {pct(r.success)} | {pct(r.unsafe)} | {pct(r.flag)} | {pct(r.timeout)} | "
                     f"{r.path_m:.2f} [{r.path_m_lo:.2f}, {r.path_m_hi:.2f}] | {r.time_s:.1f} [{r.time_s_lo:.1f}, {r.time_s_hi:.1f}] | "
                     f"{r.scans:.1f} [{r.scans_lo:.1f}, {r.scans_hi:.1f}] | {d} | {ds} | {p} |")
    for _, r in ag.iterrows():
        lines.append(f"| {r.policy} | {pct(r.success, r.success_sd)} | {pct(r.unsafe, r.unsafe_sd)} | "
                     f"{pct(r.flag, r.flag_sd)} | {pct(r.timeout, r.timeout_sd)} | {r.path_m:.2f} ± {r.path_m_sd:.2f} | "
                     f"{r.time_s:.1f} ± {r.time_s_sd:.1f} | {r.scans:.1f} ± {r.scans_sd:.1f} | "
                     f"{r.time_s_diff:+.1f} ± {r.time_s_diff_sd:.1f} | {r.scans_diff:+.1f} ± {r.scans_diff_sd:.1f} | "
                     f"{r.time_s_diff_p:.2g} / {r.scans_diff_p:.2g} |")
    lines += ["", "Rule rows: mean [95% bootstrap CI] over episodes; paired Δ vs rule with bootstrap CI and Wilcoxon p. "
              "PPO-Lag rows: mean ± std across 3 training seeds (each seed evaluated deterministically on the same "
              "episodes); Δ = paired difference vs rule; p = largest Wilcoxon p across the 3 seeds.", ""]
    return "\n".join(lines)


def mcnemar_md(ep):
    """Paired exact McNemar test on success vs rule, per training seed (same episodes)."""
    ref = ep[ep.policy == "rule"].set_index("seed").outcome.eq("success").sort_index()
    lines = ["Success vs rule, paired exact McNemar per training seed (b = only policy succeeds, "
             "c = only rule succeeds):", "", "| policy | seed | success % | Δ pts vs rule | b | c | p |", "|---|---|---|---|---|---|---|"]
    for pol in [p for p in ep.policy.unique() if "|" in p]:
        x = ep[ep.policy == pol].set_index("seed").outcome.eq("success").sort_index()
        b, c = int((x & ~ref).sum()), int((~x & ref).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        lines.append(f"| {group_name(pol)} | {pol.split('|')[0].rsplit('_s', 1)[1]} | {100 * x.mean():.1f} | "
                     f"{100 * (x.mean() - ref.mean()):+.1f} | {b} | {c} | {p:.3g} |")
    return "\n".join(lines + [""])


args_seed0 = 5000


def main():
    global args_seed0
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--runs-dir", default="results_rl")
    ap.add_argument("--out", default="results_rl/eval")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--set", default="v1", choices=list(POLICY_SETS))
    a = ap.parse_args()
    GROUP.update({run: label for run, _, label, _ in POLICY_SETS[a.set]})
    args_seed0 = a.seed0
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    chunks = [seeds[i:i + 25] for i in range(0, len(seeds), 25)]
    for setting in a.settings:
        t0 = time.time()
        pols = [("rule", False, "mask", None), ("rule+shield", True, "mask", None)]
        for run, d, _, evals in POLICY_SETS[a.set]:
            p = os.path.join(a.runs_dir, d, "agent.pt")
            pols += [(f"{run}|{ev}", *EVALS[ev], p) for ev in evals]
        jobs = [(setting, name, sh, mode, p, c) for name, sh, mode, p in pols for c in chunks]
        with Pool(a.procs) as pool:
            ep = pd.DataFrame([r for rs in pool.map(_job, jobs, chunksize=1) for r in rs])
        ep.to_csv(os.path.join(a.out, f"episodes_{setting}.csv"), index=False)
        runs = summarize(ep, np.random.default_rng(0))
        agg = aggregate(runs)
        out = pd.concat([runs, agg], ignore_index=True)
        out.to_csv(os.path.join(a.out, f"eval_{setting}.csv"), index=False)
        block = fmt_md(setting, runs, agg) + mcnemar_md(ep)
        open(os.path.join(a.out, f"eval_{setting}.md"), "w").write("# Report 3 deterministic evaluation (simulated)\n\n" + block)
        print(block, flush=True)
        print(f"[{setting}] {len(ep)} episodes in {time.time() - t0:.0f}s", flush=True)
    json.dump(vars(a), open(os.path.join(a.out, "eval_args.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
