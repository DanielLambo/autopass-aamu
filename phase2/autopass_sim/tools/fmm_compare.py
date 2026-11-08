"""FMM vs A* outcomes for the same policies on the same episodes (simulated, no retraining).

A* reference = the committed evaluations: results_rl/v2/eval (rule, v2-A, v2-B, v1 shield-trained) and
results_rl/eval (v1 unshielded-trained; its "eval_shield" is the post-hoc shield, named eval_postshield here).
Pairs every episode (setting, policy, seed) and reports, per policy group, outcome rates under both planners,
the paired success test (exact McNemar per training seed) and the paired time difference.

    python tools/fmm_compare.py --fmm results_rl/fmm/eval --out results_rl/fmm
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

SETTINGS = ("main", "uniform", "budget20", "budget12")
GROUPS = [  # (label, regex on policy name)
    ("rule", r"^rule$"),
    ("rule + shield", r"^rule\+shield$"),
    ("v2-A, mask shield", r"^A_mask_budget_s\d\|eval_shield$"),
    ("v2-A, no shield", r"^A_mask_budget_s\d\|eval_noshield$"),
    ("v2-B, no shield", r"^B_noshield_budget_s\d\|eval_noshield$"),
    ("v2-B, mask shield", r"^B_noshield_budget_s\d\|eval_shield$"),
    ("v1 (train shield), post shield", r"^v1_ppolag_shield_s\d\|eval_postshield$"),
    ("v1, no shield", r"^ppolag_s\d\|eval_noshield$"),
    ("v1, post shield", r"^ppolag_s\d\|eval_postshield$"),
]


def reference(setting):
    v2 = pd.read_csv(f"results_rl/v2/eval/episodes_{setting}.csv")
    v1 = pd.read_csv(f"results_rl/eval/episodes_{setting}.csv")
    v1 = v1[v1.policy.str.match(r"ppolag_s\d")].copy()
    v1["policy"] = v1.policy.str.replace("|eval_shield", "|eval_postshield", regex=False)
    return pd.concat([v2, v1], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmm", default="results_rl/fmm/eval")
    ap.add_argument("--out", default="results_rl/fmm")
    a = ap.parse_args()
    seed_rows, rows = [], []
    for setting in SETTINGS:
        f = pd.read_csv(os.path.join(a.fmm, f"episodes_{setting}.csv"))
        m = f.merge(reference(setting), on=["policy", "seed"], suffixes=("_fmm", "_astar"))
        assert len(m) == len(f), (setting, len(m), len(f))
        for label, rx in GROUPS:
            g = m[m.policy.str.match(rx)]
            if g.empty:
                continue
            r = dict(setting=setting, policy=label, runs=g.policy.nunique(), episodes=len(g))
            for o in ("success", "unsafe", "flag", "timeout"):
                for pm in ("astar", "fmm"):
                    r[f"{o} % {pm}"] = round(100 * (g[f"outcome_{pm}"] == o).mean(), 2)
            for pm in ("astar", "fmm"):
                r[f"time s {pm}"] = round(g[f"time_s_{pm}"].mean(), 2)
                r[f"path m {pm}"] = round(g[f"path_m_{pm}"].mean(), 2)
            r["Δtime s (fmm−astar)"] = round((g.time_s_fmm - g.time_s_astar).mean(), 2)
            r["same outcome %"] = round(100 * (g.outcome_fmm == g.outcome_astar).mean(), 1)
            ps = []
            for pol, h in g.groupby("policy"):
                sf, sa = h.outcome_fmm.eq("success").values, h.outcome_astar.eq("success").values
                b, c = int((sf & ~sa).sum()), int((~sf & sa).sum())
                p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
                d = h.time_s_fmm.values - h.time_s_astar.values
                pt = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                ps.append(p)
                seed_rows.append(dict(setting=setting, group=label, policy=pol, b_only_fmm=b, c_only_astar=c,
                                      mcnemar_p=float(f"{p:.3g}"), success_fmm=round(100 * sf.mean(), 1),
                                      success_astar=round(100 * sa.mean(), 1),
                                      unsafe_fmm=int(h.outcome_fmm.eq("unsafe").sum()),
                                      unsafe_astar=int(h.outcome_astar.eq("unsafe").sum()),
                                      timeout_fmm=int(h.outcome_fmm.eq("timeout").sum()),
                                      timeout_astar=int(h.outcome_astar.eq("timeout").sum()),
                                      dtime_s=round(d.mean(), 2), wilcoxon_time_p=float(f"{pt:.3g}")))
            r["min McNemar p (success)"] = float(f"{min(ps):.3g}")
            rows.append(r)
    t, s = pd.DataFrame(rows), pd.DataFrame(seed_rows)
    t.to_csv(os.path.join(a.out, "fmm_vs_astar.csv"), index=False)
    s.to_csv(os.path.join(a.out, "fmm_vs_astar_per_seed.csv"), index=False)
    cols = ["setting", "policy", "success % astar", "success % fmm", "unsafe % astar", "unsafe % fmm",
            "timeout % astar", "timeout % fmm", "time s astar", "time s fmm", "Δtime s (fmm−astar)",
            "same outcome %", "min McNemar p (success)"]
    md = ["# FMM vs A* paths: same policies, same episodes (simulated, no retraining)", "",
          "Episode seeds 5000-5599 per setting. A* rows are the committed evaluations (reproduced exactly by the "
          "current code). PPO rows pool 3 training seeds (1800 episodes). McNemar: exact test on success, FMM vs A*, "
          "per training seed; the smallest p of the seeds is shown. Per-seed detail: fmm_vs_astar_per_seed.csv.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in t.iterrows()]
    open(os.path.join(a.out, "fmm_vs_astar.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    nr_compare(a.out)


def nr_compare(out):
    """No-repositioning worlds: RL-env rows under FMM vs the committed A* run (addendum)."""
    fa = pd.read_csv("results_rl/v2/addendum/nr_episodes.csv")
    ff = pd.read_csv(os.path.join(out, "nr", "nr_episodes.csv"))
    fa, ff = fa[fa.source == "RL env"], ff[ff.source == "RL env"]
    m = ff.merge(fa, on=["policy", "layout", "world"], suffixes=("_fmm", "_astar"))
    rows = []
    for pol, g in m.groupby("policy", sort=False):
        r = {"policy": pol, "worlds": len(g)}
        for o, names in (("success", ["success"]), ("unsafe", ["unsafe"]), ("flag / no route", ["flag", "no_route"]),
                         ("timeout", ["timeout"])):
            r[f"{o} astar → fmm"] = f"{int(g.outcome_astar.isin(names).sum())} → {int(g.outcome_fmm.isin(names).sum())}"
        r["time s astar → fmm"] = f"{g.time_astar.mean():.2f} → {g.time_fmm.mean():.2f}"
        r["same outcome %"] = round(100 * (g.outcome_fmm == g.outcome_astar).mean(), 1)
        rows.append(r)
    t = pd.DataFrame(rows)
    cols = list(t.columns)
    md = ["# No-repositioning worlds: FMM vs A* in the RL env (simulated)", "",
          "Report 2's 600 worlds (room + corridor, world seeds 1000-1299), w_A = 81.3 cm. Counts out of 600. "
          "A* = results_rl/v2/addendum/nr_episodes.csv; FMM = nr/nr_episodes.csv. Report 2 planner rows are "
          "unchanged (they do not use the RL env) and are in nr/nr_table.md.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    open(os.path.join(out, "nr", "nr_fmm_vs_astar.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
