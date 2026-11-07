"""Paired comparison v2-A-noWA vs v2-A, same training seed, same 600 episodes (simulated).

Reads the eval_noWA episodes and writes nowa_vs_v2A.md / .csv next to them.

    python tools/nowa_compare.py --dir results_rl/v2/addendum/eval_noWA
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rl/v2/addendum/eval_noWA")
    a = ap.parse_args()
    rows = []
    for setting in ("main", "uniform"):
        ep = pd.read_csv(os.path.join(a.dir, f"episodes_{setting}.csv"))
        for shield in (True, False):
            for s in range(3):
                g = {k: ep[ep.policy.str.startswith(f"{k}_s{s}|") & (ep.eval_shield == shield)].set_index("seed").sort_index()
                     for k in ("A_noWA", "A_mask_budget")}
                x, y = g["A_noWA"], g["A_mask_budget"]
                assert len(x) == len(y) == 600 and (x.index == y.index).all()
                sx, sy = x.outcome.eq("success").values, y.outcome.eq("success").values
                b, c = int((sx & ~sy).sum()), int((~sx & sy).sum())
                p_mc = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
                d = x.time_s.values - y.time_s.values
                p_t = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                rows.append({"setting": setting, "eval shield": shield, "seed": s,
                             "noWA success %": round(100 * sx.mean(), 1), "v2-A success %": round(100 * sy.mean(), 1),
                             "noWA unsafe %": round(100 * x.outcome.eq("unsafe").mean(), 1),
                             "v2-A unsafe %": round(100 * y.outcome.eq("unsafe").mean(), 1),
                             "noWA timeout %": round(100 * x.outcome.eq("timeout").mean(), 1),
                             "v2-A timeout %": round(100 * y.outcome.eq("timeout").mean(), 1),
                             "b (only noWA)": b, "c (only v2-A)": c, "McNemar p": float(f"{p_mc:.3g}"),
                             "Δtime_s noWA−v2-A": round(d.mean(), 2), "Wilcoxon p (time)": float(f"{p_t:.3g}"),
                             "same outcome %": round(100 * (x.outcome.values == y.outcome.values).mean(), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.dir, "nowa_vs_v2A.csv"), index=False)
    cols = list(df.columns)
    md = ["# v2-A-noWA vs v2-A, paired by training seed (simulated)", "",
          "600 episodes (seeds 5000-5599), deterministic. b/c = episodes only noWA / only v2-A succeeds; "
          "exact McNemar; Δtime paired per episode with Wilcoxon p. noWA zeroes the explicit w_A feature only; "
          "the per-door margin features (width − w_A) still carry w_A implicitly.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
         ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    open(os.path.join(a.dir, "nowa_vs_v2A.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
