"""Paired statistics quoted in Report 2: McNemar exact tests, paired time differences, evidence O,
failure modes and RRT*/A* agreement. Writes <results>/report_numbers.json.

Run: python tools/paired_stats.py results_rev
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest, t as tdist


def paired_time(a, b):
    d = (b.time - a.time).dropna()
    m, s, n = d.mean(), d.std(ddof=1), len(d)
    h = tdist.ppf(0.975, n - 1) * s / np.sqrt(n)
    return [round(m, 2), round(m - h, 2), round(m + h, 2)]


def mcnemar(a_bad, b_bad):
    n10 = int((a_bad & ~b_bad).sum()); n01 = int((~a_bad & b_bad).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return dict(a_only=n10, b_only=n01, p=float(p))


def main(res):
    d = pd.read_csv(f"{res}/missions_raw.csv")
    d["unsafe"] = d.outcome == "unsafe"
    idx = ["layout", "world"]
    M = {m: g.set_index(idx).sort_index() for m, g in d.groupby("method")}
    out = dict(counts={}, compare={}, by_layout={}, failures={}, rrt={}, cpu={})
    for m, g in M.items():
        out["counts"][m] = dict(n=len(g), unsafe=int(g.unsafe.sum()), success=int((g.outcome == "success").sum()),
                                no_route=int((g.outcome == "no_route").sum()),
                                unsafe_by_layout={l: int(x.unsafe.sum()) for l, x in g.groupby(level=0)})
    for a, b in [("B1", "P-A*(4)n"), ("B1", "B2"), ("B2", "P-A*(4)")]:
        A, B = M[a], M[b].loc[M[a].index]
        out["compare"][f"{a}->{b}"] = dict(mcnemar(A.unsafe, B.unsafe), unsafe=[int(A.unsafe.sum()), int(B.unsafe.sum())],
                                            dtime=paired_time(A, B))
        out["by_layout"][f"{a}->{b}"] = {l: paired_time(A.loc[l], B.loc[l]) for l in ["corridor", "room"]}
    for m in ["B1", "B2", "P-A*(4)", "P-A*(4)n"]:
        out["by_layout"][m + "_time"] = {l: round(M[m].loc[l].time.mean(), 1) for l in ["corridor", "room"]}
    # evidence before first decision, error at decision, flag share (trusted methods)
    for m in ["B2", "P-A*(1)", "P-A*(2)", "P-A*(4)", "P-A*(8)", "P-RRT*(4)"]:
        g = M[m]
        dec = [json.loads(x) for x in g.decisions]
        flags = sum(1 for ds in dec for e in ds if e[2] == "FLAG"); n = sum(len(ds) for ds in dec)
        out.setdefault("evidence", {})[m] = {l: dict(O=round(float(x.first_O.mean()), 3),
                                                     err=round(float(x.first_est_err.abs().mean()), 3))
                                             for l, x in g.groupby(level=0)}
        out.setdefault("flag_share", {})[m] = round(flags / n, 3)
    for m in ["B2", "P-A*(4)", "P-RRT*(4)"]:
        f = M[m][M[m].outcome != "success"].reset_index()
        out["failures"][m] = f[["layout", "world", "outcome", "decisions"]].to_dict(orient="records")
    R = M["P-RRT*(4)"]; A = M["P-A*(4)"].loc[R.index]
    seq = lambda s: [(e[0], e[2]) for e in json.loads(s)]
    out["rrt"] = dict(missions=len(R), per_layout={l: int(len(x)) for l, x in R.groupby(level=0)},
                      same_outcome=int((R.outcome == A.outcome).sum()),
                      same_decisions=int(sum(seq(x) == seq(y) for x, y in zip(R.decisions, A.decisions))),
                      unsafe=int(R.unsafe.sum()), dtime=paired_time(A, R))
    out["cpu"] = {m: round(float(g.cpu_s.mean()), 2) for m, g in M.items()}
    json.dump(out, open(f"{res}/report_numbers.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["counts", "compare", "by_layout", "rrt", "cpu", "flag_share"]}, indent=1))
    print(json.dumps(out["evidence"], indent=1)); print(json.dumps(out["failures"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_rev")
