"""Harder-scenario and ablation statistics quoted in Report 2, Section 6.5 and 6.2.
Reads <results>/missions_raw_scen.csv, writes <results>/scenario_numbers.json.
Run: python -m autopass_sim.experiments --worlds 300 --rrt-worlds 0 --out results_rev \
        --methods "B2,P-A*(4),B2-nr,P-A*(4)-nr,B2-b4,P-A*(4)-b4,B1-rhohat" --tag _scen
     python tools/scenario_stats.py results_rev
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest, t as tdist


def pt(a, b):
    x = (b - a).dropna(); m = x.mean(); h = tdist.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [round(float(m), 2), round(float(m - h), 2), round(float(m + h), 2)]


def main(res):
    d = pd.read_csv(f"{res}/missions_raw_scen.csv")
    M = {m: g.set_index(["layout", "world"]).sort_index() for m, g in d.groupby("method")}
    out = {}
    for m, g in M.items():
        dec = [e for s in g.decisions for e in json.loads(s)]
        out[m] = dict(unsafe=int((g.outcome == "unsafe").sum()), success=int((g.outcome == "success").sum()),
                      no_route=int((g.outcome == "no_route").sum()),
                      flag=round(sum(e[2] == "FLAG" for e in dec) / len(dec), 3), ndec=len(dec),
                      time=round(float(g.time.mean()), 2), repos=round(float(g.repositions.mean()), 3),
                      repos_any=int((g.repositions > 0).sum()), dwell=round(float(g.dwell_scans.mean() * 0.5), 2),
                      tfd=round(float(g.t_first_decision.mean()), 2),
                      by={l: dict(success=int((x.outcome == "success").sum()), time=round(float(x.time.mean()), 2))
                          for l, x in g.groupby(level=0)})
    cmp = {}
    for a, b in [("B2", "P-A*(4)"), ("B2-nr", "P-A*(4)-nr"), ("B2-b4", "P-A*(4)-b4")]:
        A, B = M[a], M[b].loc[M[a].index]
        sa, sb = A.outcome == "success", B.outcome == "success"
        n10, n01 = int((sa & ~sb).sum()), int((~sa & sb).sum())
        p = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
        cmp[f"{a}->{b}"] = dict(success=[int(sa.sum()), int(sb.sum())], a_only=n10, b_only=n01, p=float(p),
                                dtime=pt(A.time, B.time), drepos=pt(A.repositions, B.repositions),
                                ddwell=pt(A.dwell_scans * 0.5, B.dwell_scans * 0.5),
                                dtfd=pt(A.t_first_decision, B.t_first_decision))
    json.dump(dict(methods=out, compare=cmp), open(f"{res}/scenario_numbers.json", "w"), indent=1)
    print(json.dumps(cmp, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_rev")
