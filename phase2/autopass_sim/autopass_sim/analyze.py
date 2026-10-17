"""Summary table with 95% confidence intervals, and the report figures.

Run:  python -m autopass_sim.analyze --results results --calib calib.json
"""
import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["B1", "B2", "P-A*(1)", "P-A*(2)", "P-A*(4)", "P-A*(8)", "P-A*(4)n", "P-RRT*(4)"]
MAROON, GOLD, GRAY, TEAL = "#7A1E2C", "#C9A227", "#555555", "#2F6F73"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0, c - h), min(1, c + h)


def mean_ci(x, z=1.96):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, m - z * se, m + z * se


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ["path_len", "time", "first_O", "cpu_s"]:
            r[k] = float(r[k])
        for k in ["dwell_scans", "false_pass", "false_fail", "unresolved", "n_decisions", "world"]:
            r[k] = int(r[k])
        r["first_est_err"] = float(r["first_est_err"]) if r["first_est_err"] != "" else np.nan
        r["decisions"] = json.loads(r["decisions"])
    return rows


def summarize(rows):
    out = []
    for layout in sorted({r["layout"] for r in rows}):
        for m in ORDER:
            R = [r for r in rows if r["layout"] == layout and r["method"] == m]
            if not R:
                continue
            n = len(R)
            # decision-level counts: every commit-point decision in every mission
            D = [d for r in R for d in r["decisions"]]
            wrong = sum(1 for d in D if (d[2] == "PASS" and d[1] < 0) or (d[2] == "FAIL" and d[1] >= 0))
            fp = sum(1 for d in D if d[2] == "PASS" and d[1] < 0)
            flag = sum(1 for d in D if d[2] == "FLAG")
            err = [abs(d[3] - d[1]) for d in D if d[3] is not None]
            # paired world set for time comparison vs B2
            b2 = {r["world"]: r for r in rows if r["layout"] == layout and r["method"] == "B2"}
            dt = [r["time"] - b2[r["world"]]["time"] for r in R if r["world"] in b2]
            out.append(dict(
                layout=layout, method=m, missions=n,
                success=wilson(sum(r["outcome"] == "success" for r in R), n),
                unsafe=wilson(sum(r["outcome"] == "unsafe" for r in R), n),
                no_route=wilson(sum(r["outcome"] == "no_route" for r in R), n),
                decisions=len(D),
                wrong=wilson(wrong, len(D)), false_pass=wilson(fp, len(D)), flag=wilson(flag, len(D)),
                abs_err=mean_ci(err), path_len=mean_ci([r["path_len"] for r in R]),
                time=mean_ci([r["time"] for r in R]), d_time_vs_B2=mean_ci(dt),
                dwell_s=mean_ci([r["dwell_scans"] * 0.5 for r in R]),
                first_O=mean_ci([r["first_O"] for r in R]),
                cpu_s=mean_ci([r["cpu_s"] for r in R])))
    return out


def fmt(t, pct=False, nd=1):
    if pct:
        return f"{100 * t[0]:.{nd}f}% [{100 * t[1]:.{nd}f}, {100 * t[2]:.{nd}f}]"
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def table(summary, path):
    cols = ["layout", "method", "missions", "success", "unsafe", "wrong", "false_pass", "flag", "abs_err",
            "path_len", "time", "d_time_vs_B2", "dwell_s", "first_O"]
    with open(path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for s in summary:
            cells = []
            for c in cols:
                v = s[c]
                if c in ("success", "unsafe", "wrong", "false_pass", "flag"):
                    cells.append(fmt(v, pct=True))
                elif isinstance(v, tuple):
                    cells.append(fmt(v, nd=2 if c in ("abs_err", "first_O") else 1))
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")


def fig_tradeoff(summary, path):
    """Sensing quality and time cost vs lambda (lambda = 0 is B2), paired against B2."""
    plt.rcParams.update({"font.size": 10})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    lam_m = [("B2", 0), ("P-A*(1)", 1), ("P-A*(2)", 2), ("P-A*(4)", 4), ("P-A*(8)", 8)]
    for k, (lay, col, mk) in enumerate([("corridor", MAROON, "o"), ("room", TEAL, "s")]):
        S = {s["method"]: s for s in summary if s["layout"] == lay}
        L = [l for m, l in lam_m if m in S]
        x = np.arange(len(L)) + (k - 0.5) * 0.12
        for ax, key in [(a1, "first_O"), (a2, "d_time_vs_B2")]:
            v = [S[m][key] for m, l in lam_m if m in S]
            ax.errorbar(x, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]],
                        fmt="-" + mk, color=col, capsize=3, label=f"{lay} layout")
    for ax in (a1, a2):
        ax.set_xticks(range(len(lam_m))); ax.set_xticklabels([str(l) for _, l in lam_m])
        ax.set_xlabel("observability weight λ (0 = shortest path)")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    a1.set_ylabel("trusted evidence O at\nfirst decision (1/cm²)")
    a1.set_title("(a) sensing quality", fontsize=10)
    a2.axhline(0, color=GRAY, lw=0.8)
    a2.set_ylabel("extra mission time vs λ=0 (s)")
    a2.set_title("(b) time cost (paired)", fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_outcomes(summary, path):
    layouts = sorted({s["layout"] for s in summary})
    methods = ["B1", "B2", "P-A*(4)n", "P-A*(4)", "P-RRT*(4)"]
    fig, axes = plt.subplots(1, len(layouts), figsize=(4.2 * len(layouts), 3.2), squeeze=False)
    for ax, lay in zip(axes[0], layouts):
        S = {s["method"]: s for s in summary if s["layout"] == lay}
        ms = [m for m in methods if m in S]
        x = np.arange(len(ms))
        corr = [100 * (1 - S[m]["wrong"][0] - S[m]["flag"][0]) for m in ms]
        wr = [100 * S[m]["wrong"][0] for m in ms]
        fl = [100 * S[m]["flag"][0] for m in ms]
        ax.bar(x, corr, color=TEAL, label="correct")
        ax.bar(x, fl, bottom=corr, color="#CFCFCF", label="FLAG")
        ax.bar(x, wr, bottom=np.array(corr) + np.array(fl), color=MAROON, label="wrong")
        ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=8, rotation=20)
        ax.set_ylim(0, 100); ax.set_ylabel("% of door decisions"); ax.set_title(f"{lay} layout", fontsize=10)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(1.1, 1.22), ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sensor", default="raw")
    a = ap.parse_args()
    rows = load(os.path.join(a.results, f"missions_{a.sensor}.csv"))
    S = summarize(rows)
    table(S, os.path.join(a.results, f"summary_{a.sensor}.md"))
    json.dump(S, open(os.path.join(a.results, f"summary_{a.sensor}.json"), "w"), indent=1, default=float)
    fig_tradeoff(S, os.path.join(a.results, f"fig_tradeoff_{a.sensor}.png"))
    fig_outcomes(S, os.path.join(a.results, f"fig_outcomes_{a.sensor}.png"))
    print(open(os.path.join(a.results, f"summary_{a.sensor}.md")).read())


if __name__ == "__main__":
    main()
