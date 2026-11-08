#!/usr/bin/env python3
"""Sprint 8.6: Dual-reference evaluation harness.

Compares detected doorway widths against two ground-truth references:
  - clear_width (primary) — jamb-face to jamb-face, door open. This is
    the passable opening and aligns with ADA Standards for Accessible
    Design §404.2.3, which defines minimum clear width as measured
    between the face of the door and the stop on the latch side.
  - rough_width (secondary) — jamb-trim to jamb-trim (framed opening).

Usage:
    python scripts/evaluate.py --match wall_02+wall_04:door_01
    python scripts/evaluate.py --detections data/doorways_v4.json \
        --output evaluation_run01_v2 --match wall_02+wall_04:door_01
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTIONS = ROOT / "data" / "doorways_v4.json"
GROUND_TRUTH = ROOT / "ground_truth.txt"
RESULTS_DIR = ROOT / "results"


def load_ground_truth():
    entries = {}
    with open(GROUND_TRUTH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",", 3)]
            fid = parts[0]
            clear = float(parts[1]) if parts[1] else None
            rough = float(parts[2]) if parts[2] else None
            desc = parts[3] if len(parts) > 3 else ""
            entries[fid] = dict(feature_id=fid, clear_width_cm=clear,
                                rough_width_cm=rough, description=desc)
    return entries


def load_detections(path):
    with open(path) as f:
        return json.load(f)


def build_matches(detections, match_args):
    det_by_wall = {d["source_wall"]: d for d in detections}
    matches = []
    for m in match_args:
        wall, fid = m.split(":")
        if wall in det_by_wall:
            matches.append((det_by_wall[wall], fid))
        else:
            print(f"  WARNING: no detection for wall '{wall}', skipping")
    return matches


def _err(detected, truth):
    if truth is None:
        return None, None, None
    signed = round(detected - truth, 2)
    absolute = round(abs(signed), 2)
    relative = round(absolute / truth * 100, 2)
    return signed, absolute, relative


def compute_errors(matches, gt):
    rows = []
    for det, fid in matches:
        if fid not in gt:
            print(f"  WARNING: '{fid}' not in ground truth, skipping")
            continue
        g = gt[fid]
        detected_cm = round(det.get("width_m_refined", det["width_m"]) * 100, 1)
        sc, ac, rc = _err(detected_cm, g["clear_width_cm"])
        sr, ar, rr = _err(detected_cm, g["rough_width_cm"])
        rows.append(dict(
            feature_id=fid, description=g["description"],
            detected_cm=detected_cm,
            clear_width_cm=g["clear_width_cm"],
            signed_err_clear=sc, abs_err_clear=ac, rel_err_clear=rc,
            rough_width_cm=g["rough_width_cm"],
            signed_err_rough=sr, abs_err_rough=ar, rel_err_rough=rr,
            source_wall=det["source_wall"], confidence=det["confidence"],
        ))
    return rows


def _agg(rows, key, n):
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None, None, None, 0.0
    mae = round(sum(vals) / len(vals), 2)
    rmse = round(math.sqrt(sum(v ** 2 for v in vals) / len(vals)), 2)
    mx = round(max(vals), 2)
    rate = round(len(vals) / n, 2) if n else 0.0
    return mae, rmse, mx, rate


def aggregate(rows, gt):
    n_total = len(gt)
    n_in_scope = sum(1 for g in gt.values() if g["clear_width_cm"] is not None)
    mae_c, rmse_c, max_c, _ = _agg(rows, "abs_err_clear", n_in_scope)
    mae_r, rmse_r, max_r, _ = _agg(rows, "abs_err_rough", n_total)
    n_matched = sum(1 for r in rows if r["abs_err_clear"] is not None)
    return dict(
        MAE_vs_clear_cm=mae_c, RMSE_vs_clear_cm=rmse_c,
        max_error_vs_clear_cm=max_c,
        MAE_vs_rough_cm=mae_r, RMSE_vs_rough_cm=rmse_r,
        max_error_vs_rough_cm=max_r,
        in_scope_detection_rate=round(n_matched / n_in_scope, 2) if n_in_scope else 0.0,
        overall_detection_rate=round(n_matched / n_total, 2) if n_total else 0.0,
        n_matched=n_matched, n_in_scope=n_in_scope, n_total=n_total,
        primary_reference="clear_width (ADA §404.2.3)",
    )


def save_results(rows, agg, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = RESULTS_DIR / f"{name}.json"
    cp = RESULTS_DIR / f"{name}.csv"
    with open(jp, "w") as f:
        json.dump({"per_feature": rows, "aggregate": agg}, f, indent=2)
    with open(cp, "w", newline="") as f:
        if rows:
            csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader()
            csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)
    print(f"  JSON: {jp}\n  CSV:  {cp}")


def plot_errors(rows, agg, name):
    fp = ROOT / "figures" / f"{name}.png"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    matched = [r for r in rows if r["signed_err_clear"] is not None]
    if not matched:
        ax.text(0.5, 0.5, "No matched detections", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        plt.savefig(str(fp), dpi=150, facecolor="white"); plt.close(); return

    import numpy as np
    labels = [r["feature_id"] for r in matched]
    x = np.arange(len(labels))
    w = 0.35
    sc = [r["signed_err_clear"] for r in matched]
    sr = [r["signed_err_rough"] for r in matched]

    b1 = ax.bar(x - w / 2, sc, w, label="vs clear", color="#1f77b4", edgecolor="black")
    b2 = ax.bar(x + w / 2, sr, w, label="vs rough", color="#ff7f0e", edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    for bar, val in list(zip(b1, sc)) + list(zip(b2, sr)):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:+.1f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")

    info = (f"MAE vs clear = {agg['MAE_vs_clear_cm']:.2f} cm\n"
            f"MAE vs rough = {agg['MAE_vs_rough_cm']:.2f} cm")
    ax.text(0.98, 0.95, info, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Signed Error (cm)")
    ax.set_title(f"Sprint 8.6: Dual-Reference Evaluation "
                 f"({agg['n_matched']}/{agg['n_in_scope']} in-scope matched)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(fp), dpi=150, facecolor="white"); plt.close()
    print(f"  Plot: {fp}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate doorway detections")
    parser.add_argument("--match", nargs="+", default=[],
                        help="wall:feature_id mappings")
    parser.add_argument("--detections", default=str(DEFAULT_DETECTIONS))
    parser.add_argument("--output", default="evaluation_run01_v2")
    args = parser.parse_args()

    gt = load_ground_truth()
    dp = Path(args.detections)
    if not dp.is_absolute():
        dp = ROOT / dp
    dets = load_detections(dp)
    n_scope = sum(1 for g in gt.values() if g["clear_width_cm"] is not None)
    print(f"Ground truth:  {len(gt)} features ({n_scope} in scope)")
    print(f"Detections:    {len(dets)} doorways ({dp.name})\n")

    matches = build_matches(dets, args.match)
    rows = compute_errors(matches, gt)
    agg = aggregate(rows, gt)

    hdr = (f"{'Feature':>12}  {'Det':>6}  {'Clear':>6}  {'Err/C':>7}  "
           f"{'Rough':>6}  {'Err/R':>7}  {'Conf':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        c = f"{r['clear_width_cm']:.1f}" if r['clear_width_cm'] else "  -"
        ec = f"{r['signed_err_clear']:+.1f}" if r['signed_err_clear'] is not None else "  -"
        ru = f"{r['rough_width_cm']:.1f}" if r['rough_width_cm'] else "  -"
        er = f"{r['signed_err_rough']:+.1f}" if r['signed_err_rough'] is not None else "  -"
        print(f"{r['feature_id']:>12}  {r['detected_cm']:6.1f}  {c:>6}  "
              f"{ec:>7}  {ru:>6}  {er:>7}  {r['confidence']:5.2f}")
    print("-" * len(hdr))
    mc = agg['MAE_vs_clear_cm']; mr = agg['MAE_vs_rough_cm']
    print(f"  MAE vs clear: {mc:.2f} cm | MAE vs rough: {mr:.2f} cm")
    print(f"  In-scope detect rate: {agg['in_scope_detection_rate']:.0%} "
          f"| Overall: {agg['overall_detection_rate']:.0%}")

    save_results(rows, agg, args.output)
    plot_errors(rows, agg, args.output)


if __name__ == "__main__":
    main()
