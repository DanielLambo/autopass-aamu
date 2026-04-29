#!/usr/bin/env python3
"""Sprint 8: Evaluation harness — compare detected doorways to ground truth.

Computes per-feature absolute/signed/relative errors and aggregate MAE,
RMSE, max error, and detection rate.  Accepts --match flags to map
detections (by source_wall) to ground truth feature IDs.

Usage:
    python scripts/evaluate.py --match wall_02+wall_04:door_01
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
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DETECTIONS = ROOT / "data" / "doorways_v3.json"
GROUND_TRUTH = ROOT / "ground_truth.txt"
RESULTS_DIR = ROOT / "results"
FIG_PATH = ROOT / "figures" / "sprint8_errors.png"


def load_ground_truth():
    entries = {}
    with open(GROUND_TRUTH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("feature_id"):
                continue
            parts = [p.strip() for p in line.split(",", 2)]
            fid, width_cm, desc = parts[0], float(parts[1]), parts[2]
            entries[fid] = {"feature_id": fid, "width_cm": width_cm,
                            "description": desc}
    return entries


def load_detections():
    with open(DETECTIONS) as f:
        return json.load(f)


def build_matches(detections, match_args):
    """Return list of (detection, feature_id) from --match wall:fid args."""
    det_by_wall = {d["source_wall"]: d for d in detections}
    matches = []
    for m in match_args:
        wall, fid = m.split(":")
        if wall not in det_by_wall:
            print(f"  WARNING: no detection for wall '{wall}', skipping")
            continue
        matches.append((det_by_wall[wall], fid))
    return matches


def compute_errors(matches, gt):
    rows = []
    for det, fid in matches:
        if fid not in gt:
            print(f"  WARNING: '{fid}' not in ground truth, skipping")
            continue
        detected_cm = det["width_m"] * 100
        truth_cm = gt[fid]["width_cm"]
        abs_err = abs(detected_cm - truth_cm)
        signed_err = detected_cm - truth_cm
        rel_err = abs_err / truth_cm * 100
        rows.append({
            "feature_id": fid,
            "description": gt[fid]["description"],
            "ground_truth_cm": truth_cm,
            "detected_cm": round(detected_cm, 1),
            "absolute_error_cm": round(abs_err, 2),
            "signed_error_cm": round(signed_err, 2),
            "relative_error_pct": round(rel_err, 2),
            "source_wall": det["source_wall"],
            "confidence": det["confidence"],
        })
    return rows


def aggregate(rows, n_gt):
    if not rows:
        return {"MAE_cm": None, "RMSE_cm": None, "max_error_cm": None,
                "detection_rate": 0.0, "n_matched": 0, "n_ground_truth": n_gt}
    errs = [r["absolute_error_cm"] for r in rows]
    return {
        "MAE_cm": round(sum(errs) / len(errs), 2),
        "RMSE_cm": round(math.sqrt(sum(e ** 2 for e in errs) / len(errs)), 2),
        "max_error_cm": round(max(errs), 2),
        "detection_rate": round(len(rows) / n_gt, 2) if n_gt else 0.0,
        "n_matched": len(rows),
        "n_ground_truth": n_gt,
    }


def save_results(rows, agg):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "evaluation_run01.json"
    csv_path = RESULTS_DIR / "evaluation_run01.csv"

    with open(json_path, "w") as f:
        json.dump({"per_feature": rows, "aggregate": agg}, f, indent=2)
    print(f"  JSON: {json_path}")

    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"  CSV:  {csv_path}")


def plot_errors(rows, agg):
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not rows:
        ax.text(0.5, 0.5, "No matched detections", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        plt.savefig(str(FIG_PATH), dpi=150, facecolor="white")
        plt.close()
        return

    labels = [r["feature_id"] for r in rows]
    signed = [r["signed_error_cm"] for r in rows]
    colors = ["#2ca02c" if s >= 0 else "#d62728" for s in signed]

    bars = ax.bar(labels, signed, color=colors, edgecolor="black", width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, signed):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:+.1f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=10,
                fontweight="bold")

    mae = agg["MAE_cm"]
    ax.text(0.98, 0.95, f"MAE = {mae:.2f} cm\nRMSE = {agg['RMSE_cm']:.2f} cm",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

    ax.set_ylabel("Signed Error (cm)")
    ax.set_title(f"Sprint 8: Detection Error vs Ground Truth "
                 f"({agg['n_matched']}/{agg['n_ground_truth']} matched)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(FIG_PATH), dpi=150, facecolor="white")
    plt.close()
    print(f"  Plot: {FIG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate doorway detections")
    parser.add_argument("--match", nargs="+", default=[],
                        help="wall:feature_id mappings (e.g. wall_02+wall_04:door_01)")
    args = parser.parse_args()

    gt = load_ground_truth()
    detections = load_detections()
    print(f"Ground truth: {len(gt)} features")
    print(f"Detections:   {len(detections)} doorways\n")

    matches = build_matches(detections, args.match)
    rows = compute_errors(matches, gt)
    agg = aggregate(rows, len(gt))

    # Console summary
    print(f"{'Feature':>14}  {'GT cm':>6}  {'Det cm':>6}  "
          f"{'Err cm':>7}  {'Rel %':>6}  {'Conf':>5}")
    print("-" * 60)
    for r in rows:
        print(f"{r['feature_id']:>14}  {r['ground_truth_cm']:6.1f}  "
              f"{r['detected_cm']:6.1f}  {r['signed_error_cm']:+7.2f}  "
              f"{r['relative_error_pct']:5.1f}%  {r['confidence']:5.2f}")
    print("-" * 60)
    print(f"{'MAE':>14}  {agg['MAE_cm']:6.2f} cm   "
          f"RMSE {agg['RMSE_cm']:.2f} cm   "
          f"detect rate {agg['detection_rate']:.0%}")

    save_results(rows, agg)
    plot_errors(rows, agg)


if __name__ == "__main__":
    main()
