"""
Agent passability classifier for AutoPass (Sprint 7).

Algorithm:
    classify(feature, profile) compares one geometric feature
    (doorway width, corridor width, threshold height, ramp slope) against
    the limits defined by an agent profile (ADA defaults, wheelchair,
    walker). Each constraint resolves to pass / flag / fail; the most
    severe status across constraints is the overall result. The signed
    margin to the tightest constraint is returned in margin_m: positive
    means the feature clears the limit by that amount, negative means it
    misses by that amount (units: meters for length features, degrees
    for slope features).

Flag-zone rationale:
    LiDAR-based geometric estimation has measurement uncertainty on the
    order of one voxel (~3 cm at our 0.03 m voxel size) plus RANSAC fit
    residual on the order of the inlier distance threshold (0.03-0.05
    m). A doorway width estimate is therefore only trustworthy to within
    a few centimeters even before sensor noise. We treat any value
    within ±2.5 cm of a hard limit (5 cm total band) as "flag" rather
    than committing to pass or fail, surfacing the cases where the true
    value could plausibly land on either side. For slope constraints
    the equivalent uncertainty band is ±0.1 deg, the propagated normal-
    estimation uncertainty for a wall patch of a few hundred points.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

FLAG_ZONE_M = 0.025
FLAG_ZONE_DEG = 0.1

_RANK = {"pass": 0, "flag": 1, "fail": 2}


@dataclass
class ClassificationResult:
    status: str
    margin_m: float
    failed_constraints: list[str] = field(default_factory=list)
    profile_name: str = ""


def load_profile(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _check_min(name: str, value: float, limit: float, zone: float):
    margin = value - limit
    if margin < -zone:
        return "fail", margin, name
    if margin < zone:
        return "flag", margin, name
    return "pass", margin, None


def _check_max(name: str, value: float, limit: float, zone: float):
    margin = limit - value
    if margin < -zone:
        return "fail", margin, name
    if margin < zone:
        return "flag", margin, name
    return "pass", margin, None


def _checks_for(feature: dict, profile: dict) -> list[tuple[str, float, Optional[str]]]:
    ftype = feature["type"]
    if ftype == "doorway":
        return [_check_min("doorway_width", feature["width_m"],
                           profile["min_doorway_width_m"], FLAG_ZONE_M)]
    if ftype == "corridor":
        return [_check_min("corridor_width", feature["width_m"],
                           profile["min_corridor_width_m"], FLAG_ZONE_M)]
    if ftype == "threshold":
        return [_check_max("threshold_height", feature["height_m"],
                           profile["max_threshold_height_m"], FLAG_ZONE_M)]
    if ftype == "ramp":
        out = [_check_max("ramp_slope", feature["slope_deg"],
                          profile["max_slope_deg"], FLAG_ZONE_DEG)]
        if "cross_slope_deg" in feature:
            out.append(_check_max("ramp_cross_slope", feature["cross_slope_deg"],
                                  profile["max_cross_slope_deg"], FLAG_ZONE_DEG))
        return out
    raise ValueError(f"unknown feature type: {ftype}")


def classify(feature: dict, profile: dict) -> ClassificationResult:
    checks = _checks_for(feature, profile)
    status = max((c[0] for c in checks), key=_RANK.get)
    margin = min(c[1] for c in checks)
    failed = [c[2] for c in checks if c[0] != "pass" and c[2] is not None]
    return ClassificationResult(
        status=status,
        margin_m=margin,
        failed_constraints=failed,
        profile_name=profile.get("name", ""),
    )
