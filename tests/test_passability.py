"""Sprint 7: tests for the agent passability classifier."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.passability import classify, load_profile  # noqa: E402

PROFILES = ROOT / "profiles"


@pytest.fixture
def wheelchair():
    return load_profile(PROFILES / "wheelchair.json")


@pytest.fixture
def walker():
    return load_profile(PROFILES / "walker.json")


@pytest.fixture
def ada():
    return load_profile(PROFILES / "ada_default.json")


def test_doorway_just_passing_wheelchair(wheelchair):
    r = classify({"type": "doorway", "width_m": 0.85}, wheelchair)
    assert r.status == "pass"
    assert r.failed_constraints == []
    assert r.profile_name == "wheelchair"


def test_doorway_just_failing_wheelchair(wheelchair):
    r = classify({"type": "doorway", "width_m": 0.75}, wheelchair)
    assert r.status == "fail"
    assert "doorway_width" in r.failed_constraints
    assert r.margin_m < 0


def test_doorway_in_flag_zone(wheelchair):
    r = classify({"type": "doorway", "width_m": 0.83}, wheelchair)
    assert r.status == "flag"


def test_bedroom_threshold_fails_all_profiles(wheelchair, walker, ada):
    feature = {"type": "threshold", "height_m": 0.10}
    for profile in (wheelchair, walker, ada):
        r = classify(feature, profile)
        assert r.status == "fail", f"{profile['name']} should fail at 10 cm threshold"
        assert "threshold_height" in r.failed_constraints


def test_ramp_4deg_passes_ada(ada):
    r = classify({"type": "ramp", "slope_deg": 4.0}, ada)
    assert r.status == "pass"


def test_ramp_5deg_fails_ada(ada):
    r = classify({"type": "ramp", "slope_deg": 5.0}, ada)
    assert r.status == "fail"
    assert "ramp_slope" in r.failed_constraints


def test_corridor_1m_wheelchair_passes(wheelchair):
    r = classify({"type": "corridor", "width_m": 1.0}, wheelchair)
    assert r.status == "pass"


def test_ramp_with_cross_slope_failure_propagates(ada):
    r = classify(
        {"type": "ramp", "slope_deg": 3.0, "cross_slope_deg": 3.0}, ada
    )
    assert r.status == "fail"
    assert "ramp_cross_slope" in r.failed_constraints


def test_unknown_feature_type_raises(ada):
    with pytest.raises(ValueError):
        classify({"type": "stair", "height_m": 0.2}, ada)
