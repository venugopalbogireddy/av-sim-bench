"""Unit tests for each metric using hand-crafted DataFrames with known answers."""

import pandas as pd
import pytest

from evaluator.metrics import (
    stop_sign_compliance,
    red_light_violation_rate,
    collision_proxy,
    route_completion,
    speed_ks_test,
    STOP_SPEED_THRESHOLD,
    STOP_DWELL_FRAMES,
    GOAL_Y,
)


def _base_row(**overrides) -> dict:
    row = {
        "run_id": "test",
        "timestamp_ms": 0,
        "agent_id": "agent_00",
        "x": 0.0,
        "y": 0.0,
        "heading": 90.0,
        "speed_mps": 8.0,
        "traffic_light_state": "GREEN",
        "stop_sign_zone": False,
        "in_intersection": False,
        "collision_flag": False,
    }
    row.update(overrides)
    return row


def _stop_zone_rows(agent_id: str, n_frames: int, speed: float) -> list[dict]:
    """Build n_frames consecutive stop-zone rows for one agent at a fixed speed."""
    return [
        _base_row(agent_id=agent_id, timestamp_ms=t * 100,
                  stop_sign_zone=True, speed_mps=speed)
        for t in range(n_frames)
    ]


# ── stop_sign_compliance ──────────────────────────────────────────────────────

class TestStopSignCompliance:
    def test_all_compliant(self):
        # Both agents hold 0 m/s for exactly STOP_DWELL_FRAMES consecutive frames
        df = pd.DataFrame(
            _stop_zone_rows("a0", STOP_DWELL_FRAMES, 0.0) +
            _stop_zone_rows("a1", STOP_DWELL_FRAMES, 0.0)
        )
        r = stop_sign_compliance(df)
        assert r.passed is True
        assert r.value == pytest.approx(1.0)

    def test_one_violator(self):
        # a0 completes full dwell; a1 never reaches 0 m/s
        df = pd.DataFrame(
            _stop_zone_rows("a0", STOP_DWELL_FRAMES, 0.0) +
            _stop_zone_rows("a1", STOP_DWELL_FRAMES, 2.5)  # rolling stop — violator
        )
        r = stop_sign_compliance(df)
        assert r.passed is False
        assert r.value == pytest.approx(0.5)

    def test_dwell_too_short_not_compliant(self):
        # Agent reaches 0 m/s but only for STOP_DWELL_FRAMES - 1 frames — not enough
        df = pd.DataFrame(
            _stop_zone_rows("a0", STOP_DWELL_FRAMES - 1, 0.0)
        )
        r = stop_sign_compliance(df)
        assert r.passed is False

    def test_no_stop_zone_rows(self):
        df = pd.DataFrame([_base_row(stop_sign_zone=False)])
        r = stop_sign_compliance(df)
        assert r.passed is True

    def test_exactly_at_threshold_not_compliant(self):
        # STOP_SPEED_THRESHOLD is 0.0, so any speed > 0 fails even with enough frames
        df = pd.DataFrame(
            _stop_zone_rows("a0", STOP_DWELL_FRAMES, 0.1)
        )
        r = stop_sign_compliance(df)
        assert r.passed is False


# ── red_light_violation_rate ──────────────────────────────────────────────────

class TestRedLightViolationRate:
    def test_no_violations(self):
        # Agent inside intersection but light is GREEN — not a violation
        df = pd.DataFrame([
            _base_row(in_intersection=True, traffic_light_state="GREEN"),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is True
        assert r.value == pytest.approx(0.0)

    def test_stopped_at_red_not_a_violation(self):
        # Agent inside intersection, light RED, but speed == 0 — waiting, not violating
        df = pd.DataFrame([
            _base_row(in_intersection=True, traffic_light_state="RED", speed_mps=0.0),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is True
        assert r.value == pytest.approx(0.0)

    def test_one_violator_of_two(self):
        # a0: inside intersection + RED + moving = violation; a1: GREEN = fine
        df = pd.DataFrame([
            _base_row(agent_id="a0", in_intersection=True, traffic_light_state="RED"),
            _base_row(agent_id="a1", in_intersection=True, traffic_light_state="GREEN"),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is False
        assert r.value == pytest.approx(0.5)

    def test_all_violate(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", in_intersection=True, traffic_light_state="RED"),
            _base_row(agent_id="a1", in_intersection=True, traffic_light_state="RED"),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is False
        assert r.value == pytest.approx(1.0)

    def test_red_outside_intersection_not_counted(self):
        # in_intersection=False even if RED + moving — field is the gate, not position
        df = pd.DataFrame([
            _base_row(in_intersection=False, traffic_light_state="RED"),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is True

    # ── SG-06: entered-on-GREEN exemption ─────────────────────────────────────

    def test_entered_on_green_exempts_mid_crossing_red(self):
        # Agent enters on GREEN, light flips RED mid-crossing — legal traversal, not a violation
        df = pd.DataFrame([
            _base_row(timestamp_ms=0,   in_intersection=False, traffic_light_state="GREEN"),
            _base_row(timestamp_ms=100, in_intersection=True,  traffic_light_state="GREEN"),
            _base_row(timestamp_ms=200, in_intersection=True,  traffic_light_state="RED"),
            _base_row(timestamp_ms=300, in_intersection=True,  traffic_light_state="RED"),
            _base_row(timestamp_ms=400, in_intersection=False, traffic_light_state="RED"),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is True
        assert r.value == pytest.approx(0.0)

    def test_entered_on_red_is_violation(self):
        # First in-intersection frame is already RED + moving — genuine violation
        df = pd.DataFrame([
            _base_row(timestamp_ms=0,   in_intersection=False, traffic_light_state="GREEN"),
            _base_row(timestamp_ms=100, in_intersection=True,  traffic_light_state="RED"),
            _base_row(timestamp_ms=200, in_intersection=True,  traffic_light_state="RED"),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is False
        assert r.value == pytest.approx(1.0)

    def test_entered_on_yellow_not_exempt(self):
        # YELLOW entry does not grant exemption — only GREEN does
        df = pd.DataFrame([
            _base_row(timestamp_ms=0,   in_intersection=False, traffic_light_state="GREEN"),
            _base_row(timestamp_ms=100, in_intersection=True,  traffic_light_state="YELLOW"),
            _base_row(timestamp_ms=200, in_intersection=True,  traffic_light_state="RED"),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is False
        assert r.value == pytest.approx(1.0)

    def test_two_crossings_first_green_second_red(self):
        # First traversal: entered GREEN, light flips RED mid-crossing → exempt
        # Second traversal: entered directly on RED → violation
        df = pd.DataFrame([
            _base_row(timestamp_ms=0,   in_intersection=False, traffic_light_state="GREEN"),
            _base_row(timestamp_ms=100, in_intersection=True,  traffic_light_state="GREEN"),
            _base_row(timestamp_ms=200, in_intersection=True,  traffic_light_state="RED"),
            _base_row(timestamp_ms=300, in_intersection=False, traffic_light_state="RED"),
            _base_row(timestamp_ms=400, in_intersection=False, traffic_light_state="RED"),
            _base_row(timestamp_ms=500, in_intersection=True,  traffic_light_state="RED"),
            _base_row(timestamp_ms=600, in_intersection=True,  traffic_light_state="RED"),
        ])
        r = red_light_violation_rate(df)
        assert r.passed is False
        assert r.value == pytest.approx(1.0)


# ── collision_proxy ───────────────────────────────────────────────────────────

class TestCollisionProxy:
    def test_no_collision(self):
        df = pd.DataFrame([_base_row(collision_flag=False)])
        r = collision_proxy(df)
        assert r.passed is True
        assert r.value == 0.0

    def test_two_collisions(self):
        df = pd.DataFrame([
            _base_row(collision_flag=True),
            _base_row(collision_flag=True),
            _base_row(collision_flag=False),
        ])
        r = collision_proxy(df)
        assert r.passed is False
        assert r.value == 2.0


# ── route_completion ──────────────────────────────────────────────────────────

class TestRouteCompletion:
    def test_all_complete(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", y=GOAL_Y + 1),
            _base_row(agent_id="a1", y=GOAL_Y),
        ])
        r = route_completion(df)
        assert r.passed is True
        assert r.value == pytest.approx(1.0)

    def test_none_complete(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", y=10.0),
            _base_row(agent_id="a1", y=20.0),
        ])
        r = route_completion(df)
        assert r.passed is False
        assert r.value == pytest.approx(0.0)

    def test_partial_completion(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", y=GOAL_Y + 1),
            _base_row(agent_id="a1", y=10.0),
        ])
        r = route_completion(df)
        assert r.passed is False
        assert r.value == pytest.approx(0.5)


# ── speed_ks_test ─────────────────────────────────────────────────────────────

class TestSpeedKsTest:
    def _cruise_df(self, speeds: list[float]) -> pd.DataFrame:
        return pd.DataFrame([
            _base_row(speed_mps=s, stop_sign_zone=False, traffic_light_state="GREEN")
            for s in speeds
        ])

    def test_identical_distributions_pass(self):
        speeds = [8.0 + i * 0.01 for i in range(100)]
        df = self._cruise_df(speeds)
        baseline = self._cruise_df(speeds)
        r = speed_ks_test(df, baseline)
        assert r.passed is True

    def test_very_different_distributions_fail(self):
        df = self._cruise_df([8.0] * 200)
        baseline = self._cruise_df([20.0] * 200)
        r = speed_ks_test(df, baseline)
        assert r.passed is False

    def test_empty_run_data(self):
        df = pd.DataFrame([_base_row(stop_sign_zone=True)])  # no cruise rows
        baseline = self._cruise_df([8.0] * 50)
        r = speed_ks_test(df, baseline)
        assert r.passed is True  # graceful fallback
