"""Tests for womd_adapter.py — mock-based, no TFRecord or tensorflow required.

All tests operate on synthetic proto-like objects that mirror the WOMD
Scenario proto interface. This keeps the test suite runnable without the
waymo-open-dataset package installed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from evaluator.schema import SCHEMA
from evaluator.womd_adapter import (
    _build_signal_timeline,
    _score_scenario,
    _stop_sign_positions,
    scenario_to_df,
    select_scenarios,
)


# ---------------------------------------------------------------------------
# Helpers to build mock WOMD proto objects
# ---------------------------------------------------------------------------

def _make_state(x=0.0, y=0.0, heading=0.0, vx=5.0, vy=0.0, valid=True):
    return SimpleNamespace(
        center_x=x, center_y=y, heading=heading,
        velocity_x=vx, velocity_y=vy, valid=valid,
    )


def _make_track(track_id: int, states: list, object_type: int = 1):
    """object_type=1 → TYPE_VEHICLE"""
    return SimpleNamespace(id=track_id, object_type=object_type, states=states)


def _make_lane_state(state_int: int, lane_id: int = 0):
    return SimpleNamespace(state=state_int, lane=lane_id)


def _make_dms(lane_state_ints: list[int]):
    return SimpleNamespace(lane_states=[_make_lane_state(s) for s in lane_state_ints])


def _make_stop_sign(x: float, y: float):
    pos = SimpleNamespace(x=x, y=y)
    ss = SimpleNamespace(position=pos)
    feature = SimpleNamespace(
        stop_sign=ss,
        WhichOneof=lambda _oneof: "stop_sign",
    )
    return feature


def _make_scenario(
    scenario_id: str = "test_scenario",
    n_steps: int = 5,
    agents: list[dict] | None = None,
    signal_states: list[int] | None = None,
    stop_sign_xy: tuple[float, float] | None = None,
) -> SimpleNamespace:
    """Build a minimal mock Scenario proto.

    agents: list of {"x": ..., "y": ..., "vx": ..., "valid": True/False}
            Each agent becomes one track with n_steps identical states.
    signal_states: per-step LANE_STATE int (default 6=GO → GREEN for all steps)
    stop_sign_xy: (x, y) of one stop sign, or None
    """
    timestamps = [i * 0.1 for i in range(n_steps)]  # 100ms steps

    if agents is None:
        agents = [{"x": 10.0, "y": 20.0, "vx": 5.0, "vy": 0.0, "valid": True}]

    tracks = []
    for i, ag in enumerate(agents):
        states = [
            _make_state(
                x=ag.get("x", 0.0),
                y=ag.get("y", 0.0),
                vx=ag.get("vx", 5.0),
                vy=ag.get("vy", 0.0),
                valid=ag.get("valid", True),
            )
            for _ in range(n_steps)
        ]
        tracks.append(_make_track(track_id=i, states=states, object_type=1))

    if signal_states is None:
        signal_states = [6] * n_steps  # LANE_STATE_GO = GREEN

    dms_list = [_make_dms([s]) for s in signal_states]

    map_features = []
    if stop_sign_xy:
        map_features.append(_make_stop_sign(*stop_sign_xy))
    else:
        # Add a dummy non-stop-sign feature so WhichOneof path is exercised
        map_features.append(SimpleNamespace(WhichOneof=lambda _: "road_edge"))

    return SimpleNamespace(
        scenario_id=scenario_id,
        timestamps_seconds=timestamps,
        tracks=tracks,
        dynamic_map_states=dms_list,
        map_features=map_features,
    )


# ---------------------------------------------------------------------------
# Tests: _build_signal_timeline
# ---------------------------------------------------------------------------

class TestBuildSignalTimeline:
    def test_all_green(self):
        sc = _make_scenario(signal_states=[6, 6, 6])  # GO
        tl = _build_signal_timeline(sc)
        assert tl == ["GREEN", "GREEN", "GREEN"]

    def test_all_red(self):
        sc = _make_scenario(signal_states=[4, 4, 4])  # STOP
        tl = _build_signal_timeline(sc)
        assert tl == ["RED", "RED", "RED"]

    def test_mixed_most_restrictive(self):
        # step 0: GO(6) + CAUTION(5) → YELLOW (most restrictive of the two)
        sc = _make_scenario(n_steps=1)
        sc.dynamic_map_states = [_make_dms([6, 5])]
        tl = _build_signal_timeline(sc)
        assert tl == ["YELLOW"]

    def test_red_beats_yellow(self):
        sc = _make_scenario(n_steps=1)
        sc.dynamic_map_states = [_make_dms([5, 4])]  # CAUTION + STOP
        tl = _build_signal_timeline(sc)
        assert tl == ["RED"]

    def test_empty_dynamic_map_returns_green(self):
        sc = _make_scenario(n_steps=3)
        sc.dynamic_map_states = []
        tl = _build_signal_timeline(sc)
        assert tl == []  # empty timeline → caller pads with GREEN


# ---------------------------------------------------------------------------
# Tests: _stop_sign_positions
# ---------------------------------------------------------------------------

class TestStopSignPositions:
    def test_no_stop_signs(self):
        sc = _make_scenario()
        assert _stop_sign_positions(sc) == []

    def test_one_stop_sign(self):
        sc = _make_scenario(stop_sign_xy=(5.0, 10.0))
        positions = _stop_sign_positions(sc)
        assert len(positions) == 1
        assert positions[0] == pytest.approx((5.0, 10.0))


# ---------------------------------------------------------------------------
# Tests: scenario_to_df — schema compliance
# ---------------------------------------------------------------------------

class TestScenarioToDf:
    def test_schema_columns_present(self):
        sc = _make_scenario()
        df = scenario_to_df(sc)
        expected_cols = {f.name for f in SCHEMA}
        assert expected_cols.issubset(set(df.columns))

    def test_correct_dtypes(self):
        sc = _make_scenario()
        df = scenario_to_df(sc)
        assert df["timestamp_ms"].dtype in (int, "int64")
        assert df["x"].dtype == float
        assert df["speed_mps"].dtype == float
        assert df["traffic_light_state"].dtype == object  # string
        assert df["stop_sign_zone"].dtype == bool
        assert df["in_intersection"].dtype == bool
        assert df["collision_flag"].dtype == bool

    def test_run_id_populated(self):
        sc = _make_scenario(scenario_id="abc123")
        df = scenario_to_df(sc)
        assert (df["run_id"] == "abc123").all()

    def test_agent_id_format(self):
        sc = _make_scenario(agents=[{"x": 0.0, "y": 0.0, "vx": 5.0, "vy": 0.0, "valid": True}])
        df = scenario_to_df(sc)
        assert df["agent_id"].iloc[0].startswith("agent_")

    def test_invalid_frames_dropped(self):
        """Frames with valid=False must be excluded."""
        sc = _make_scenario(
            n_steps=4,
            agents=[{"x": 0.0, "y": 0.0, "vx": 5.0, "vy": 0.0, "valid": True}],
        )
        # Make 2 of 4 states invalid
        sc.tracks[0].states[1] = _make_state(valid=False)
        sc.tracks[0].states[3] = _make_state(valid=False)
        df = scenario_to_df(sc)
        assert len(df) == 2

    def test_non_vehicle_agents_excluded(self):
        """Pedestrian (type=2) and cyclist (type=3) tracks must be filtered out."""
        sc = _make_scenario()
        sc.tracks.append(_make_track(track_id=99, states=sc.tracks[0].states[:], object_type=2))
        df = scenario_to_df(sc)
        agent_ids = df["agent_id"].unique()
        # Should only have agent_0 (vehicle), not agent_99 (pedestrian)
        assert "agent_99" not in agent_ids

    def test_speed_computed_from_velocity_components(self):
        vx, vy = 3.0, 4.0  # speed = 5.0
        sc = _make_scenario(agents=[{"x": 0.0, "y": 0.0, "vx": vx, "vy": vy, "valid": True}])
        df = scenario_to_df(sc)
        assert df["speed_mps"].iloc[0] == pytest.approx(5.0)

    def test_collision_flag_always_false(self):
        sc = _make_scenario()
        df = scenario_to_df(sc)
        assert not df["collision_flag"].any()

    def test_stop_sign_zone_true_when_near_stop_sign(self):
        # Agent at (10, 20), stop sign at (10, 20) → distance=0 → in zone
        sc = _make_scenario(
            agents=[{"x": 10.0, "y": 20.0, "vx": 0.0, "vy": 0.0, "valid": True}],
            stop_sign_xy=(10.0, 20.0),
        )
        df = scenario_to_df(sc)
        assert df["stop_sign_zone"].all()

    def test_stop_sign_zone_false_when_far(self):
        # Agent at (0, 0), stop sign at (100, 100) → far away → not in zone
        sc = _make_scenario(
            agents=[{"x": 0.0, "y": 0.0, "vx": 5.0, "vy": 0.0, "valid": True}],
            stop_sign_xy=(100.0, 100.0),
        )
        df = scenario_to_df(sc)
        assert not df["stop_sign_zone"].any()

    def test_red_light_state_propagated(self):
        # All steps RED (LANE_STATE_STOP = 4)
        sc = _make_scenario(signal_states=[4, 4, 4, 4, 4])
        df = scenario_to_df(sc)
        assert (df["traffic_light_state"] == "RED").all()

    def test_empty_scenario_returns_empty_df(self):
        sc = _make_scenario()
        sc.tracks = []  # no agents
        df = scenario_to_df(sc)
        assert df.empty
        assert set(df.columns) == {f.name for f in SCHEMA}

    def test_timestamp_ms_correct(self):
        # timestamps_seconds = [0.0, 0.1, 0.2] → ms = [0, 100, 200]
        sc = _make_scenario(n_steps=3)
        df = scenario_to_df(sc)
        ts = sorted(df["timestamp_ms"].unique())
        assert ts == [0, 100, 200]


# ---------------------------------------------------------------------------
# Tests: _score_scenario
# ---------------------------------------------------------------------------

class TestScoreScenario:
    def test_empty_df_returns_worst_score(self):
        df = pd.DataFrame(columns=[f.name for f in SCHEMA])
        score = _score_scenario(df)
        assert score["violation_rate"] == 1.0
        assert score["compliance"] == 0.0

    def test_no_violations_scores_zero(self):
        sc = _make_scenario(signal_states=[6] * 5)  # all GREEN
        df = scenario_to_df(sc)
        score = _score_scenario(df)
        assert score["violation_rate"] == 0.0


# ---------------------------------------------------------------------------
# Tests: select_scenarios
# ---------------------------------------------------------------------------

class TestSelectScenarios:
    def _compliant_df(self, run_id="golden"):
        sc = _make_scenario(
            scenario_id=run_id,
            agents=[{"x": 0.0, "y": 0.0, "vx": 5.0, "vy": 0.0, "valid": True}],
            signal_states=[6] * 5,
        )
        return scenario_to_df(sc)

    def _violating_df(self, run_id="regression"):
        # Agent inside intersection (no light positions → in_intersection=False by default,
        # so we just mark high speed at RED as a proxy)
        sc = _make_scenario(
            scenario_id=run_id,
            signal_states=[4] * 5,  # all RED
        )
        return scenario_to_df(sc)

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="No scenarios"):
            select_scenarios([])

    def test_returns_three_keys(self):
        dfs = [self._compliant_df(), self._violating_df()]
        result = select_scenarios(dfs)
        assert set(result.keys()) == {"womd_golden", "womd_regression", "womd_noisy"}

    def test_single_scenario_fills_all_three_slots(self):
        """When only one scenario available, all three labels point to the same data."""
        result = select_scenarios([self._compliant_df()])
        assert len(result) == 3
        for df in result.values():
            assert not df.empty
