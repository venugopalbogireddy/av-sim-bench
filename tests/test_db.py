"""Unit tests for src/evaluator/db.py — LogStore."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evaluator.db import LogStore
from evaluator.schema import SCHEMA


# ── fixtures ───────────────────────────────────────────────────────────────────

def _make_record(
    run_id: str,
    ts: int = 0,
    agent: str = "agent_00",
    speed: float = 8.0,
) -> dict:
    return {
        "run_id": run_id,
        "timestamp_ms": ts,
        "agent_id": agent,
        "x": 0.0,
        "y": 0.0,
        "heading": 90.0,
        "speed_mps": speed,
        "traffic_light_state": "GREEN",
        "stop_sign_zone": False,
        "in_intersection": False,
        "collision_flag": False,
    }


def _write_parquet(path: Path, records: list[dict]) -> None:
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, path)


# ── expected schema after DuckDB round-trip ───────────────────────────────────

_EXPECTED_DTYPES: dict[str, str] = {
    "run_id":               "str",
    "timestamp_ms":         "int64",
    "agent_id":             "str",
    "x":                    "float64",
    "y":                    "float64",
    "heading":              "float64",
    "speed_mps":            "float64",
    "traffic_light_state":  "str",
    "stop_sign_zone":       "bool",
    "in_intersection":      "bool",
    "collision_flag":       "bool",
}


# ── test_from_dir_run_ids ─────────────────────────────────────────────────────

def test_from_dir_run_ids(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "golden_2026_01_01.parquet",     [_make_record("golden")])
    _write_parquet(tmp_path / "regression_2026_01_01.parquet", [_make_record("regression")])
    _write_parquet(tmp_path / "noisy_2026_01_01.parquet",      [_make_record("noisy")])

    with LogStore.from_dir(tmp_path) as store:
        ids = store.run_ids()

    assert ids == ["golden", "noisy", "regression"]  # sorted alphabetically


# ── test_load_run_returns_correct_rows ────────────────────────────────────────

def test_load_run_returns_correct_rows(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "golden_2026_01_01.parquet", [
        _make_record("golden", ts=0),
        _make_record("golden", ts=100),
    ])
    _write_parquet(tmp_path / "regression_2026_01_01.parquet", [
        _make_record("regression", ts=0),
    ])

    with LogStore.from_dir(tmp_path) as store:
        df = store.load_run("golden")

    assert len(df) == 2
    assert set(df["run_id"].unique()) == {"golden"}


# ── test_load_run_schema_columns ──────────────────────────────────────────────

def test_load_run_schema_columns(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "golden_2026_01_01.parquet", [_make_record("golden")])

    with LogStore.from_dir(tmp_path) as store:
        df = store.load_run("golden")

    # Exact column order must match the P1 schema
    assert list(df.columns) == list(_EXPECTED_DTYPES.keys())

    # Strict dtype check — this is the backbone for downstream metric functions
    for col, expected in _EXPECTED_DTYPES.items():
        actual = str(df[col].dtype)
        assert actual == expected, (
            f"Column '{col}': expected dtype '{expected}', got '{actual}'"
        )

    # filename metadata column must never leak into metric-facing DataFrames
    assert "filename" not in df.columns


# ── test_from_file_load_all ───────────────────────────────────────────────────

def test_from_file_load_all(tmp_path: Path) -> None:
    path = tmp_path / "golden_2026_01_01.parquet"
    records = [_make_record("golden", ts=i * 100) for i in range(5)]
    _write_parquet(path, records)

    with LogStore.from_file(path) as store:
        df = store.load_all()

    assert len(df) == 5
    assert list(df.columns) == list(_EXPECTED_DTYPES.keys())
    assert "filename" not in df.columns


# ── test_run_isolation ────────────────────────────────────────────────────────

def test_run_isolation(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "golden_2026_01_01.parquet", [
        _make_record("golden", ts=0),
        _make_record("golden", ts=100),
    ])
    _write_parquet(tmp_path / "regression_2026_01_01.parquet", [
        _make_record("regression", ts=0),
    ])

    with LogStore.from_dir(tmp_path) as store:
        df_g = store.load_run("golden")
        df_r = store.load_run("regression")

    assert set(df_g["run_id"].unique()) == {"golden"}
    assert set(df_r["run_id"].unique()) == {"regression"}
    # no cross-contamination
    assert len(df_g) == 2
    assert len(df_r) == 1


# ── test_run_file_map ─────────────────────────────────────────────────────────

def test_run_file_map(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "golden_2026_01_15.parquet",     [_make_record("golden")])
    _write_parquet(tmp_path / "regression_2026_01_20.parquet", [_make_record("regression")])

    with LogStore.from_dir(tmp_path) as store:
        fmap = store.run_file_map()

    assert fmap["golden"]     == "golden_2026_01_15"
    assert fmap["regression"] == "regression_2026_01_20"


# ── test_load_run_latest_only ─────────────────────────────────────────────────

def test_load_run_latest_only(tmp_path: Path) -> None:
    # Two historical golden files written on different dates
    _write_parquet(tmp_path / "golden_2026_01_01.parquet", [
        _make_record("golden", ts=0, speed=5.0),
    ])
    _write_parquet(tmp_path / "golden_2026_01_02.parquet", [
        _make_record("golden", ts=0, speed=9.0),
    ])

    with LogStore.from_dir(tmp_path) as store:
        df = store.load_run("golden")

    # Must return only the latest file — no double-counting across history
    assert len(df) == 1
    assert df["speed_mps"].iloc[0] == pytest.approx(9.0)
