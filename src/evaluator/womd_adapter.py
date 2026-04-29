"""Waymo Open Motion Dataset (WOMD) → P1 Parquet adapter.

TF-FREE. Reads WOMD TFRecords using pure Python + google-protobuf only.
No tensorflow, no waymo-open-dataset pip package required.

One-time setup (run once to compile the protos):
    PYTHONPATH=src python -m evaluator.cli womd setup \
        --womd-src ../waymo-car/waymo-open-dataset/src

Then ingest:
    PYTHONPATH=src python -m evaluator.cli womd ingest \
        --input ../waymo-car/waymo-open-dataset/src/waymo_open_dataset/utils/testdata/ \
        --womd-src ../waymo-car/waymo-open-dataset/src \
        --out data/womd
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from evaluator.schema import SCHEMA

# ---------------------------------------------------------------------------
# Traffic-signal state mapping  (map.proto TrafficSignalLaneState.State enum)
# ---------------------------------------------------------------------------
_LANE_STATE_TO_STR: dict[int, str] = {
    0: "GREEN",    # LANE_STATE_UNKNOWN — default safe
    1: "RED",      # LANE_STATE_ARROW_STOP
    2: "YELLOW",   # LANE_STATE_ARROW_CAUTION
    3: "GREEN",    # LANE_STATE_ARROW_GO
    4: "RED",      # LANE_STATE_STOP
    5: "YELLOW",   # LANE_STATE_CAUTION
    6: "GREEN",    # LANE_STATE_GO
    7: "RED",      # LANE_STATE_FLASHING_STOP
    8: "YELLOW",   # LANE_STATE_FLASHING_CAUTION
}

# Proximity thresholds (metres) for derived spatial columns
_STOP_SIGN_RADIUS_M   = 15.0
_INTERSECTION_RADIUS_M = 20.0

_TYPE_VEHICLE = 1   # ObjectType.TYPE_VEHICLE


# ---------------------------------------------------------------------------
# One-time proto compilation
# ---------------------------------------------------------------------------

def compile_protos(womd_src: Path) -> None:
    """Compile WOMD .proto files into *_pb2.py using grpcio-tools.

    Only needs to run once. Writes _pb2.py files next to the .proto sources
    inside the womd_src tree.

    Args:
        womd_src: Path to waymo-open-dataset/src  (contains waymo_open_dataset/)
    """
    try:
        import grpc_tools.protoc  # noqa: F401
    except ImportError:
        raise ImportError(
            "grpcio-tools not installed.\n"
            "Run: pip install grpcio-tools"
        )

    import grpc_tools.protoc

    proto_dir = womd_src / "waymo_open_dataset" / "protos"
    root_dir  = womd_src / "waymo_open_dataset"

    # Protos at root level (dataset.proto, label.proto)
    root_protos = list(root_dir.glob("*.proto"))
    # Protos in protos/ subdirectory
    sub_protos = list(proto_dir.glob("*.proto"))

    all_protos = [str(p.relative_to(womd_src)) for p in root_protos + sub_protos]

    args = ["protoc", f"-I{womd_src}", f"--python_out={womd_src}"] + all_protos
    ret = grpc_tools.protoc.main(args)
    if ret != 0:
        raise RuntimeError(f"protoc compilation failed (exit {ret})")

    pb2_count = len(list(womd_src.rglob("*_pb2.py")))
    print(f"  ✓ Compiled {len(all_protos)} proto files → {pb2_count} _pb2.py files")


def _add_proto_path(womd_src: Path) -> None:
    """Add womd_src to sys.path so compiled *_pb2.py modules are importable."""
    src_str = str(womd_src.resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _check_protos_compiled(womd_src: Path) -> bool:
    """Return True if the two key pb2 files already exist."""
    scenario_pb2 = womd_src / "waymo_open_dataset" / "protos" / "scenario_pb2.py"
    return scenario_pb2.exists()


# ---------------------------------------------------------------------------
# TF-free TFRecord reader
# ---------------------------------------------------------------------------

def _read_raw_tfrecords(path: Path) -> list[bytes]:
    """Read all raw byte payloads from a TFRecord file.

    TFRecord wire format per record:
        uint64  length          (little-endian)
        uint32  masked_crc32(length_bytes)
        bytes   data            (serialized proto)
        uint32  masked_crc32(data)

    CRC validation is skipped for speed; corrupt files will surface as
    protobuf parse errors downstream.
    """
    records: list[bytes] = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if not hdr:
                break
            if len(hdr) < 8:
                break
            length = struct.unpack("<Q", hdr)[0]
            f.read(4)               # skip masked CRC of length
            data = f.read(length)
            f.read(4)               # skip masked CRC of data
            records.append(data)
    return records


# ---------------------------------------------------------------------------
# Core conversion — pure function, no I/O
# ---------------------------------------------------------------------------

def _build_signal_timeline(scenario: Any) -> list[str]:
    """Per-timestep dominant traffic signal: most restrictive across all lanes."""
    priority = {"RED": 2, "YELLOW": 1, "GREEN": 0}
    timeline: list[str] = []
    for dms in scenario.dynamic_map_states:
        dominant = "GREEN"
        for ls in dms.lane_states:
            s = _LANE_STATE_TO_STR.get(ls.state, "GREEN")
            if priority[s] > priority[dominant]:
                dominant = s
        timeline.append(dominant)
    return timeline


def _stop_sign_positions(scenario: Any) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    for feature in scenario.map_features:
        if feature.WhichOneof("feature_data") == "stop_sign":
            pos = feature.stop_sign.position
            positions.append((pos.x, pos.y))
    return positions


def _traffic_light_positions(scenario: Any) -> list[tuple[float, float]]:
    """Approximate intersection centres from signal-controlled lane polylines.

    WOMD MapFeature has no 'traffic_light' field. Instead, traffic signal
    info lives in dynamic_map_states → lane_states → lane ID. We collect all
    lane IDs that ever have a signal, then use the midpoint of each lane's
    polyline as the intersection proxy centre.
    """
    # Collect all lane IDs that appear in any traffic signal state
    signal_lane_ids: set[int] = set()
    for dms in scenario.dynamic_map_states:
        for ls in dms.lane_states:
            signal_lane_ids.add(ls.lane)

    if not signal_lane_ids:
        return []

    # Build lane_id → polyline lookup from map_features
    lane_map: dict[int, Any] = {}
    for feature in scenario.map_features:
        if feature.WhichOneof("feature_data") == "lane":
            lane_map[feature.id] = feature.lane

    # Use midpoint of each signal lane's polyline as its intersection position
    positions: list[tuple[float, float]] = []
    for lid in signal_lane_ids:
        if lid not in lane_map:
            continue
        pts = lane_map[lid].polyline
        if not pts:
            continue
        mid = len(pts) // 2
        positions.append((pts[mid].x, pts[mid].y))

    return positions


def _nearest_dist(x: float, y: float, pts: list[tuple[float, float]]) -> float:
    if not pts:
        return math.inf
    return min(math.sqrt((x - px) ** 2 + (y - py) ** 2) for px, py in pts)


def scenario_to_df(scenario: Any) -> pd.DataFrame:
    """Convert one WOMD Scenario proto → P1-schema DataFrame.

    Vehicle agents only. Invalid frames (state.valid == False) dropped.
    Returns empty DataFrame (with correct columns) if no vehicle data found.
    """
    run_id = scenario.scenario_id
    timestamps_ms = [int(t * 1000) for t in scenario.timestamps_seconds]
    n_steps = len(timestamps_ms)

    signal_timeline = _build_signal_timeline(scenario)
    if len(signal_timeline) < n_steps:
        signal_timeline += ["GREEN"] * (n_steps - len(signal_timeline))

    stop_positions  = _stop_sign_positions(scenario)
    light_positions = _traffic_light_positions(scenario)

    rows: list[dict] = []
    for track in scenario.tracks:
        if track.object_type != _TYPE_VEHICLE:
            continue

        agent_id = f"agent_{track.id}"
        for t_idx, state in enumerate(track.states):
            if not state.valid:
                continue

            x = float(state.center_x)
            y = float(state.center_y)
            speed_mps = math.sqrt(float(state.velocity_x) ** 2 + float(state.velocity_y) ** 2)
            ts_ms = timestamps_ms[t_idx] if t_idx < n_steps else t_idx * 100
            tl    = signal_timeline[t_idx] if t_idx < len(signal_timeline) else "GREEN"

            rows.append({
                "run_id":              run_id,
                "timestamp_ms":        ts_ms,
                "agent_id":            agent_id,
                "x":                   x,
                "y":                   y,
                "heading":             float(state.heading),
                "speed_mps":           speed_mps,
                "traffic_light_state": tl,
                "stop_sign_zone":      _nearest_dist(x, y, stop_positions)  <= _STOP_SIGN_RADIUS_M,
                "in_intersection":     _nearest_dist(x, y, light_positions) <= _INTERSECTION_RADIUS_M,
                "collision_flag":      False,
            })

    if not rows:
        return pd.DataFrame(columns=[f.name for f in SCHEMA])
    return pd.DataFrame(rows)


def _score_scenario(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"violation_rate": 1.0, "compliance": 0.0, "speed_std": 0.0}

    in_int = df[df["in_intersection"]]
    if not in_int.empty:
        violating = in_int[(in_int["traffic_light_state"] == "RED") & (in_int["speed_mps"] > 0.1)]
        violation_rate = len(violating["agent_id"].unique()) / max(len(df["agent_id"].unique()), 1)
    else:
        violation_rate = 0.0

    in_zone = df[df["stop_sign_zone"]]
    compliance = float((in_zone.groupby("agent_id")["speed_mps"].min() < 0.5).mean()) if not in_zone.empty else 1.0
    speed_std  = float(df["speed_mps"].std()) if len(df) > 1 else 0.0

    return {"violation_rate": violation_rate, "compliance": compliance, "speed_std": speed_std}


def select_scenarios(dfs: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Pick best golden / regression / noisy from a list of scenario DataFrames."""
    if not dfs:
        raise ValueError("No scenarios to select from")

    scores = [_score_scenario(df) for df in dfs]

    golden_idx     = min(range(len(scores)), key=lambda i: (scores[i]["violation_rate"], -scores[i]["compliance"]))
    regression_idx = max(range(len(scores)), key=lambda i: scores[i]["violation_rate"])
    remaining      = [i for i in range(len(dfs)) if i != regression_idx]
    noisy_idx      = max(remaining, key=lambda i: scores[i]["speed_std"]) if remaining else regression_idx

    return {
        "womd_golden":     dfs[golden_idx],
        "womd_regression": dfs[regression_idx],
        "womd_noisy":      dfs[noisy_idx],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_tfrecord_dir(
    input_dir: Path,
    out_dir: Path,
    womd_src: Path,
    glob: str = "*.tfrecord",
    max_scenarios: int = 50,
) -> dict[str, Path]:
    """Convert WOMD TFRecords → 3 P1 Parquet files (golden / regression / noisy).

    Args:
        input_dir:     Directory with .tfrecord files.
        out_dir:       Where to write Parquet output.
        womd_src:      Path to waymo-open-dataset/src (must have compiled _pb2.py files).
        glob:          File pattern (default *.tfrecord).
        max_scenarios: Cap on scenarios parsed.

    Returns:
        Dict label → output Path.
    """
    _add_proto_path(womd_src)

    if not _check_protos_compiled(womd_src):
        raise RuntimeError(
            f"Proto files not compiled yet in {womd_src}.\n"
            "Run: PYTHONPATH=src python -m evaluator.cli womd setup --womd-src <path>"
        )

    from waymo_open_dataset.protos import scenario_pb2  # type: ignore[import]

    out_dir.mkdir(parents=True, exist_ok=True)
    tf_files = sorted(input_dir.glob(glob))
    if not tf_files:
        raise FileNotFoundError(f"No TFRecord files matching '{glob}' in {input_dir}")

    print(f"Found {len(tf_files)} TFRecord file(s)")
    all_dfs: list[pd.DataFrame] = []

    for tf_path in tf_files:
        print(f"  Reading {tf_path.name} …")
        for raw in _read_raw_tfrecords(tf_path):
            sc = scenario_pb2.Scenario()
            try:
                sc.ParseFromString(raw)
            except Exception:
                # File may contain non-Scenario protos (perception frames etc.) — skip
                continue
            df = scenario_to_df(sc)
            if not df.empty:
                all_dfs.append(df)
            if len(all_dfs) >= max_scenarios:
                break
        if len(all_dfs) >= max_scenarios:
            break

    if not all_dfs:
        raise RuntimeError("No valid vehicle data found in TFRecords.")

    print(f"Converted {len(all_dfs)} scenario(s) — selecting golden/regression/noisy …")
    selected = select_scenarios(all_dfs)

    written: dict[str, Path] = {}
    for label, df in selected.items():
        df = df.copy()
        df["run_id"] = label
        table    = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
        out_path = out_dir / f"{label}.parquet"
        pq.write_table(table, out_path)
        print(f"  ✓ {label}: {len(df):,} rows → {out_path}")
        written[label] = out_path

    return written
