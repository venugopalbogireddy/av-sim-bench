"""Synthetic driving log generator — produces Parquet files for 3 scenario types."""

from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from evaluator.schema import SCHEMA, STOP_ZONE_Y, INTERSECTION_Y
from evaluator.metrics import STOP_SPEED_THRESHOLD, STOP_DWELL_FRAMES

RNG = np.random.default_rng(42)

_TIMESTEPS = 500          # 5 seconds at 100 ms resolution
_NUM_AGENTS = 4
_GOAL_Y = 50.0            # agents must reach y >= 50 to complete route


def _light_sequence(n: int) -> list[str]:
    """Traffic-light cycle that ensures RED is active when agents cross the intersection.

    Agents cross y=22-28 (intersection) around step 155-200 after crawling through the
    stop zone.  Starting with RED(200) guarantees the crossing happens under RED, so
    golden agents wait and regression agents can meaningfully violate.
    """
    cycle = ["RED"] * 200 + ["GREEN"] * 250 + ["YELLOW"] * 50
    return [cycle[i % len(cycle)] for i in range(n)]


def _golden(num_agents: int = _NUM_AGENTS) -> pd.DataFrame:
    """Well-behaved: full stop at stop signs, obeys red lights, no collisions."""
    lights = _light_sequence(_TIMESTEPS)
    records = []
    for agent_idx in range(num_agents):
        agent_id = f"agent_{agent_idx:02d}"
        y = 0.0
        dwell_count = 0  # consecutive frames held at full stop inside the stop zone
        for t in range(_TIMESTEPS):
            ts = t * 100
            light = lights[t]
            in_stop = STOP_ZONE_Y[0] <= y <= STOP_ZONE_Y[1]
            in_intersection = INTERSECTION_Y[0] <= y <= INTERSECTION_Y[1]

            if in_stop:
                if dwell_count < STOP_DWELL_FRAMES:
                    speed = STOP_SPEED_THRESHOLD  # hold full stop for required dwell period
                    dwell_count += 1
                else:
                    speed = 8.0 + RNG.normal(0, 0.05)  # dwell complete — proceed through zone
            elif in_intersection and light == "RED":
                speed = 0.0                       # waiting at red
            else:
                speed = 8.0 + RNG.normal(0, 0.05)  # nominal cruise

            y = min(y + speed * 0.1, _GOAL_Y + 5)
            records.append({
                "run_id": "golden",
                "timestamp_ms": ts,
                "agent_id": agent_id,
                "x": float(agent_idx * 3),
                "y": round(y, 4),
                "heading": 90.0,
                "speed_mps": round(float(speed), 4),
                "traffic_light_state": light,
                "stop_sign_zone": in_stop,
                "in_intersection": in_intersection,
                "collision_flag": False,
            })
    return pd.DataFrame(records)


def _regression(num_agents: int = _NUM_AGENTS) -> pd.DataFrame:
    """Bad actor: runs a red light (agent_00) and rolls a stop sign (agent_01)."""
    lights = _light_sequence(_TIMESTEPS)
    records = []
    for agent_idx in range(num_agents):
        agent_id = f"agent_{agent_idx:02d}"
        y = 0.0
        dwell_count = 0  # only used by compliant agents (not agent_00 or agent_01)
        for t in range(_TIMESTEPS):
            ts = t * 100
            light = lights[t]
            in_stop = STOP_ZONE_Y[0] <= y <= STOP_ZONE_Y[1]
            in_intersection = INTERSECTION_Y[0] <= y <= INTERSECTION_Y[1]

            if agent_idx == 0 and in_intersection and light == "RED":
                # agent_00 blows the red light — no stop at all
                speed = 7.5
            elif agent_idx == 1 and in_stop:
                # agent_01 rolls the stop sign — never reaches full stop
                speed = 2.0
            elif in_stop:
                # compliant agents: dwell then proceed
                if dwell_count < STOP_DWELL_FRAMES:
                    speed = STOP_SPEED_THRESHOLD
                    dwell_count += 1
                else:
                    speed = 8.0 + RNG.normal(0, 0.05)
            elif in_intersection and light == "RED":
                speed = 0.0
            else:
                speed = 8.0 + RNG.normal(0, 0.05)

            y = min(y + speed * 0.1, _GOAL_Y + 5)
            collision = agent_idx == 0 and 24.0 <= y <= 26.0 and light == "RED"
            records.append({
                "run_id": "regression",
                "timestamp_ms": ts,
                "agent_id": agent_id,
                "x": float(agent_idx * 3),
                "y": round(y, 4),
                "heading": 90.0,
                "speed_mps": round(float(speed), 4),
                "traffic_light_state": light,
                "stop_sign_zone": in_stop,
                "in_intersection": in_intersection,
                "collision_flag": bool(collision),
            })
    return pd.DataFrame(records)


def _noisy(num_agents: int = _NUM_AGENTS) -> pd.DataFrame:
    """Slight speed jitter at cruise; full stop at signs and lights — no violations."""
    lights = _light_sequence(_TIMESTEPS)
    records = []
    for agent_idx in range(num_agents):
        agent_id = f"agent_{agent_idx:02d}"
        y = 0.0
        dwell_count = 0  # consecutive frames held at full stop inside the stop zone
        for t in range(_TIMESTEPS):
            ts = t * 100
            light = lights[t]
            in_stop = STOP_ZONE_Y[0] <= y <= STOP_ZONE_Y[1]
            in_intersection = INTERSECTION_Y[0] <= y <= INTERSECTION_Y[1]

            if in_stop:
                if dwell_count < STOP_DWELL_FRAMES:
                    speed = STOP_SPEED_THRESHOLD  # hold full stop for dwell period
                    dwell_count += 1
                else:
                    speed = 8.0 + RNG.normal(0, 1.2)  # dwell complete — proceed with jitter
            elif in_intersection and light == "RED":
                speed = 0.0
            else:
                speed = 8.0 + RNG.normal(0, 1.2)  # higher variance than golden

            y = min(y + max(speed, 0) * 0.1, _GOAL_Y + 5)
            records.append({
                "run_id": "noisy",
                "timestamp_ms": ts,
                "agent_id": agent_id,
                "x": float(agent_idx * 3),
                "y": round(y, 4),
                "heading": 90.0,
                "speed_mps": round(float(max(speed, 0)), 4),
                "traffic_light_state": light,
                "stop_sign_zone": in_stop,
                "in_intersection": in_intersection,
                "collision_flag": False,
            })
    return pd.DataFrame(records)


def generate_all(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime("%Y_%m_%d")
    scenarios = {"golden": _golden, "regression": _regression, "noisy": _noisy}
    paths: dict[str, Path] = {}
    for name, fn in scenarios.items():
        df = fn()
        path = out_dir / f"{name}_{today}.parquet"
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        paths[name] = path
        print(f"  wrote {path} ({len(df):,} rows)")
    return paths


if __name__ == "__main__":
    generate_all(Path("data"))
