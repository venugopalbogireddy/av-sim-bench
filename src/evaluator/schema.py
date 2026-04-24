"""Canonical Parquet schema and map geometry constants for the sim-log evaluator.

Centralising the schema here means log_gen.py, metrics.py, and any future
readers all agree on column names, types, and zone boundaries without
duplicating constants.
"""

import pyarrow as pa

# ---------------------------------------------------------------------------
# Map geometry — defines the physical zones agents interact with.
# These bounds must match the scenario map used during log generation.
# If the map changes, update here and re-generate logs.
# ---------------------------------------------------------------------------

STOP_ZONE_Y: tuple[float, float] = (18.0, 22.0)      # y-range of the stop-sign zone
INTERSECTION_Y: tuple[float, float] = (22.0, 28.0)   # y-range of the signalised intersection

# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

# P1 schema — continuous (x, y) coordinate logs
SCHEMA = pa.schema([
    pa.field("run_id",              pa.string()),
    pa.field("timestamp_ms",        pa.int64()),
    pa.field("agent_id",            pa.string()),
    pa.field("x",                   pa.float64()),
    pa.field("y",                   pa.float64()),
    pa.field("heading",             pa.float64()),
    pa.field("speed_mps",           pa.float64()),
    pa.field("traffic_light_state", pa.string()),    # GREEN | YELLOW | RED
    pa.field("stop_sign_zone",      pa.bool_()),     # True when agent is inside STOP_ZONE_Y
    pa.field("in_intersection",     pa.bool_()),     # True when agent is inside INTERSECTION_Y
    pa.field("collision_flag",      pa.bool_()),
])

# P2 schema — graph-based logs; adds node_id (encodes x,y) and action.
# x, y are derived from node coordinates so P1 metrics that use them still work.
P2_SCHEMA = pa.schema([
    pa.field("run_id",               pa.string()),
    pa.field("timestamp_ms",         pa.int64()),
    pa.field("agent_id",             pa.string()),
    pa.field("node_id",              pa.string()),   # "node_x_y" — graph position
    pa.field("x",                    pa.float64()),   # grid col (matches node_id)
    pa.field("y",                    pa.float64()),   # grid row (matches node_id)
    pa.field("heading",              pa.float64()),
    pa.field("speed_mps",            pa.float64()),
    pa.field("traffic_light_state",  pa.string()),
    pa.field("stop_sign_zone",       pa.bool_()),
    pa.field("in_intersection",      pa.bool_()),
    pa.field("collision_flag",       pa.bool_()),
    pa.field("action",               pa.string()),   # accelerate | brake | hold | turn
])
