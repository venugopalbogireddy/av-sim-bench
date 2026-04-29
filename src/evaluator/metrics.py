"""5 evaluation metrics for simulated driving logs.

Each metric is a pure function: DataFrame -> MetricResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class MetricResult:
    name: str
    value: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.passed = bool(self.passed)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "details": self.details,
        }


# ── 1. Stop-sign compliance ───────────────────────────────────────────────────

STOP_SPEED_THRESHOLD = 0.0   # m/s — agent must reach a full stop (speed == 0)
STOP_DWELL_FRAMES   = 10     # consecutive frames at 0 m/s required (~1 second at 100ms resolution)


def _has_dwell(group: pd.DataFrame) -> bool:
    """Return True if the agent held speed <= 0 for at least STOP_DWELL_FRAMES consecutive frames.

    Uses a rolling sum over a boolean series — a window of STOP_DWELL_FRAMES that sums
    to STOP_DWELL_FRAMES means every frame in that window was at full stop.
    """
    at_zero = (
        group.sort_values("timestamp_ms")["speed_mps"]
        .le(STOP_SPEED_THRESHOLD)
    )
    return bool(at_zero.rolling(STOP_DWELL_FRAMES).sum().eq(STOP_DWELL_FRAMES).any())


def stop_sign_compliance(df: pd.DataFrame) -> MetricResult:
    """Fraction of agents that held a full stop for >= STOP_DWELL_FRAMES consecutive frames
    inside the stop-sign zone.

    Mirrors the real-world legal standard: a complete stop means coming to rest (0 m/s)
    and holding that rest for at least 1 second — not just a momentary speed dip.
    Pass criterion: all agents (rate == 1.0) must satisfy the dwell requirement.
    """
    in_zone = df[df["stop_sign_zone"]]
    if in_zone.empty:
        return MetricResult("stop_sign_compliance", 1.0, True,
                            {"note": "no stop-sign zones in log"})

    compliant = in_zone.groupby("agent_id").apply(_has_dwell, include_groups=False)
    rate = float(compliant.mean())
    return MetricResult(
        "stop_sign_compliance",
        round(rate, 4),
        rate == 1.0,
        {
            "compliance_per_agent": compliant.to_dict(),
            "threshold_mps": STOP_SPEED_THRESHOLD,
            "required_dwell_frames": STOP_DWELL_FRAMES,
        },
    )


# ── 2. Red-light violation rate ───────────────────────────────────────────────

def _entered_on_green(df: pd.DataFrame) -> pd.Index:
    """Return row indices of in-intersection frames for agents that entered on GREEN.

    For each agent, detects every contiguous intersection traversal by finding
    False→True transitions on `in_intersection`. If the light was GREEN at the
    entry frame of a traversal, all rows in that traversal are exempt from the
    red-light violation filter — the agent entered legally and should not be
    penalised if the signal flips RED mid-crossing.

    Handles multiple crossings per agent: each traversal is evaluated independently.
    """
    exempt: list = []
    for _, agent_df in df.groupby("agent_id"):
        s = agent_df.sort_values("timestamp_ms")
        in_zone = s["in_intersection"]
        # False→True transition marks the start of a new traversal
        is_entry = in_zone & ~in_zone.shift(1, fill_value=False)
        # cumsum gives a monotonically increasing ID per traversal
        crossing_id = is_entry.cumsum()
        for _, crossing in s[in_zone].groupby(crossing_id[in_zone]):
            if crossing.iloc[0]["traffic_light_state"] == "GREEN":
                exempt.extend(crossing.index.tolist())
    return pd.Index(exempt)


def red_light_violation_rate(df: pd.DataFrame) -> MetricResult:
    """Fraction of agents that drove through the intersection while light == RED.

    Uses the `in_intersection` boolean field from the log schema rather than
    deriving position from hardcoded y-bounds — keeping metric logic map-agnostic.
    A violation requires the agent to be inside the intersection AND moving (speed > 0)
    while the light is RED. An agent stopped inside the intersection waiting for green
    is NOT a violation.

    Agents that entered the intersection on GREEN are exempt even if the light flips
    RED mid-crossing (SG-06). Only agents whose first in-intersection frame is RED
    (or YELLOW) are subject to the violation filter.
    """
    exempt = _entered_on_green(df)

    candidate_rows = df[
        df["in_intersection"]
        & (df["traffic_light_state"] == "RED")
        & (df["speed_mps"] > STOP_SPEED_THRESHOLD)
    ]
    violating_rows = candidate_rows[~candidate_rows.index.isin(exempt)]

    all_agents = df["agent_id"].unique()
    violators = violating_rows["agent_id"].unique()
    rate = len(violators) / max(len(all_agents), 1)
    return MetricResult(
        "red_light_violation_rate",
        round(rate, 4),
        rate == 0.0,
        {"violating_agents": list(violators), "total_agents": len(all_agents)},
    )


# ── 3. Collision proxy ────────────────────────────────────────────────────────

def collision_proxy(df: pd.DataFrame) -> MetricResult:
    """Count of rows where collision_flag == True."""
    count = int(df["collision_flag"].sum())
    return MetricResult(
        "collision_proxy",
        float(count),
        count == 0,
        {"collision_rows": count},
    )


# ── 4. Route completion % ─────────────────────────────────────────────────────

GOAL_Y = 50.0

def route_completion(df: pd.DataFrame) -> MetricResult:
    """Fraction of agents whose maximum y ever reached GOAL_Y."""
    reached = (
        df.groupby("agent_id")["y"]
        .max()
        .ge(GOAL_Y)
    )
    rate = float(reached.mean())
    return MetricResult(
        "route_completion",
        round(rate, 4),
        rate == 1.0,
        {"reached_goal": reached.to_dict(), "goal_y": GOAL_Y},
    )


# ── 5. Speed-distribution KS test ────────────────────────────────────────────

KS_PVALUE_THRESHOLD = 0.05   # p < threshold → distributions differ → flag drift

def speed_ks_test(df: pd.DataFrame, baseline_df: pd.DataFrame) -> MetricResult:
    """Two-sample KS test comparing cruise speeds vs. baseline."""
    def _cruise(d: pd.DataFrame) -> np.ndarray:
        return d.loc[~d["stop_sign_zone"] & (d["traffic_light_state"] != "RED"),
                     "speed_mps"].dropna().to_numpy()

    run_speeds = _cruise(df)
    base_speeds = _cruise(baseline_df)

    if run_speeds.size == 0 or base_speeds.size == 0:
        return MetricResult("speed_ks_test", 1.0, True,
                            {"note": "insufficient cruise data"})

    stat, pvalue = ks_2samp(run_speeds, base_speeds)
    passed = pvalue >= KS_PVALUE_THRESHOLD
    return MetricResult(
        "speed_ks_test",
        round(float(pvalue), 6),
        passed,
        {
            "ks_statistic": round(float(stat), 6),
            "p_value": round(float(pvalue), 6),
            "threshold": KS_PVALUE_THRESHOLD,
            "interpretation": "no drift" if passed else "speed distribution drift detected",
        },
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame, baseline_df: pd.DataFrame) -> list[MetricResult]:
    return [
        stop_sign_compliance(df),
        red_light_violation_rate(df),
        collision_proxy(df),
        route_completion(df),
        speed_ks_test(df, baseline_df),
    ]
