"""Deterministic (rule-based) traffic generator.

Simulates a rigid rule-based controller that always targets a fixed speed,
uses a fixed headway, never changes lanes, and always turns straight. This
is intentionally miscalibrated vs the baseline — the A/B eval should
detect significant divergence on every metric dimension.

Used to validate that the stat-test pipeline correctly flags a bad generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIXED_SPEED_MPS = 10.0    # one speed for all vehicles
FIXED_GAP_S = 1.0         # fixed headway, no variance
FIXED_LANE_CHANGES = 0    # never changes lanes
FIXED_TURN = "straight"   # always goes straight


def generate(
    n: int = 1000,
    rng: np.random.Generator | None = None,  # accepted but unused — pure deterministic
    run_id: str = "deterministic",
) -> pd.DataFrame:
    """Return n rows of deterministic rule-based traffic samples.

    All agents drive at a fixed speed, fixed gap, zero lane changes, straight.
    Distributions are point masses — maximally different from the baseline
    mixture/stochastic distributions.
    """
    _ = rng  # deterministic: rng not used

    speed = np.full(n, FIXED_SPEED_MPS)
    gap = np.full(n, FIXED_GAP_S)
    lane_changes = np.zeros(n, dtype=int)
    turn_choice = [FIXED_TURN] * n

    return pd.DataFrame({
        "run_id": run_id,
        "speed_mps": speed,
        "gap_s": gap,
        "lane_changes": lane_changes,
        "turn_choice": turn_choice,
    })
