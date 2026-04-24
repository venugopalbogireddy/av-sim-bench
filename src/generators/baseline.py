"""Ground-truth traffic distribution generator.

Produces samples from the canonical distributions used as the evaluation
baseline:
  speed_mps    — mixture of two Gaussians (free-flow + congested)
  gap_s        — exponential (memoryless headway)
  lane_changes — Poisson count per trip
  turn_choice  — multinomial draw (left / straight / right)

All public functions accept an optional numpy Generator for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── distribution parameters ───────────────────────────────────────────────────

SPEED_MU = [12.0, 5.0]          # free-flow, congested
SPEED_SIGMA = [2.0, 1.5]
SPEED_WEIGHTS = [0.7, 0.3]

GAP_MEAN_S = 2.0                 # exponential mean headway (λ = 1/mean)

LANE_CHANGE_LAMBDA = 3.0         # Poisson rate per trip

TURN_PROBS = [0.6, 0.25, 0.15]  # left, straight, right
TURN_LABELS = ["left", "straight", "right"]


def generate(
    n: int = 1000,
    rng: np.random.Generator | None = None,
    run_id: str = "baseline",
) -> pd.DataFrame:
    """Return a DataFrame with n rows of ground-truth traffic samples.

    Columns: run_id, speed_mps, gap_s, lane_changes, turn_choice
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # speed: mixture of Gaussians
    component = rng.choice(len(SPEED_WEIGHTS), size=n, p=SPEED_WEIGHTS)
    speed = np.array([
        rng.normal(SPEED_MU[c], SPEED_SIGMA[c]) for c in component
    ])
    speed = np.clip(speed, 0.0, None)  # physical lower bound

    # gap: exponential
    gap = rng.exponential(scale=GAP_MEAN_S, size=n)

    # lane changes: Poisson
    lane_changes = rng.poisson(lam=LANE_CHANGE_LAMBDA, size=n)

    # turn choice: multinomial → one draw per row
    turn_idx = rng.choice(len(TURN_PROBS), size=n, p=TURN_PROBS)
    turn_choice = [TURN_LABELS[i] for i in turn_idx]

    return pd.DataFrame({
        "run_id": run_id,
        "speed_mps": speed,
        "gap_s": gap,
        "lane_changes": lane_changes,
        "turn_choice": turn_choice,
    })
