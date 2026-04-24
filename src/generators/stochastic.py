"""Parametric stochastic traffic generator.

Two modes controlled by `mode`:
  "well_tuned"    — parameters close to the baseline; A/B eval should NOT
                    flag significant divergence (null holds).
  "miscalibrated" — parameters shifted enough that A/B eval should detect
                    divergence on at least speed and gap (regression case).

Both modes use the same distributional families as the baseline (Gaussian
mixture, exponential, Poisson, multinomial) so shape divergence is the
signal, not family mismatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# well-tuned: params close to baseline (small intentional drift)
_WELL_TUNED = {
    "speed_mu": [11.8, 5.2],
    "speed_sigma": [2.1, 1.4],
    "speed_weights": [0.68, 0.32],
    "gap_mean": 2.1,
    "lc_lambda": 3.1,
    "turn_probs": [0.58, 0.26, 0.16],
}

# miscalibrated: params shifted to trigger detection
_MISCALIBRATED = {
    "speed_mu": [14.0, 7.0],      # both modes faster
    "speed_sigma": [3.0, 2.0],    # wider spread
    "speed_weights": [0.5, 0.5],  # equal mode weights (vs 0.7/0.3 baseline)
    "gap_mean": 0.8,              # much tighter headway
    "lc_lambda": 8.0,             # 2.7× more lane changes
    "turn_probs": [0.3, 0.5, 0.2],  # shifted turn preference
}


def generate(
    n: int = 1000,
    rng: np.random.Generator | None = None,
    run_id: str | None = None,
    mode: str = "well_tuned",
) -> pd.DataFrame:
    """Return n stochastic traffic samples in the requested mode.

    Args:
        n:      Number of samples.
        rng:    Numpy Generator for reproducibility.
        run_id: Label for the run_id column (defaults to mode name).
        mode:   "well_tuned" or "miscalibrated".
    """
    if mode not in ("well_tuned", "miscalibrated"):
        raise ValueError(f"mode must be 'well_tuned' or 'miscalibrated', got {mode!r}")
    if rng is None:
        rng = np.random.default_rng(1)
    if run_id is None:
        run_id = mode

    p = _WELL_TUNED if mode == "well_tuned" else _MISCALIBRATED

    # speed: mixture of Gaussians
    component = rng.choice(len(p["speed_weights"]), size=n, p=p["speed_weights"])
    speed = np.array([
        rng.normal(p["speed_mu"][c], p["speed_sigma"][c]) for c in component
    ])
    speed = np.clip(speed, 0.0, None)

    gap = rng.exponential(scale=p["gap_mean"], size=n)
    lane_changes = rng.poisson(lam=p["lc_lambda"], size=n)

    turn_labels = ["left", "straight", "right"]
    turn_idx = rng.choice(len(p["turn_probs"]), size=n, p=p["turn_probs"])
    turn_choice = [turn_labels[i] for i in turn_idx]

    return pd.DataFrame({
        "run_id": run_id,
        "speed_mps": speed,
        "gap_s": gap,
        "lane_changes": lane_changes,
        "turn_choice": turn_choice,
    })
