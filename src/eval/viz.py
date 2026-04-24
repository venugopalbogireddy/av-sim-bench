"""Visualisation layer for A/B distribution comparisons.

Produces:
  - Histogram overlays (seaborn) per continuous metric
  - Empirical CDF comparison
  - Q-Q plot (candidate quantiles vs baseline quantiles)
  - 2×2 summary grid saved to outputs/p3/
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_TODAY = date.today().isoformat()
_OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "p3"

sns.set_theme(style="whitegrid", palette="muted")


def _hist_overlay(ax: plt.Axes, a: np.ndarray, b: np.ndarray, col: str, label_a: str, label_b: str) -> None:
    sns.histplot(a, ax=ax, label=label_a, stat="density", alpha=0.5, bins=30)
    sns.histplot(b, ax=ax, label=label_b, stat="density", alpha=0.5, bins=30, color="coral")
    ax.set_title(f"{col} — histogram")
    ax.set_xlabel(col)
    ax.legend()


def _ecdf(ax: plt.Axes, a: np.ndarray, b: np.ndarray, col: str, label_a: str, label_b: str) -> None:
    for arr, label in [(a, label_a), (b, label_b)]:
        sorted_arr = np.sort(arr)
        ecdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.step(sorted_arr, ecdf, where="post", label=label)
    ax.set_title(f"{col} — ECDF")
    ax.set_xlabel(col)
    ax.set_ylabel("F(x)")
    ax.legend()


def _qqplot(ax: plt.Axes, a: np.ndarray, b: np.ndarray, col: str, label_a: str, label_b: str) -> None:
    quantiles = np.linspace(0, 1, 100)
    qa = np.quantile(a, quantiles)
    qb = np.quantile(b, quantiles)
    ax.scatter(qa, qb, s=8, alpha=0.7)
    lo, hi = min(qa.min(), qb.min()), max(qa.max(), qb.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
    ax.set_title(f"{col} — Q-Q ({label_b} vs {label_a})")
    ax.set_xlabel(f"{label_a} quantiles")
    ax.set_ylabel(f"{label_b} quantiles")
    ax.legend()


def plot_summary(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    label: str,
    out_dir: Path | None = None,
) -> Path:
    """Generate a 3×3 summary grid (hist + ECDF + QQ for speed, gap, lane_changes).

    Saves PNG to outputs/p3/ and returns the path.
    """
    out = (out_dir or _OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["speed_mps", "gap_s", "lane_changes"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"A/B Distribution Comparison — baseline vs {label}", fontsize=14)

    for row_idx, col in enumerate(cols):
        a = baseline_df[col].to_numpy(dtype=float)
        b = candidate_df[col].to_numpy(dtype=float)
        _hist_overlay(axes[row_idx, 0], a, b, col, "baseline", label)
        _ecdf(axes[row_idx, 1], a, b, col, "baseline", label)
        _qqplot(axes[row_idx, 2], a, b, col, "baseline", label)

    plt.tight_layout()
    path = out / f"{_TODAY}-{label}-summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
