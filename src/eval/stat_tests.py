"""Statistical tests for comparing two traffic sample distributions.

Each test function takes two array-likes and returns a TestResult.
The module-level `run_all_tests` function runs all 5 tests on a pair of
DataFrames, applies BH-FDR correction, and returns the corrected results.

Tests:
  1. Kolmogorov-Smirnov  — overall CDF shape
  2. Wasserstein (EMD)   — distributional distance (magnitude, not p-value)
  3. Chi-square          — binned frequency comparison
  4. Anderson-Darling    — tail-sensitive two-sample test
  5. Energy distance     — scalar divergence (complements KS)

Multiple-testing correction: Benjamini-Hochberg FDR at α=0.05.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

ALPHA = 0.05


@dataclass
class TestResult:
    test_name: str
    metric: str          # which column was tested (e.g. "speed_mps")
    statistic: float
    p_value: float       # NaN for Wasserstein/energy (no analytic p-value)
    significant: bool    # after BH-FDR correction
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.significant = bool(self.significant)
        self.statistic = float(self.statistic)
        self.p_value = float(self.p_value)


# ── individual test functions ────────────────────────────────────────────────

def ks_test(a: np.ndarray, b: np.ndarray, metric: str) -> TestResult:
    stat, p = stats.ks_2samp(a, b)
    return TestResult("ks", metric, stat, p, significant=False)


def wasserstein_test(a: np.ndarray, b: np.ndarray, metric: str) -> TestResult:
    dist = float(stats.wasserstein_distance(a, b))
    # Wasserstein has no analytic p-value; use permutation bootstrap for significance
    rng = np.random.default_rng(42)
    combined = np.concatenate([a, b])
    n_a = len(a)
    n_perm = 500
    null = np.array([
        stats.wasserstein_distance(
            rng.permutation(combined)[:n_a],
            rng.permutation(combined)[n_a:],
        )
        for _ in range(n_perm)
    ])
    p = float((null >= dist).mean())
    return TestResult("wasserstein", metric, dist, p, significant=False)


def chisquare_test(a: np.ndarray, b: np.ndarray, metric: str) -> TestResult:
    # bin into 10 equal-width bins covering combined range
    combined = np.concatenate([a, b])
    bins = np.linspace(combined.min(), combined.max(), 11)
    obs_a, _ = np.histogram(a, bins=bins)
    obs_b, _ = np.histogram(b, bins=bins)
    # add small epsilon to avoid zero cells
    obs_a = obs_a + 1e-6
    obs_b = obs_b + 1e-6
    # chi-square goodness-of-fit: compare obs_a vs expected scaled from obs_b
    scale = obs_a.sum() / obs_b.sum()
    expected = obs_b * scale
    stat, p = stats.chisquare(obs_a, f_exp=expected)
    return TestResult("chisquare", metric, float(stat), float(p), significant=False)


def anderson_darling_test(a: np.ndarray, b: np.ndarray, metric: str) -> TestResult:
    result = stats.anderson_ksamp([a, b])
    # anderson_ksamp returns significance_level as approximate p-value
    stat = float(result.statistic)
    p = float(result.pvalue)
    return TestResult("anderson_darling", metric, stat, p, significant=False)


def energy_distance_test(a: np.ndarray, b: np.ndarray, metric: str) -> TestResult:
    dist = float(stats.energy_distance(a, b))
    # permutation bootstrap for p-value
    rng = np.random.default_rng(99)
    combined = np.concatenate([a, b])
    n_a = len(a)
    n_perm = 500
    null = np.array([
        stats.energy_distance(
            rng.permutation(combined)[:n_a],
            rng.permutation(combined)[n_a:],
        )
        for _ in range(n_perm)
    ])
    p = float((null >= dist).mean())
    return TestResult("energy_distance", metric, dist, p, significant=False)


# ── multi-metric, multi-test runner with BH-FDR ─────────────────────────────

_CONTINUOUS_METRICS = ("speed_mps", "gap_s")
_COUNT_METRICS = ("lane_changes",)
_CATEGORICAL_METRICS = ("turn_choice",)

_CONTINUOUS_TESTS = [ks_test, wasserstein_test, anderson_darling_test, energy_distance_test]
_COUNT_TESTS = [ks_test, chisquare_test, wasserstein_test]
_CATEGORICAL_TESTS = [chisquare_test]


def _encode_categorical(series: pd.Series) -> np.ndarray:
    """Map string categories to ints for distance-based tests."""
    cats = sorted(series.unique())
    mapping = {c: i for i, c in enumerate(cats)}
    return series.map(mapping).to_numpy(dtype=float)


def run_all_tests(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    alpha: float = ALPHA,
) -> list[TestResult]:
    """Run all applicable tests across all metric dimensions; apply BH-FDR.

    Returns a flat list of TestResult objects with `significant` set after
    Benjamini-Hochberg correction.
    """
    results: list[TestResult] = []

    for col in _CONTINUOUS_METRICS:
        a = baseline[col].to_numpy(dtype=float)
        b = candidate[col].to_numpy(dtype=float)
        for fn in _CONTINUOUS_TESTS:
            results.append(fn(a, b, col))

    for col in _COUNT_METRICS:
        a = baseline[col].to_numpy(dtype=float)
        b = candidate[col].to_numpy(dtype=float)
        for fn in _COUNT_TESTS:
            results.append(fn(a, b, col))

    for col in _CATEGORICAL_METRICS:
        a = _encode_categorical(baseline[col])
        b = _encode_categorical(candidate[col])
        results.append(chisquare_test(a, b, col))

    _apply_bh_fdr(results, alpha)
    return results


def _apply_bh_fdr(results: list[TestResult], alpha: float) -> None:
    """Apply Benjamini-Hochberg correction in-place."""
    m = len(results)
    if m == 0:
        return

    # sort by p-value ascending
    indexed = sorted(enumerate(results), key=lambda t: t[1].p_value)
    for rank, (orig_idx, r) in enumerate(indexed, start=1):
        threshold = (rank / m) * alpha
        results[orig_idx].significant = bool(r.p_value <= threshold)
