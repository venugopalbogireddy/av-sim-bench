"""Unit tests for src/eval/stat_tests.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.stat_tests import (
    ALPHA,
    TestResult,
    _apply_bh_fdr,
    anderson_darling_test,
    chisquare_test,
    energy_distance_test,
    ks_test,
    run_all_tests,
    wasserstein_test,
)

RNG = np.random.default_rng(0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _identical(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    a = RNG.normal(10, 2, n)
    return a, a.copy()


def _different(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    a = RNG.normal(10, 2, n)
    b = RNG.normal(20, 2, n)   # clearly distinct
    return a, b


# ── TestResult dataclass ──────────────────────────────────────────────────────

def test_test_result_types():
    r = TestResult("ks", "speed_mps", statistic=np.float64(0.05), p_value=np.float64(0.5), significant=np.bool_(True))
    assert isinstance(r.significant, bool)
    assert isinstance(r.statistic, float)
    assert isinstance(r.p_value, float)


# ── individual tests — same distribution (should NOT be significant) ──────────

def test_ks_identical_not_significant():
    a, b = _identical()
    r = ks_test(a, b, "speed_mps")
    assert r.test_name == "ks"
    assert r.metric == "speed_mps"
    assert r.p_value > ALPHA


def test_ks_different_low_pvalue():
    a, b = _different()
    r = ks_test(a, b, "speed_mps")
    assert r.p_value < 0.01


def test_wasserstein_identical_near_zero():
    a, b = _identical(300)
    r = wasserstein_test(a, b, "speed_mps")
    assert r.statistic < 0.5


def test_wasserstein_different_large_stat():
    a, b = _different(300)
    r = wasserstein_test(a, b, "speed_mps")
    assert r.statistic > 5.0


def test_chisquare_identical_not_significant():
    a, b = _identical()
    r = chisquare_test(a, b, "speed_mps")
    assert r.p_value > 0.01


def test_chisquare_different_low_pvalue():
    a, b = _different()
    r = chisquare_test(a, b, "speed_mps")
    assert r.p_value < 0.05


def test_anderson_darling_different_low_pvalue():
    a, b = _different()
    r = anderson_darling_test(a, b, "speed_mps")
    assert r.p_value < 0.05


def test_energy_distance_identical_near_zero():
    a, b = _identical(300)
    r = energy_distance_test(a, b, "speed_mps")
    assert r.statistic < 1.0


def test_energy_distance_different_large():
    a, b = _different(300)
    r = energy_distance_test(a, b, "speed_mps")
    assert r.statistic > 3.0


# ── BH-FDR correction ────────────────────────────────────────────────────────

def test_bh_fdr_all_high_pvalues():
    results = [TestResult("ks", "x", 0.01, p, False) for p in [0.5, 0.6, 0.7, 0.8, 0.9]]
    _apply_bh_fdr(results, alpha=0.05)
    assert all(not r.significant for r in results)


def test_bh_fdr_all_very_low_pvalues():
    results = [TestResult("ks", "x", 0.5, p, False) for p in [0.0001, 0.0002, 0.0003]]
    _apply_bh_fdr(results, alpha=0.05)
    assert all(r.significant for r in results)


def test_bh_fdr_mixed():
    pvals = [0.001, 0.01, 0.04, 0.5, 0.9]
    results = [TestResult("ks", "x", 0.5, p, False) for p in pvals]
    _apply_bh_fdr(results, alpha=0.05)
    n_sig = sum(r.significant for r in results)
    assert 1 <= n_sig <= 3


def test_bh_fdr_empty():
    _apply_bh_fdr([], alpha=0.05)  # must not raise


# ── run_all_tests integration ─────────────────────────────────────────────────

def _make_df(run_id: str, speed: np.ndarray, gap: np.ndarray,
             lc: np.ndarray, turn: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "run_id": run_id,
        "speed_mps": speed,
        "gap_s": gap,
        "lane_changes": lc,
        "turn_choice": turn,
    })


def test_run_all_tests_identical_no_significance():
    n = 400
    speed = RNG.normal(12, 2, n)
    gap = RNG.exponential(2.0, n)
    lc = RNG.poisson(3, n)
    turns = RNG.choice(["left", "straight", "right"], n, p=[0.6, 0.25, 0.15]).tolist()
    base = _make_df("baseline", speed, gap, lc, turns)
    cand = _make_df("cand", speed.copy(), gap.copy(), lc.copy(), turns.copy())
    results = run_all_tests(base, cand)
    assert isinstance(results, list)
    assert len(results) > 0
    # with identical data almost nothing should be significant
    n_sig = sum(r.significant for r in results)
    assert n_sig <= 2


def test_run_all_tests_very_different_detects_divergence():
    n = 500
    base = _make_df(
        "baseline",
        RNG.normal(12, 2, n),
        RNG.exponential(2.0, n),
        RNG.poisson(3, n),
        RNG.choice(["left", "straight", "right"], n, p=[0.6, 0.25, 0.15]).tolist(),
    )
    cand = _make_df(
        "cand",
        np.full(n, 25.0),   # deterministic point mass — maximal divergence
        np.full(n, 0.1),
        np.zeros(n, dtype=int),
        ["straight"] * n,
    )
    results = run_all_tests(base, cand)
    n_sig = sum(r.significant for r in results)
    assert n_sig >= 3


def test_run_all_tests_returns_test_result_objects():
    n = 200
    base = _make_df("b", RNG.normal(10, 1, n), RNG.exponential(1, n),
                    RNG.poisson(2, n), ["left"] * n)
    cand = _make_df("c", RNG.normal(10, 1, n), RNG.exponential(1, n),
                    RNG.poisson(2, n), ["left"] * n)
    results = run_all_tests(base, cand)
    for r in results:
        assert isinstance(r, TestResult)
        assert isinstance(r.significant, bool)
