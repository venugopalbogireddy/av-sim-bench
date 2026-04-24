"""Unit tests for src/eval/ab_framework.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.ab_framework import (
    run_ab_eval,
    save_json_report,
    save_markdown_report,
    wasserstein_bootstrap_ci,
)
from generators import baseline as _base
from generators import deterministic as _det
from generators import stochastic as _stoch


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def test_bootstrap_ci_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 200)
    obs, lo, hi = wasserstein_bootstrap_ci(a, a.copy(), n_boot=200, rng=np.random.default_rng(1))
    assert obs < 1.0   # identical arrays → near-zero observed distance
    assert lo >= 0.0
    assert hi > lo


def test_bootstrap_ci_different():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 300)
    b = rng.normal(20, 2, 300)
    obs, lo, hi = wasserstein_bootstrap_ci(a, b, n_boot=200, rng=np.random.default_rng(2))
    assert obs > 5.0
    assert lo > 0


# ── run_ab_eval — deterministic case (should be DIVERGENT) ───────────────────

def test_run_ab_eval_deterministic_divergent(tmp_path):
    summary = run_ab_eval(
        baseline_fn=_base.generate,
        candidate_fn=_det.generate,
        label="deterministic",
        n=500,
        save_parquet=False,
    )
    assert summary["verdict"] == "DIVERGENT"
    assert summary["n_significant"] >= 3
    assert summary["n_samples"] == 500
    assert "bootstrap_ci" in summary
    assert "speed_mps" in summary["bootstrap_ci"]


# ── run_ab_eval — well-tuned stochastic (should be EQUIVALENT) ───────────────

def test_run_ab_eval_well_tuned_equivalent():
    cand_fn = lambda **kw: _stoch.generate(mode="well_tuned", **kw)
    summary = run_ab_eval(
        baseline_fn=_base.generate,
        candidate_fn=cand_fn,
        label="stochastic_well_tuned",
        n=1000,
        save_parquet=False,
    )
    assert summary["verdict"] == "EQUIVALENT"
    assert summary["n_significant"] == 0


# ── run_ab_eval — miscalibrated stochastic (should be DIVERGENT) ─────────────

def test_run_ab_eval_miscalibrated_divergent():
    cand_fn = lambda **kw: _stoch.generate(mode="miscalibrated", **kw)
    summary = run_ab_eval(
        baseline_fn=_base.generate,
        candidate_fn=cand_fn,
        label="stochastic_miscalibrated",
        n=1000,
        save_parquet=False,
    )
    assert summary["verdict"] == "DIVERGENT"
    assert summary["n_significant"] >= 2


# ── summary structure ─────────────────────────────────────────────────────────

def test_run_ab_eval_summary_structure():
    summary = run_ab_eval(
        baseline_fn=_base.generate,
        candidate_fn=_det.generate,
        label="det_struct",
        n=200,
        save_parquet=False,
    )
    assert set(summary.keys()) >= {"label", "n_samples", "alpha", "n_tests",
                                    "n_significant", "verdict", "bootstrap_ci", "tests"}
    for t in summary["tests"]:
        assert "test" in t
        assert "metric" in t
        assert "significant" in t
        assert isinstance(t["significant"], bool)


# ── report writers ────────────────────────────────────────────────────────────

def test_save_json_report(tmp_path):
    summary = run_ab_eval(
        baseline_fn=_base.generate,
        candidate_fn=_det.generate,
        label="det_json",
        n=200,
        save_parquet=False,
    )
    path = save_json_report(summary, out_dir=tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["label"] == "det_json"
    assert "tests" in loaded


def test_save_markdown_report(tmp_path):
    summary = run_ab_eval(
        baseline_fn=_base.generate,
        candidate_fn=_det.generate,
        label="det_md",
        n=200,
        save_parquet=False,
    )
    path = save_markdown_report(summary, out_dir=tmp_path)
    assert path.exists()
    content = path.read_text()
    assert "## Test Results" in content
    assert "Bootstrap CI" in content
    assert summary["verdict"] in content
