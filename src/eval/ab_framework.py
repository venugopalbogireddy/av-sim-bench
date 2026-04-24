"""A/B evaluation framework for comparing traffic generators.

Orchestrates:
  1. Generating baseline and candidate samples
  2. Running all statistical tests (with BH-FDR correction)
  3. Bootstrap confidence intervals on the Wasserstein distance
  4. Saving Parquet logs to data/p3/
  5. Producing JSON + Markdown reports in outputs/p3/
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from eval.stat_tests import TestResult, run_all_tests

_TODAY = date.today().isoformat()
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "p3"
_OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "p3"


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def wasserstein_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (observed, lower, upper) for Wasserstein distance bootstrap CI."""
    if rng is None:
        rng = np.random.default_rng(7)
    observed = stats.wasserstein_distance(a, b)
    boot = np.array([
        stats.wasserstein_distance(
            rng.choice(a, size=len(a), replace=True),
            rng.choice(b, size=len(b), replace=True),
        )
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return float(observed), lo, hi


# ── main evaluation function ─────────────────────────────────────────────────

def run_ab_eval(
    baseline_fn: Callable[..., pd.DataFrame],
    candidate_fn: Callable[..., pd.DataFrame],
    label: str,
    n: int = 1000,
    alpha: float = 0.05,
    save_parquet: bool = True,
) -> dict:
    """Run a full A/B evaluation comparing baseline_fn vs candidate_fn.

    Returns a result dict suitable for JSON serialisation.
    """
    rng_base = np.random.default_rng(0)
    rng_cand = np.random.default_rng(1)

    baseline_df = baseline_fn(n=n, rng=rng_base, run_id="baseline")
    candidate_df = candidate_fn(n=n, rng=rng_cand, run_id=label)

    if save_parquet:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        baseline_df.to_parquet(_DATA_DIR / f"{_TODAY}-baseline.parquet", index=False)
        candidate_df.to_parquet(_DATA_DIR / f"{_TODAY}-{label}.parquet", index=False)

    test_results = run_all_tests(baseline_df, candidate_df, alpha=alpha)

    # bootstrap CI for each continuous metric
    ci_data: dict[str, dict] = {}
    for col in ("speed_mps", "gap_s"):
        obs, lo, hi = wasserstein_bootstrap_ci(
            baseline_df[col].to_numpy(dtype=float),
            candidate_df[col].to_numpy(dtype=float),
        )
        ci_data[col] = {"wasserstein_observed": obs, "ci_95_lo": lo, "ci_95_hi": hi}

    summary = {
        "label": label,
        "n_samples": n,
        "alpha": alpha,
        "n_tests": len(test_results),
        "n_significant": sum(r.significant for r in test_results),
        "verdict": "DIVERGENT" if any(r.significant for r in test_results) else "EQUIVALENT",
        "bootstrap_ci": ci_data,
        "tests": [_result_to_dict(r) for r in test_results],
    }
    return summary


def _result_to_dict(r: TestResult) -> dict:
    return {
        "test": r.test_name,
        "metric": r.metric,
        "statistic": round(r.statistic, 6),
        "p_value": round(r.p_value, 6) if not math.isnan(r.p_value) else None,
        "significant": r.significant,
    }


# ── report writers ────────────────────────────────────────────────────────────

def save_json_report(summary: dict, out_dir: Path | None = None) -> Path:
    out = (out_dir or _OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_TODAY}-{summary['label']}-report.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def save_markdown_report(summary: dict, out_dir: Path | None = None) -> Path:
    out = (out_dir or _OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_TODAY}-{summary['label']}-report.md"

    lines = [
        f"# A/B Evaluation Report — {summary['label']}",
        f"_Generated: {_TODAY}_",
        "",
        f"**Verdict:** {summary['verdict']}  ",
        f"**N samples:** {summary['n_samples']} | **α:** {summary['alpha']} | "
        f"**Tests run:** {summary['n_tests']} | **Significant:** {summary['n_significant']}",
        "",
        "## Test Results",
        "",
        "| Test | Metric | Statistic | p-value | Significant |",
        "|---|---|---|---|---|",
    ]
    for t in summary["tests"]:
        p_str = f"{t['p_value']:.4f}" if t["p_value"] is not None else "—"
        sig = "**YES**" if t["significant"] else "no"
        lines.append(
            f"| {t['test']} | {t['metric']} | {t['statistic']:.4f} | {p_str} | {sig} |"
        )

    lines += [
        "",
        "## Bootstrap CI (Wasserstein, 95%)",
        "",
        "| Metric | Observed | 95% CI Lower | 95% CI Upper |",
        "|---|---|---|---|",
    ]
    for col, ci in summary["bootstrap_ci"].items():
        lines.append(
            f"| {col} | {ci['wasserstein_observed']:.4f} | "
            f"{ci['ci_95_lo']:.4f} | {ci['ci_95_hi']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n")
    return path
