"""End-to-end pipeline tests using generated Parquet logs."""

import json
import tempfile
from pathlib import Path

import pytest

from evaluator.log_gen import generate_all
from evaluator.metrics import compute_all
import pyarrow.parquet as pq


@pytest.fixture(scope="module")
def generated_logs(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    return generate_all(data_dir)


def _load(path: Path):
    return pq.read_table(path).to_pandas()


class TestPipelineRegression:
    def test_regression_red_light_fails(self, generated_logs):
        """regression run must trigger red-light violation."""
        df = _load(generated_logs["regression"])
        baseline = _load(generated_logs["golden"])
        results = compute_all(df, baseline)
        rl = next(r for r in results if r.name == "red_light_violation_rate")
        assert rl.passed is False, "regression run should have red-light violations"
        assert rl.value > 0.0

    def test_regression_stop_sign_fails(self, generated_logs):
        """regression run must flag non-compliant stop behaviour."""
        df = _load(generated_logs["regression"])
        baseline = _load(generated_logs["golden"])
        results = compute_all(df, baseline)
        ss = next(r for r in results if r.name == "stop_sign_compliance")
        assert ss.passed is False

    def test_regression_has_collisions(self, generated_logs):
        df = _load(generated_logs["regression"])
        baseline = _load(generated_logs["golden"])
        results = compute_all(df, baseline)
        col = next(r for r in results if r.name == "collision_proxy")
        assert col.passed is False


class TestPipelineGolden:
    def test_golden_all_pass(self, generated_logs):
        df = _load(generated_logs["golden"])
        baseline = _load(generated_logs["golden"])
        results = compute_all(df, baseline)
        for r in results:
            assert r.passed is True, f"golden should pass {r.name}, got value={r.value}"

    def test_golden_full_route_completion(self, generated_logs):
        df = _load(generated_logs["golden"])
        baseline = _load(generated_logs["golden"])
        results = compute_all(df, baseline)
        rc = next(r for r in results if r.name == "route_completion")
        assert rc.value == pytest.approx(1.0)


class TestPipelineNoisy:
    def test_noisy_no_violations(self, generated_logs):
        """noisy run should pass violation metrics despite speed jitter."""
        df = _load(generated_logs["noisy"])
        baseline = _load(generated_logs["golden"])
        results = compute_all(df, baseline)
        rl = next(r for r in results if r.name == "red_light_violation_rate")
        col = next(r for r in results if r.name == "collision_proxy")
        assert rl.passed is True
        assert col.passed is True

    def test_noisy_ks_detects_drift(self, generated_logs):
        """noisy has higher speed variance — KS test should flag it."""
        df = _load(generated_logs["noisy"])
        baseline = _load(generated_logs["golden"])
        results = compute_all(df, baseline)
        ks = next(r for r in results if r.name == "speed_ks_test")
        assert ks.passed is False, "noisy speed distribution should drift from golden"
