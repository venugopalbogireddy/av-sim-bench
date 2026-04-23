"""CLI entry point: evaluator run --logs data/ --baseline data/golden.parquet --out outputs/"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

from evaluator.log_gen import generate_all
from evaluator.metrics import compute_all, MetricResult


def _load(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _build_dashboard(results_by_run: dict[str, list[MetricResult]],
                     baseline_df: pd.DataFrame,
                     log_dfs: dict[str, pd.DataFrame],
                     out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sim-Log Evaluator Dashboard", fontsize=14, fontweight="bold")

    run_names = list(results_by_run.keys())
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    # Panel 1: Speed histogram overlay
    ax = axes[0, 0]
    for i, (name, df) in enumerate(log_dfs.items()):
        cruise = df.loc[~df["stop_sign_zone"] & (df["traffic_light_state"] != "RED"), "speed_mps"]
        ax.hist(cruise, bins=40, alpha=0.6, label=name, color=colors[i % len(colors)])
    ax.set_title("Cruise Speed Distribution")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("count")
    ax.legend()

    # Panel 2: Stop-sign compliance bar
    ax = axes[0, 1]
    compliance_vals = [
        next(r.value for r in results if r.name == "stop_sign_compliance")
        for results in results_by_run.values()
    ]
    bars = ax.bar(run_names, compliance_vals,
                  color=[colors[i] for i in range(len(run_names))])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Stop-Sign Compliance Rate")
    ax.set_ylabel("fraction compliant")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, compliance_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)

    # Panel 3: Violation counts (red-light + collision)
    ax = axes[1, 0]
    rl_vals = [
        next(r.value for r in results if r.name == "red_light_violation_rate")
        for results in results_by_run.values()
    ]
    col_vals = [
        next(r.value for r in results if r.name == "collision_proxy")
        for results in results_by_run.values()
    ]
    x = range(len(run_names))
    w = 0.35
    ax.bar([xi - w / 2 for xi in x], rl_vals, width=w, label="red-light rate", color="#e74c3c", alpha=0.8)
    ax.bar([xi + w / 2 for xi in x], col_vals, width=w, label="collision count", color="#e67e22", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(run_names)
    ax.set_title("Violation Summary")
    ax.legend()

    # Panel 4: KS test p-value
    ax = axes[1, 1]
    ks_vals = [
        next(r.value for r in results if r.name == "speed_ks_test")
        for results in results_by_run.values()
    ]
    bar_colors = ["#2ecc71" if v >= 0.05 else "#e74c3c" for v in ks_vals]
    bars = ax.bar(run_names, ks_vals, color=bar_colors)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="α = 0.05")
    ax.set_title("Speed KS-Test p-value")
    ax.set_ylabel("p-value")
    ax.legend()
    for bar, val in zip(bars, ks_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  dashboard → {out_path}")


@click.group()
def main() -> None:
    """Simulation log evaluator CLI."""


@main.command()
@click.option("--logs", required=True, type=click.Path(exists=True), help="Directory of .parquet log files")
@click.option("--baseline", required=True, type=click.Path(exists=True), help="Baseline parquet file (golden run)")
@click.option("--out", required=True, type=click.Path(), help="Output directory for metrics.json + dashboard.png")
def run(logs: str, baseline: str, out: str) -> None:
    """Evaluate all .parquet logs in LOGS against BASELINE."""
    logs_dir = Path(logs)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = _load(Path(baseline))
    print(f"Baseline loaded: {len(baseline_df):,} rows from {baseline}")

    log_files = sorted(logs_dir.glob("*.parquet"))
    if not log_files:
        raise click.ClickException(f"No .parquet files found in {logs_dir}")

    all_metrics: dict[str, list] = {}
    log_dfs: dict[str, pd.DataFrame] = {}

    for log_path in log_files:
        run_name = log_path.stem
        df = _load(log_path)
        log_dfs[run_name] = df
        results = compute_all(df, baseline_df)
        all_metrics[run_name] = results

        print(f"\n── {run_name} ({len(df):,} rows) ──")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.value}")

    # Write metrics.json
    metrics_out = out_dir / "metrics.json"
    payload = {
        run: [r.as_dict() for r in results]
        for run, results in all_metrics.items()
    }
    metrics_out.write_text(json.dumps(payload, indent=2))
    print(f"\n  metrics → {metrics_out}")

    # Write dashboard
    _build_dashboard(all_metrics, baseline_df, log_dfs, out_dir / "dashboard.png")


@main.command()
@click.option("--out", default="data", show_default=True, type=click.Path(), help="Directory to write generated logs")
def generate(out: str) -> None:
    """Generate synthetic driving logs (golden / regression / noisy)."""
    print("Generating synthetic logs …")
    generate_all(Path(out))
    print("Done.")


if __name__ == "__main__":
    main()
