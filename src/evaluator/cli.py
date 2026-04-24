"""CLI — P1 log-replay evaluator + P2 graph-world sim and eval."""

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


# ═══════════════════════════════════════════════════════════════════════════
#  P2 — graph-world sim + eval
# ═══════════════════════════════════════════════════════════════════════════

def _build_p2_dashboard(
    results_by_run: dict[str, list[MetricResult]],
    log_dfs: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    run_names = list(results_by_run.keys())
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("P2 Grid-World Sim Evaluator Dashboard", fontsize=14, fontweight="bold")

    def _val(results, name):
        r = next((r for r in results if r.name == name), None)
        return r.value if r else 0.0

    # 1 — Speed histogram
    ax = axes[0, 0]
    for i, (name, df) in enumerate(log_dfs.items()):
        cruise = df.loc[~df["stop_sign_zone"] & (df["traffic_light_state"] != "RED"), "speed_mps"]
        ax.hist(cruise, bins=30, alpha=0.6, label=name, color=colors[i % len(colors)])
    ax.set_title("Cruise Speed Distribution")
    ax.set_xlabel("speed (m/s)")
    ax.legend()

    # 2 — Stop-sign compliance
    ax = axes[0, 1]
    vals = [_val(r, "stop_sign_compliance") for r in results_by_run.values()]
    bars = ax.bar(run_names, vals, color=[colors[i] for i in range(len(run_names))])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Stop-Sign Compliance")
    ax.set_ylim(0, 1.15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # 3 — Red-light violation rate
    ax = axes[0, 2]
    vals = [_val(r, "red_light_violation_rate") for r in results_by_run.values()]
    bar_colors = ["#2ecc71" if v == 0 else "#e74c3c" for v in vals]
    bars = ax.bar(run_names, vals, color=bar_colors)
    ax.set_title("Red-Light Violation Rate")
    ax.set_ylim(0, 1.15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # 4 — Route completion
    ax = axes[1, 0]
    vals = [_val(r, "route_completion") for r in results_by_run.values()]
    bar_colors = ["#2ecc71" if v == 1.0 else "#e74c3c" for v in vals]
    bars = ax.bar(run_names, vals, color=bar_colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Route Completion Rate")
    ax.set_ylim(0, 1.15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # 5 — Route-plan adherence
    ax = axes[1, 1]
    vals = [_val(r, "route_plan_adherence") for r in results_by_run.values()]
    bar_colors = ["#2ecc71" if v == 1.0 else "#e74c3c" for v in vals]
    bars = ax.bar(run_names, vals, color=bar_colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Route-Plan Adherence")
    ax.set_ylim(0, 1.15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # 6 — KS test p-value
    ax = axes[1, 2]
    vals = [_val(r, "speed_ks_test") for r in results_by_run.values()]
    bar_colors = ["#2ecc71" if v >= 0.05 else "#e74c3c" for v in vals]
    bars = ax.bar(run_names, vals, color=bar_colors)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="α=0.05")
    ax.set_title("Speed KS-Test p-value")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  dashboard → {out_path}")


def _reconstruct_plans(config_path: Path) -> tuple[str, dict[str, list[str]]]:
    """Load config, build graph, run A* for each agent → (goal_node_id, plans)."""
    from sim.config import CityConfig
    from sim.graph import RoadGraph

    config = CityConfig.from_yaml(config_path)
    graph = RoadGraph.make_city(
        rows=config.rows,
        cols=config.cols,
        traffic_light_nodes=config.traffic_light_nodes,
        stop_sign_nodes=config.stop_sign_nodes,
        blocked_edges=config.blocked_edges,
        one_way_edges=config.one_way_edges,
    )
    plans: dict[str, list[str]] = {}
    goal_node_id = ""
    for spec in config.agents:
        start_id = f"node_{spec.start[0]}_{spec.start[1]}"
        goal_id = f"node_{spec.goal[0]}_{spec.goal[1]}"
        goal_node_id = goal_id
        plans[spec.agent_id] = graph.astar_path(start_id, goal_id)
    return goal_node_id, plans


@main.group()
def sim() -> None:
    """P2 graph-world simulator commands."""


@sim.command("run")
@click.option("--config", required=True, type=click.Path(exists=True),
              help="Scenario YAML config (e.g. configs/p2_golden.yaml)")
@click.option("--out", default="data/p2", show_default=True, type=click.Path(),
              help="Output directory for Parquet logs")
def sim_run(config: str, out: str) -> None:
    """Run one simulation scenario and write a Parquet log."""
    from sim.loop import run_scenario
    config_path = Path(config)
    out_dir = Path(out)
    print(f"Running scenario: {config_path.stem} …")
    path = run_scenario(config_path, out_dir)
    print(f"Done → {path}")


@sim.command("eval")
@click.option("--logs", required=True, type=click.Path(exists=True),
              help="Directory of P2 .parquet logs")
@click.option("--baseline", required=True, type=click.Path(exists=True),
              help="Baseline parquet (golden run)")
@click.option("--config", required=True, type=click.Path(exists=True),
              help="Scenario YAML config used to reconstruct A* plans")
@click.option("--out", required=True, type=click.Path(),
              help="Output directory for metrics.json + dashboard.png")
def sim_eval(logs: str, baseline: str, config: str, out: str) -> None:
    """Evaluate P2 graph-world logs (6 metrics, graph-aware)."""
    from evaluator.sim_metrics import compute_all as p2_compute_all

    logs_dir = Path(logs)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_node_id, plans = _reconstruct_plans(Path(config))
    baseline_df = _load(Path(baseline))
    print(f"Baseline loaded: {len(baseline_df):,} rows | goal={goal_node_id}")

    log_files = sorted(logs_dir.glob("*.parquet"))
    if not log_files:
        raise click.ClickException(f"No .parquet files found in {logs_dir}")

    all_metrics: dict[str, list] = {}
    log_dfs: dict[str, pd.DataFrame] = {}

    for log_path in log_files:
        run_name = log_path.stem
        df = _load(log_path)
        log_dfs[run_name] = df
        results = p2_compute_all(df, baseline_df, goal_node_id, plans)
        all_metrics[run_name] = results

        print(f"\n── {run_name} ({len(df):,} rows) ──")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.value}")

    metrics_out = out_dir / "p2_metrics.json"
    payload = {run: [r.as_dict() for r in res] for run, res in all_metrics.items()}
    metrics_out.write_text(json.dumps(payload, indent=2))
    print(f"\n  metrics → {metrics_out}")

    _build_p2_dashboard(all_metrics, log_dfs, out_dir / "p2_dashboard.png")


@main.group()
def pipeline() -> None:
    """P2 end-to-end commands (sim → eval in one shot)."""


@pipeline.command("run")
@click.option("--configs-dir", default="configs", show_default=True, type=click.Path(exists=True),
              help="Directory containing p2_*.yaml scenario configs")
@click.option("--out-data", default="data/p2", show_default=True, type=click.Path(),
              help="Output directory for Parquet logs")
@click.option("--out-eval", default="outputs/p2", show_default=True, type=click.Path(),
              help="Output directory for metrics + dashboard")
@click.option("--baseline-scenario", default="golden", show_default=True,
              help="Scenario name used as baseline (must be in configs-dir)")
def pipeline_run(configs_dir: str, out_data: str, out_eval: str, baseline_scenario: str) -> None:
    """Run all p2_*.yaml scenarios then evaluate. One command to rule them all."""
    from sim.loop import run_scenario
    from evaluator.sim_metrics import compute_all as p2_compute_all

    configs_path = Path(configs_dir)
    out_data_path = Path(out_data)
    out_eval_path = Path(out_eval)
    out_eval_path.mkdir(parents=True, exist_ok=True)

    config_files = sorted(configs_path.glob("p2_*.yaml"))
    if not config_files:
        raise click.ClickException(f"No p2_*.yaml files found in {configs_path}")

    # ── sim phase ────────────────────────────────────────────────────────────
    print("=== SIM PHASE ===")
    log_paths: dict[str, Path] = {}
    for cfg in config_files:
        print(f"\nScenario: {cfg.stem}")
        path = run_scenario(cfg, out_data_path)
        log_paths[cfg.stem] = path

    # ── find baseline ────────────────────────────────────────────────────────
    baseline_cfg = configs_path / f"p2_{baseline_scenario}.yaml"
    if not baseline_cfg.exists():
        raise click.ClickException(f"Baseline config not found: {baseline_cfg}")

    # find latest baseline parquet
    baseline_files = sorted(out_data_path.glob(f"p2_{baseline_scenario}_*.parquet"))
    if not baseline_files:
        raise click.ClickException("Baseline parquet not found in out_data")
    baseline_df = _load(baseline_files[-1])
    goal_node_id, plans = _reconstruct_plans(baseline_cfg)

    # ── eval phase ───────────────────────────────────────────────────────────
    print("\n=== EVAL PHASE ===")
    all_metrics: dict[str, list] = {}
    log_dfs: dict[str, pd.DataFrame] = {}

    for log_path in sorted(out_data_path.glob("p2_*.parquet")):
        run_name = log_path.stem
        df = _load(log_path)
        log_dfs[run_name] = df
        results = p2_compute_all(df, baseline_df, goal_node_id, plans)
        all_metrics[run_name] = results
        print(f"\n── {run_name} ({len(df):,} rows) ──")
        for r in results:
            print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.value}")

    metrics_out = out_eval_path / "p2_metrics.json"
    payload = {run: [r.as_dict() for r in res] for run, res in all_metrics.items()}
    metrics_out.write_text(json.dumps(payload, indent=2))
    print(f"\n  metrics → {metrics_out}")
    _build_p2_dashboard(all_metrics, log_dfs, out_eval_path / "p2_dashboard.png")


if __name__ == "__main__":
    main()
