"""Graph-aware evaluation metrics for P2 grid-world logs.

Reuses MetricResult and the 3 schema-compatible metrics from metrics.py
(stop_sign_compliance, red_light_violation_rate, collision_proxy, speed_ks_test).
Adds graph-specific versions of route_completion and route_plan_adherence.
"""

from __future__ import annotations

import pandas as pd

from evaluator.metrics import (
    MetricResult,
    collision_proxy,
    red_light_violation_rate,
    speed_ks_test,
    stop_sign_compliance,
)


# ── 4 (P2). Route completion — graph version ─────────────────────────────────

def route_completion(df: pd.DataFrame, goal_node_id: str) -> MetricResult:
    """Fraction of agents whose log contains the goal node_id at least once."""
    reached = (
        df.groupby("agent_id")["node_id"]
        .apply(lambda s: goal_node_id in s.values)
    )
    rate = float(reached.mean())
    return MetricResult(
        "route_completion",
        round(rate, 4),
        rate == 1.0,
        {"reached_goal": reached.to_dict(), "goal_node_id": goal_node_id},
    )


# ── 6 (new). Route-plan adherence ────────────────────────────────────────────

def route_plan_adherence(
    df: pd.DataFrame,
    plans: dict[str, list[str]],
) -> MetricResult:
    """Fraction of agents that visited only nodes in their A* plan.

    A deviation is any node_id the agent visited that is NOT in the
    planned sequence.  Shortcuts (valid alternative paths) are NOT penalised
    here — see Project 2a in development.md for that extension.
    """
    if not plans:
        return MetricResult("route_plan_adherence", 1.0, True,
                            {"note": "no plans to evaluate"})

    compliance: dict[str, bool] = {}
    deviations_detail: dict[str, list[str]] = {}

    for agent_id, plan in plans.items():
        plan_set = set(plan)
        agent_df = df[df["agent_id"] == agent_id]
        visited = agent_df["node_id"].unique().tolist()
        off_plan = [n for n in visited if n not in plan_set]
        compliance[agent_id] = len(off_plan) == 0
        if off_plan:
            deviations_detail[agent_id] = off_plan

    rate = float(sum(compliance.values()) / max(len(compliance), 1))
    return MetricResult(
        "route_plan_adherence",
        round(rate, 4),
        rate == 1.0,
        {
            "compliance_per_agent": compliance,
            "deviations": deviations_detail,
        },
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_all(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    goal_node_id: str,
    plans: dict[str, list[str]],
) -> list[MetricResult]:
    """Run all 6 metrics and return results."""
    return [
        stop_sign_compliance(df),
        red_light_violation_rate(df),
        collision_proxy(df),
        route_completion(df, goal_node_id),
        speed_ks_test(df, baseline_df),
        route_plan_adherence(df, plans),
    ]
