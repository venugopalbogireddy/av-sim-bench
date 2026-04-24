"""Unit tests for graph-aware metrics in sim_metrics.py."""

import pandas as pd
import pytest

from evaluator.sim_metrics import route_completion, route_plan_adherence
from evaluator.metrics import MetricResult


def _base_row(**overrides) -> dict:
    row = {
        "run_id": "test",
        "timestamp_ms": 0,
        "agent_id": "agent_00",
        "node_id": "node_0_0",
        "x": 0.0,
        "y": 0.0,
        "heading": 0.0,
        "speed_mps": 8.0,
        "traffic_light_state": "GREEN",
        "stop_sign_zone": False,
        "in_intersection": False,
        "collision_flag": False,
        "action": "accelerate",
    }
    row.update(overrides)
    return row


# ── route_completion (graph version) ─────────────────────────────────────────

class TestRouteCompletionGraph:
    def test_all_agents_reach_goal(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", node_id="node_4_4"),
            _base_row(agent_id="a1", node_id="node_4_4"),
        ])
        r = route_completion(df, "node_4_4")
        assert r.passed is True
        assert r.value == pytest.approx(1.0)

    def test_no_agents_reach_goal(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", node_id="node_0_0"),
            _base_row(agent_id="a1", node_id="node_2_2"),
        ])
        r = route_completion(df, "node_4_4")
        assert r.passed is False
        assert r.value == pytest.approx(0.0)

    def test_partial_completion(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", node_id="node_4_4"),   # reached
            _base_row(agent_id="a1", node_id="node_1_1"),   # did not reach
        ])
        r = route_completion(df, "node_4_4")
        assert r.passed is False
        assert r.value == pytest.approx(0.5)

    def test_goal_in_middle_of_log_counts(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", node_id="node_0_0", timestamp_ms=0),
            _base_row(agent_id="a0", node_id="node_4_4", timestamp_ms=100),
            _base_row(agent_id="a0", node_id="node_3_3", timestamp_ms=200),  # overshot
        ])
        r = route_completion(df, "node_4_4")
        assert r.passed is True


# ── route_plan_adherence ──────────────────────────────────────────────────────

class TestRoutePlanAdherence:
    _plan = ["node_0_0", "node_1_0", "node_2_0", "node_3_0", "node_4_0",
             "node_4_1", "node_4_2", "node_4_3", "node_4_4"]

    def test_on_plan_agent_passes(self):
        df = pd.DataFrame([
            _base_row(agent_id="agent_00", node_id=n)
            for n in self._plan
        ])
        r = route_plan_adherence(df, {"agent_00": self._plan})
        assert r.passed is True
        assert r.value == pytest.approx(1.0)

    def test_off_plan_agent_fails(self):
        rows = [_base_row(agent_id="agent_00", node_id=n) for n in self._plan]
        rows.append(_base_row(agent_id="agent_00", node_id="node_2_2"))  # deviation
        df = pd.DataFrame(rows)
        r = route_plan_adherence(df, {"agent_00": self._plan})
        assert r.passed is False
        assert "agent_00" in r.details["deviations"]

    def test_all_agents_compliant(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", node_id="node_0_0"),
            _base_row(agent_id="a0", node_id="node_1_0"),
            _base_row(agent_id="a1", node_id="node_0_0"),
            _base_row(agent_id="a1", node_id="node_1_0"),
        ])
        plans = {"a0": ["node_0_0", "node_1_0"], "a1": ["node_0_0", "node_1_0"]}
        r = route_plan_adherence(df, plans)
        assert r.passed is True
        assert r.value == pytest.approx(1.0)

    def test_partial_adherence(self):
        df = pd.DataFrame([
            _base_row(agent_id="a0", node_id="node_0_0"),   # on plan
            _base_row(agent_id="a1", node_id="node_3_3"),   # off plan
        ])
        plans = {
            "a0": ["node_0_0", "node_1_0"],
            "a1": ["node_0_0", "node_1_0"],
        }
        r = route_plan_adherence(df, plans)
        assert r.passed is False
        assert r.value == pytest.approx(0.5)

    def test_empty_plans_dict(self):
        df = pd.DataFrame([_base_row()])
        r = route_plan_adherence(df, {})
        assert r.value == pytest.approx(1.0)   # vacuously true


# ── integration: pipeline-level sanity ───────────────────────────────────────

class TestSimMetricsPipeline:
    def test_compute_all_returns_six_metrics(self):
        from sim.graph import RoadGraph
        from sim.config import AgentSpec
        from sim.agent import DriverAgent
        from sim.loop import SimLoop
        from sim.config import CityConfig
        from evaluator.sim_metrics import compute_all

        # build a tiny 3×3 city and run golden scenario
        graph = RoadGraph.make_city(rows=3, cols=3,
                                    traffic_light_nodes=[(0, 0)],
                                    stop_sign_nodes=[(1, 0)])
        spec = AgentSpec(agent_id="a0", start=(0, 0), goal=(2, 2))
        agent = DriverAgent(spec, graph)
        records = []
        for t in range(100):
            graph.advance_tick()
            row = agent.tick(t)
            row["run_id"] = "test"
            records.append(row)
            if agent.done:
                break

        import pandas as pd
        df = pd.DataFrame(records)
        for col in ("stop_sign_zone", "in_intersection", "collision_flag"):
            df[col] = df[col].astype(bool)

        goal = "node_2_2"
        plan = graph.astar_path("node_0_0", "node_2_2")
        results = compute_all(df, df, goal, {"a0": plan})
        assert len(results) == 6
        names = [r.name for r in results]
        assert "route_completion" in names
        assert "route_plan_adherence" in names
