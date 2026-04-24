"""Unit tests for DriverAgent tick logic and bug modes."""

import pytest
from sim.agent import STOP_DWELL_TICKS, DriverAgent
from sim.config import AgentSpec
from sim.graph import NODE_TYPE_STOP_SIGN, NODE_TYPE_TRAFFIC_LIGHT, RoadGraph


def _make_graph():
    return RoadGraph.make_city(
        rows=5, cols=5,
        traffic_light_nodes=[(0, 0), (4, 4)],
        stop_sign_nodes=[(0, 2), (2, 0)],
    )


def _agent(graph, ignore_red=False, roll_stop=False, noise=0.0,
           start=(0, 0), goal=(4, 0)):
    # Default goal=(4,0): straight path along y=0, must cross stop_sign at (2,0)
    spec = AgentSpec(
        agent_id="test_agent",
        start=start,
        goal=goal,
        ignore_red_lights=ignore_red,
        roll_stop_signs=roll_stop,
        speed_noise_std=noise,
    )
    return DriverAgent(spec, graph)


class TestAgentBasics:
    def test_plan_is_computed_on_init(self):
        g = _make_graph()
        a = _agent(g)
        assert len(a.plan) > 0
        assert a.plan[0] == "node_0_0"
        assert a.plan[-1] == "node_4_0"   # goal is (4,0)

    def test_tick_returns_required_fields(self):
        g = _make_graph()
        a = _agent(g)
        row = a.tick(0)
        required = {"timestamp_ms", "agent_id", "node_id", "x", "y",
                    "speed_mps", "traffic_light_state", "stop_sign_zone",
                    "in_intersection", "collision_flag", "action"}
        assert required.issubset(row.keys())

    def test_agent_advances_along_plan(self):
        g = _make_graph()
        a = _agent(g)
        for t in range(300):
            a.tick(t)
            if a.done:
                break
        assert a.done, "agent should reach goal within 300 ticks"

    def test_done_agent_stays_still(self):
        g = _make_graph()
        spec = AgentSpec(agent_id="a", start=(4, 4), goal=(4, 4))
        a = DriverAgent(spec, g)
        row = a.tick(0)
        assert row["speed_mps"] == 0.0
        assert row["action"] == "hold"


class TestStopSignBehavior:
    def test_compliant_agent_stops_at_stop_sign(self):
        g = _make_graph()
        a = _agent(g, noise=0.0)
        stop_speeds = []
        for t in range(300):
            row = a.tick(t)
            if row["stop_sign_zone"]:
                stop_speeds.append(row["speed_mps"])
            if a.done:
                break
        # compliant agent must reach 0 m/s inside stop zone
        assert any(s == 0.0 for s in stop_speeds), "agent should fully stop at stop sign"

    def test_rolling_agent_never_fully_stops(self):
        g = _make_graph()
        a = _agent(g, roll_stop=True, noise=0.0)
        for t in range(300):
            row = a.tick(t)
            if row["stop_sign_zone"]:
                assert row["speed_mps"] > 0.0, "rolling agent should never fully stop"
            if a.done:
                break

    def test_stop_dwell_requirement(self):
        """Compliant agent must dwell >= STOP_DWELL_TICKS ticks at speed=0."""
        g = _make_graph()
        a = _agent(g, noise=0.0)
        zero_streak = 0
        max_streak = 0
        for t in range(300):
            row = a.tick(t)
            if row["stop_sign_zone"] and row["speed_mps"] == 0.0:
                zero_streak += 1
                max_streak = max(max_streak, zero_streak)
            else:
                zero_streak = 0
            if a.done:
                break
        assert max_streak >= STOP_DWELL_TICKS


class TestRedLightBehavior:
    def test_compliant_agent_stops_at_red_light(self):
        g = _make_graph()
        a = _agent(g, noise=0.0)
        # advance graph ticks to get a RED phase
        for _ in range(25):   # 20 GREEN + 5 YELLOW → RED starts at tick 25
            g.advance_tick()
        # force agent to be at traffic_light node
        tl_node = next(
            nid for nid, n in g.nodes.items()
            if n.node_type == NODE_TYPE_TRAFFIC_LIGHT and n.light_phase == "RED"
        )
        a.plan = [tl_node, "node_4_4"]
        a._plan_idx = 0
        a.visited_nodes = [tl_node]

        row = a.tick(0)
        assert row["speed_mps"] == 0.0
        assert row["action"] == "brake"

    def test_reckless_agent_drives_through_red(self):
        g = _make_graph()
        a = _agent(g, ignore_red=True, noise=0.0)
        for _ in range(25):
            g.advance_tick()
        tl_node = next(
            nid for nid, n in g.nodes.items()
            if n.node_type == NODE_TYPE_TRAFFIC_LIGHT and n.light_phase == "RED"
        )
        a.plan = [tl_node, "node_4_4"]
        a._plan_idx = 0
        a.visited_nodes = [tl_node]

        row = a.tick(0)
        assert row["speed_mps"] > 0.0
