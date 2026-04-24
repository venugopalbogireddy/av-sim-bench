"""Unit tests for RoadGraph, Node, and A* path-finding."""

import pytest
from sim.graph import NODE_TYPE_ROAD, NODE_TYPE_STOP_SIGN, NODE_TYPE_TRAFFIC_LIGHT, Node, RoadGraph


# ── Node ─────────────────────────────────────────────────────────────────────

class TestNode:
    def test_node_id_format(self):
        n = Node(x=3, y=4)
        assert n.node_id == "node_3_4"

    def test_from_id_roundtrip(self):
        n = Node(x=2, y=7)
        x, y = Node.from_id(n.node_id)
        assert x == 2 and y == 7

    def test_road_node_always_green(self):
        n = Node(x=0, y=0, node_type=NODE_TYPE_ROAD)
        assert n.light_phase == "GREEN"

    def test_traffic_light_cycles(self):
        n = Node(x=0, y=0, node_type=NODE_TYPE_TRAFFIC_LIGHT)
        phases = {n.light_phase for _ in range(50) if not n.advance_tick() or True}
        # should see GREEN and eventually RED over 50 ticks
        assert "GREEN" in phases
        assert "RED" in phases

    def test_stop_sign_always_green_phase(self):
        n = Node(x=1, y=1, node_type=NODE_TYPE_STOP_SIGN)
        n.advance_tick()
        assert n.light_phase == "GREEN"   # stop signs use dwell logic, not phase


# ── RoadGraph construction ────────────────────────────────────────────────────

class TestRoadGraph:
    def _city(self, rows=5, cols=5, **kwargs):
        return RoadGraph.make_city(rows=rows, cols=cols, **kwargs)

    def test_node_count(self):
        g = self._city()
        assert len(g.nodes) == 25   # 5×5

    def test_corner_node_types(self):
        g = self._city(
            traffic_light_nodes=[(0, 0), (4, 4)],
            stop_sign_nodes=[(0, 2)],
        )
        assert g.node("node_0_0").node_type == NODE_TYPE_TRAFFIC_LIGHT
        assert g.node("node_4_4").node_type == NODE_TYPE_TRAFFIC_LIGHT
        assert g.node("node_0_2").node_type == NODE_TYPE_STOP_SIGN
        assert g.node("node_2_2").node_type == NODE_TYPE_ROAD

    def test_bidirectional_edges_by_default(self):
        g = self._city(rows=3, cols=3)
        assert "node_1_1" in g.neighbors("node_0_1")
        assert "node_0_1" in g.neighbors("node_1_1")

    def test_one_way_edge_removes_reverse(self):
        g = self._city(rows=3, cols=3, one_way_edges=[((0, 0), (1, 0))])
        assert "node_1_0" in g.neighbors("node_0_0")   # forward exists
        assert "node_0_0" not in g.neighbors("node_1_0")  # reverse removed

    def test_blocked_edge_excluded_from_neighbors(self):
        g = self._city(rows=3, cols=3, blocked_edges=[((0, 0), (1, 0))])
        assert "node_1_0" not in g.neighbors("node_0_0")

    def test_no_self_loops(self):
        g = self._city(rows=3, cols=3)
        for node_id in g.nodes:
            assert node_id not in g.neighbors(node_id)

    def test_corner_has_two_neighbors(self):
        g = self._city(rows=3, cols=3)
        assert len(g.neighbors("node_0_0")) == 2  # right and up only


# ── A* path-finding ───────────────────────────────────────────────────────────

class TestAstar:
    def _city(self):
        return RoadGraph.make_city(
            rows=5, cols=5,
            traffic_light_nodes=[(0, 0), (4, 4)],
            stop_sign_nodes=[(2, 0), (0, 2)],
        )

    def test_path_exists(self):
        g = self._city()
        path = g.astar_path("node_0_0", "node_4_4")
        assert len(path) > 0

    def test_path_starts_and_ends_correctly(self):
        g = self._city()
        path = g.astar_path("node_0_0", "node_4_4")
        assert path[0] == "node_0_0"
        assert path[-1] == "node_4_4"

    def test_path_nodes_are_adjacent(self):
        g = self._city()
        path = g.astar_path("node_0_0", "node_4_4")
        for a, b in zip(path, path[1:]):
            ax, ay = Node.from_id(a)
            bx, by = Node.from_id(b)
            assert abs(ax - bx) + abs(ay - by) == 1, f"{a} → {b} not adjacent"

    def test_no_path_through_all_blocked(self):
        # block every edge from (1,y) to (2,y) — cuts the grid vertically
        blocked = [((1, y), (2, y)) for y in range(5)] + [((2, y), (1, y)) for y in range(5)]
        g = RoadGraph.make_city(rows=5, cols=5, blocked_edges=blocked)
        path = g.astar_path("node_0_0", "node_4_4")
        assert path == []

    def test_trivial_path_same_node(self):
        g = self._city()
        path = g.astar_path("node_2_2", "node_2_2")
        # NetworkX returns single-node path for same start/goal
        assert path == ["node_2_2"] or path == []
