"""Road graph — nodes with (x, y) coordinates, typed intersections, directed edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


# ── Node ─────────────────────────────────────────────────────────────────────

NODE_TYPE_ROAD          = "road"
NODE_TYPE_STOP_SIGN     = "stop_sign"
NODE_TYPE_TRAFFIC_LIGHT = "traffic_light"

LIGHT_CYCLE = ["GREEN"] * 20 + ["YELLOW"] * 5 + ["RED"] * 25  # 50-tick cycle


@dataclass
class Node:
    x: int
    y: int
    node_type: str = NODE_TYPE_ROAD    # "road" | "stop_sign" | "traffic_light"
    phase_offset: int = 0              # stagger different intersections so not all in sync
    _tick: int = field(default=0, repr=False, compare=False)

    @property
    def node_id(self) -> str:
        return f"node_{self.x}_{self.y}"

    @property
    def light_phase(self) -> str:
        """Current traffic-light phase. Only meaningful for traffic_light nodes."""
        if self.node_type != NODE_TYPE_TRAFFIC_LIGHT:
            return "GREEN"
        return LIGHT_CYCLE[(self._tick + self.phase_offset) % len(LIGHT_CYCLE)]

    def advance_tick(self) -> None:
        self._tick += 1

    @staticmethod
    def from_id(node_id: str) -> tuple[int, int]:
        """Parse 'node_x_y' → (x, y). Useful in metrics."""
        _, x, y = node_id.split("_")
        return int(x), int(y)


# ── Edge ─────────────────────────────────────────────────────────────────────

@dataclass
class Edge:
    length_m: float = 1.0
    speed_limit_mps: float = 10.0
    one_way: bool = False   # if True only the declared direction exists
    blocked: bool = False


# ── RoadGraph ─────────────────────────────────────────────────────────────────

class RoadGraph:
    def __init__(self, g: nx.DiGraph, nodes: dict[str, Node]) -> None:
        self.g = g
        self.nodes = nodes  # node_id → Node

    # ── queries ──────────────────────────────────────────────────────────────

    def node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def neighbors(self, node_id: str) -> list[str]:
        """Traversable successors (blocked edges excluded)."""
        return [
            v for v in self.g.successors(node_id)
            if not self.g[node_id][v].get("blocked", False)
        ]

    def astar_path(self, start: str, goal: str) -> list[str]:
        """Return shortest path as list of node_ids using A* with Euclidean heuristic.

        Uses a subgraph view that excludes blocked edges so A* never routes through them.
        """
        def heuristic(a: str, b: str) -> float:
            ax, ay = Node.from_id(a)
            bx, by = Node.from_id(b)
            return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

        traversable = nx.subgraph_view(
            self.g,
            filter_edge=lambda u, v: not self.g[u][v].get("blocked", False),
        )
        try:
            return nx.astar_path(
                traversable, start, goal,
                heuristic=heuristic,
                weight="length_m",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def advance_tick(self) -> None:
        """Advance all traffic-light phases by one tick."""
        for node in self.nodes.values():
            node.advance_tick()

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def make_city(
        cls,
        rows: int,
        cols: int,
        traffic_light_nodes: list[tuple[int, int]] | None = None,
        stop_sign_nodes: list[tuple[int, int]] | None = None,
        blocked_edges: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        one_way_edges: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        light_phase_offsets: dict[tuple[int, int], int] | None = None,
    ) -> "RoadGraph":
        """phase_offset for each traffic_light node can be set explicitly via
        light_phase_offsets.  If a node has no explicit offset, a coordinate-based
        default is used so nearby intersections are naturally staggered:
            offset = (x * 5 + y * 3) % cycle_len
        """
        """Build a rows×cols grid city.

        All edges are bidirectional by default.  one_way_edges removes the
        reverse direction.  blocked_edges mark edges as impassable.
        """
        tl_set = {tuple(n) for n in (traffic_light_nodes or [])}
        ss_set = {tuple(n) for n in (stop_sign_nodes or [])}
        blocked_set = {(tuple(a), tuple(b)) for a, b in (blocked_edges or [])}
        one_way_set = {(tuple(a), tuple(b)) for a, b in (one_way_edges or [])}
        offsets = {tuple(k): v for k, v in (light_phase_offsets or {}).items()}
        cycle_len = len(LIGHT_CYCLE)

        nodes: dict[str, Node] = {}
        g = nx.DiGraph()

        # create nodes
        for x in range(cols):
            for y in range(rows):
                coord = (x, y)
                if coord in tl_set:
                    ntype = NODE_TYPE_TRAFFIC_LIGHT
                    phase_offset = offsets.get(coord, (x * 5 + y * 3) % cycle_len)
                elif coord in ss_set:
                    ntype = NODE_TYPE_STOP_SIGN
                    phase_offset = 0
                else:
                    ntype = NODE_TYPE_ROAD
                    phase_offset = 0
                n = Node(x=x, y=y, node_type=ntype, phase_offset=phase_offset)
                nodes[n.node_id] = n
                g.add_node(n.node_id, data=n)

        # create edges — evaluate each directed edge independently.
        # one_way_set entry (a, b) means only a→b exists; b→a must NOT be added.
        for x in range(cols):
            for y in range(rows):
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx_, ny_ = x + dx, y + dy
                    if not (0 <= nx_ < cols and 0 <= ny_ < rows):
                        continue
                    src_coord = (x, y)
                    dst_coord = (nx_, ny_)

                    is_blocked = src_coord, dst_coord in blocked_set or (src_coord, dst_coord) in blocked_set
                    # if the reverse direction is the canonical one-way, this direction must not exist
                    reverse_is_one_way = (dst_coord, src_coord) in one_way_set

                    if (src_coord, dst_coord) in blocked_set:
                        continue
                    if reverse_is_one_way:
                        continue

                    src_id = f"node_{x}_{y}"
                    dst_id = f"node_{nx_}_{ny_}"
                    g.add_edge(src_id, dst_id, length_m=1.0, speed_limit_mps=10.0, blocked=False)

        return cls(g, nodes)
