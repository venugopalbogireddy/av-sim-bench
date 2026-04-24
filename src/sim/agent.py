"""Rule-based driver agent with A* planning and configurable bug modes."""

from __future__ import annotations

import numpy as np

from sim.config import AgentSpec
from sim.graph import NODE_TYPE_STOP_SIGN, NODE_TYPE_TRAFFIC_LIGHT, RoadGraph

RNG = np.random.default_rng(42)

CRUISE_SPEED_MPS = 8.0
STOP_DWELL_TICKS = 10        # consecutive ticks at speed=0 required at stop signs
TICK_DT = 0.1                # seconds per tick (100 ms)


class DriverAgent:
    """Rule-based agent that follows an A* plan and respects traffic rules.

    Bug modes (set via AgentSpec):
      - ignore_red_lights: drives through intersection at cruise speed when RED
      - roll_stop_signs:   never comes to a full stop, drives at reduced speed
    """

    def __init__(self, spec: AgentSpec, graph: RoadGraph) -> None:
        self.spec = spec
        self.graph = graph
        self.agent_id = spec.agent_id
        self.goal_node_id = f"node_{spec.goal[0]}_{spec.goal[1]}"
        start_node_id = f"node_{spec.start[0]}_{spec.start[1]}"

        self.plan: list[str] = graph.astar_path(start_node_id, self.goal_node_id)
        self._plan_idx: int = 0          # index into self.plan for current position
        self.visited_nodes: list[str] = [start_node_id]
        self.speed: float = 0.0
        self._stop_dwell: int = 0        # ticks spent fully stopped at current stop sign
        self._stop_satisfied: bool = False
        self.done: bool = len(self.plan) == 0 or start_node_id == self.goal_node_id

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def current_node_id(self) -> str:
        return self.plan[self._plan_idx] if self.plan else self.visited_nodes[-1]

    def tick(self, tick_num: int) -> dict:
        """Advance one simulation step, return a log-row dict.

        Row always reflects the node the agent occupied at the START of this tick
        (pre-advance), so node_id, in_intersection, stop_sign_zone, and x/y are
        always consistent with each other.
        """
        pre_node_id = self.current_node_id
        node = self.graph.node(pre_node_id)
        light = node.light_phase
        at_stop = node.node_type == NODE_TYPE_STOP_SIGN
        at_light = node.node_type == NODE_TYPE_TRAFFIC_LIGHT
        in_intersection = at_light  # signalised intersections only

        action, speed = self._decide(node, light, at_stop, at_light)
        heading = self._heading()

        # advance to next node if moving and not at the goal
        was_done_before = self.done
        if speed > 0 and not self.done:
            self._advance_node()

        self.speed = speed

        # When this tick caused arrival at the goal, record the GOAL node so
        # route_completion can find it.  Speed is recorded as 0 (agent has arrived
        # and stopped) so violation metrics aren't triggered by the final move.
        just_arrived = (not was_done_before) and self.done
        record_node_id = self.current_node_id if just_arrived else pre_node_id
        record_node = self.graph.node(record_node_id)
        record_speed = 0.0 if just_arrived else round(float(speed), 4)
        record_action = "hold" if just_arrived else action

        return {
            "timestamp_ms": tick_num * int(TICK_DT * 1000),
            "agent_id": self.agent_id,
            "node_id": record_node_id,
            "x": float(record_node.x),
            "y": float(record_node.y),
            "heading": heading,
            "speed_mps": record_speed,
            "traffic_light_state": record_node.light_phase,
            "stop_sign_zone": record_node.node_type == NODE_TYPE_STOP_SIGN,
            "in_intersection": record_node.node_type == NODE_TYPE_TRAFFIC_LIGHT,
            "collision_flag": False,         # updated by sim loop if needed
            "action": record_action,
        }

    # ── private ──────────────────────────────────────────────────────────────

    def _decide(
        self,
        node,
        light: str,
        at_stop: bool,
        at_light: bool,
    ) -> tuple[str, float]:
        """Return (action, speed_mps) for this tick."""
        if self.done:
            return "hold", 0.0

        noise = RNG.normal(0, self.spec.speed_noise_std)

        # ── stop-sign logic ──────────────────────────────────────────────────
        if at_stop and not self.spec.roll_stop_signs:
            if not self._stop_satisfied:
                self._stop_dwell += 1
                if self._stop_dwell >= STOP_DWELL_TICKS:
                    self._stop_satisfied = True
                return "brake", 0.0
        elif at_stop and self.spec.roll_stop_signs:
            # rolling stop: slow down but never fully stop
            return "brake", max(1.5 + noise, 0.5)

        # reset dwell when leaving stop zone
        if not at_stop:
            self._stop_dwell = 0
            self._stop_satisfied = False

        # ── red-light logic ──────────────────────────────────────────────────
        if at_light and light == "RED" and not self.spec.ignore_red_lights:
            return "brake", 0.0

        # ── cruise ──────────────────────────────────────────────────────────
        cruise = max(CRUISE_SPEED_MPS + noise, 0.5)
        return "accelerate", cruise

    def _advance_node(self) -> None:
        if self._plan_idx + 1 < len(self.plan):
            self._plan_idx += 1
            self.visited_nodes.append(self.current_node_id)
            if self.current_node_id == self.goal_node_id:
                self.done = True

    def _heading(self) -> float:
        """Simple heading: angle from previous node to current (degrees)."""
        if len(self.visited_nodes) < 2:
            return 0.0
        prev = self.visited_nodes[-2] if len(self.visited_nodes) >= 2 else self.current_node_id
        px, py = [int(v) for v in prev.split("_")[1:]]
        cx, cy = [int(v) for v in self.current_node_id.split("_")[1:]]
        import math
        return math.degrees(math.atan2(cy - py, cx - px)) % 360
