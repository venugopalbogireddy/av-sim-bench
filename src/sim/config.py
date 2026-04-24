"""YAML-driven city and scenario configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AgentSpec:
    agent_id: str
    start: tuple[int, int]
    goal: tuple[int, int]
    ignore_red_lights: bool = False
    roll_stop_signs: bool = False
    speed_noise_std: float = 0.05   # std of Gaussian noise on cruise speed


@dataclass
class CityConfig:
    scenario: str
    rows: int
    cols: int
    traffic_light_nodes: list[tuple[int, int]] = field(default_factory=list)
    stop_sign_nodes: list[tuple[int, int]] = field(default_factory=list)
    blocked_edges: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)
    one_way_edges: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)
    agents: list[AgentSpec] = field(default_factory=list)
    max_ticks: int = 300

    @classmethod
    def from_yaml(cls, path: Path) -> "CityConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        agents = [
            AgentSpec(
                agent_id=a["agent_id"],
                start=tuple(a["start"]),
                goal=tuple(a["goal"]),
                ignore_red_lights=a.get("ignore_red_lights", False),
                roll_stop_signs=a.get("roll_stop_signs", False),
                speed_noise_std=a.get("speed_noise_std", 0.05),
            )
            for a in raw.get("agents", [])
        ]

        return cls(
            scenario=raw["scenario"],
            rows=raw["city"]["rows"],
            cols=raw["city"]["cols"],
            traffic_light_nodes=[tuple(n) for n in raw["city"].get("traffic_light_nodes", [])],
            stop_sign_nodes=[tuple(n) for n in raw["city"].get("stop_sign_nodes", [])],
            blocked_edges=[
                (tuple(e[0]), tuple(e[1]))
                for e in raw["city"].get("blocked_edges", [])
            ],
            one_way_edges=[
                (tuple(e[0]), tuple(e[1]))
                for e in raw["city"].get("one_way_edges", [])
            ],
            agents=agents,
            max_ticks=raw.get("max_ticks", 300),
        )
