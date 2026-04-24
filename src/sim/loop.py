"""Fixed-step simulation loop — ticks agents, collects logs, writes Parquet."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sim.agent import DriverAgent
from sim.config import AgentSpec, CityConfig
from sim.graph import RoadGraph

# P2 Parquet schema — extends P1 with node_id and action
P2_SCHEMA = pa.schema([
    pa.field("run_id",               pa.string()),
    pa.field("timestamp_ms",         pa.int64()),
    pa.field("agent_id",             pa.string()),
    pa.field("node_id",              pa.string()),
    pa.field("x",                    pa.float64()),
    pa.field("y",                    pa.float64()),
    pa.field("heading",              pa.float64()),
    pa.field("speed_mps",            pa.float64()),
    pa.field("traffic_light_state",  pa.string()),
    pa.field("stop_sign_zone",       pa.bool_()),
    pa.field("in_intersection",      pa.bool_()),
    pa.field("collision_flag",       pa.bool_()),
    pa.field("action",               pa.string()),
])


class SimLoop:
    def __init__(self, config: CityConfig, graph: RoadGraph) -> None:
        self.config = config
        self.graph = graph
        self.agents = [
            DriverAgent(spec, graph) for spec in config.agents
        ]

    def run(self) -> pd.DataFrame:
        records: list[dict] = []
        run_id = self.config.scenario

        for tick in range(self.config.max_ticks):
            self.graph.advance_tick()

            for agent in self.agents:
                row = agent.tick(tick)
                row["run_id"] = run_id
                records.append(row)

            if all(a.done for a in self.agents):
                break

        df = pd.DataFrame(records)
        # ensure bool columns are not object dtype
        for col in ("stop_sign_zone", "in_intersection", "collision_flag"):
            df[col] = df[col].astype(bool)
        return df

    def run_and_save(self, out_dir: Path) -> Path:
        df = self.run()
        today = date.today().strftime("%Y_%m_%d")
        fname = f"p2_{self.config.scenario}_{today}.parquet"
        out_path = out_dir / fname
        table = pa.Table.from_pandas(df, schema=P2_SCHEMA, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")
        print(f"  wrote {out_path} ({len(df):,} rows)")
        return out_path


def run_scenario(config_path: Path, out_dir: Path) -> Path:
    """Convenience function: load YAML → build graph → run → save."""
    config = CityConfig.from_yaml(config_path)
    graph = RoadGraph.make_city(
        rows=config.rows,
        cols=config.cols,
        traffic_light_nodes=config.traffic_light_nodes,
        stop_sign_nodes=config.stop_sign_nodes,
        blocked_edges=config.blocked_edges,
        one_way_edges=config.one_way_edges,
    )
    sim = SimLoop(config, graph)
    out_dir.mkdir(parents=True, exist_ok=True)
    return sim.run_and_save(out_dir)
