"""DuckDB-backed log store for scanning Parquet files as a virtual table."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class LogStore:
    """DuckDB-backed log reader.

    Scans a directory (or single file) of Parquet logs as one lazy virtual
    table. Uses filename=true so every row carries its source file path,
    which enables:
      - run_file_map(): {run_id → dated file stem} for display labels
      - load_run():     restricts to the latest file per run_id, preventing
                        double-counting when historical files accumulate
      - Predicate pushdown on run_id skips unrelated row groups at scale
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_dir(cls, logs_dir: Path) -> "LogStore":
        """Register all *.parquet files in logs_dir as a DuckDB view.

        No data is read until a query method is called (lazy scan).
        """
        pattern = str(logs_dir / "*.parquet")
        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW logs AS "
            f"SELECT * FROM read_parquet('{pattern}', filename=true, union_by_name=true)"
        )
        return cls(con)

    @classmethod
    def from_file(cls, path: Path) -> "LogStore":
        """Register a single Parquet file as a DuckDB view.

        Used for loading a baseline file when the exact path is already known.
        """
        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW logs AS "
            f"SELECT * FROM read_parquet('{path}', filename=true)"
        )
        return cls(con)

    # ── query methods ─────────────────────────────────────────────────────────

    def run_ids(self) -> list[str]:
        """Return distinct run_id values present in the view, sorted."""
        rows = self._con.execute(
            "SELECT DISTINCT run_id FROM logs ORDER BY run_id"
        ).fetchall()
        return [r[0] for r in rows]

    def run_file_map(self) -> dict[str, str]:
        """Return {run_id: latest_file_stem} for display labels.

        MAX(filename) is correct because ISO-date file names sort
        lexicographically in chronological order — the latest file wins.
        """
        rows = self._con.execute(
            "SELECT run_id, MAX(filename) FROM logs GROUP BY run_id ORDER BY run_id"
        ).fetchall()
        return {run_id: Path(latest).stem for run_id, latest in rows}

    def load_run(self, run_id: str) -> pd.DataFrame:
        """Return rows for run_id from its latest source file as a DataFrame.

        Restricting to the latest file prevents double-counting when historical
        files for the same run_id accumulate across pipeline runs.
        Predicate pushdown on run_id skips row groups that cannot match,
        keeping reads O(target_rows) at WOMD scale.
        The filename metadata column is excluded from the returned DataFrame.
        """
        return self._con.execute(
            """
            SELECT * EXCLUDE (filename) FROM logs
            WHERE run_id = ?
              AND filename = (SELECT MAX(filename) FROM logs WHERE run_id = ?)
            """,
            [run_id, run_id],
        ).df()

    def load_all(self) -> pd.DataFrame:
        """Return all rows as a DataFrame without the filename column.

        Used for single-file baseline loads via from_file().
        """
        return self._con.execute("SELECT * EXCLUDE (filename) FROM logs").df()

    # ── resource management ───────────────────────────────────────────────────

    def close(self) -> None:
        """Close the DuckDB connection."""
        try:
            self._con.close()
        except Exception:
            pass

    def __enter__(self) -> "LogStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
