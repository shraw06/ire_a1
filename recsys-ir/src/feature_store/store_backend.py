"""Swappable storage backend for feature stores - Parquet + DuckDB.

Provides a thin wrapper around DuckDB's ability to read Parquet files
directly, giving columnar-efficient selective reads (e.g. only
``title``/``abstract`` for BM25, only ``embedding`` for ANN) plus
SQL-style filtering - without loading the entire file into memory.

Design rationale (vs. alternatives):
  - Plain pickled pandas: loses columnar efficiency, forces load-everything.
  - SQLite: loses columnar read efficiency for wide text/embedding columns,
    adds unneeded write-lock contention.
  - Full feature-store framework (Feast): correct shape for production, but
    overkill setup/learning cost for an assignment.
  - Parquet + DuckDB: no server process, one pure-Python-installable dep,
    fast selective reads, SQL-style time-filtered joins, no ingestion step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


class ParquetStore:
    """Thin query layer over a single Parquet file via DuckDB.

    DuckDB reads the Parquet file on demand - no full materialization at
    init time.  Column projection and filter pushdown happen inside DuckDB's
    engine, so only the requested data is scanned.

    Parameters
    ----------
    parquet_path : Path
        Absolute or relative path to the ``.parquet`` file.
    table_alias : str, optional
        SQL alias for the Parquet view (default: ``"t"``).
    """

    def __init__(self, parquet_path: Path | str, table_alias: str = "t") -> None:
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        self.alias = table_alias
        # In-memory DuckDB connection - no file, no write-lock.
        self._con = duckdb.connect(database=":memory:")
        # Register the Parquet as a view for convenient SQL access.
        self._con.execute(
            f"CREATE VIEW {self.alias} AS SELECT * FROM read_parquet('{self.parquet_path}')"
        )
        logger.debug("ParquetStore opened: %s (alias=%s)", self.parquet_path, self.alias)

    # Single-row lookup

    def get_by_id(
        self,
        id_col: str,
        id_val: str,
        columns: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return a single row matching ``id_col == id_val``, or ``None``.

        Parameters
        ----------
        id_col : str
            Primary-key column name.
        id_val : str
            Value to match.
        columns : list[str], optional
            Column projection; all columns if ``None``.
        """
        cols = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols} FROM {self.alias} WHERE {id_col} = $1 LIMIT 1"
        result = self._con.execute(sql, [id_val]).fetchone()
        if result is None:
            return None
        col_names = [desc[0] for desc in self._con.description]
        return dict(zip(col_names, result))

    # Batch lookup

    def batch_get(
        self,
        id_col: str,
        id_vals: list[str],
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return rows matching ``id_col IN (id_vals)``.

        Parameters
        ----------
        id_col : str
            Column to filter on.
        id_vals : list[str]
            Values to match.
        columns : list[str], optional
            Column projection.
        """
        if not id_vals:
            return []
        cols = ", ".join(columns) if columns else "*"
        placeholders = ", ".join(["$" + str(i + 1) for i in range(len(id_vals))])
        sql = f"SELECT {cols} FROM {self.alias} WHERE {id_col} IN ({placeholders})"
        rows = self._con.execute(sql, id_vals).fetchall()
        col_names = [desc[0] for desc in self._con.description]
        return [dict(zip(col_names, row)) for row in rows]

    # Arbitrary SQL escape hatch

    def query_sql(
        self,
        sql: str,
        params: list | None = None,
    ) -> list[dict[str, Any]]:
        """Execute arbitrary SQL and return results as list of dicts.

        The Parquet view is available as ``self.alias`` (default ``"t"``).
        """
        result = self._con.execute(sql, params or [])
        if result.description is None:
            return []
        col_names = [desc[0] for desc in result.description]
        return [dict(zip(col_names, row)) for row in result.fetchall()]

    # Bulk read into Polars

    def get_dataframe(
        self,
        columns: list[str] | None = None,
        where: str | None = None,
        params: list | None = None,
    ) -> pl.DataFrame:
        """Read (optionally filtered) data into a Polars DataFrame.

        Parameters
        ----------
        columns : list[str], optional
            Column projection.
        where : str, optional
            SQL WHERE clause (without the ``WHERE`` keyword), e.g.
            ``"dataset = 'mind'"``.
        params : list, optional
            Positional parameters for the WHERE clause.
        """
        cols = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols} FROM {self.alias}"
        if where:
            sql += f" WHERE {where}"
        result = self._con.execute(sql, params or [])
        # Fetch as Arrow, then zero-copy into Polars.
        arrow_table = result.fetch_arrow_table()
        return pl.from_arrow(arrow_table)

    # Metadata

    def row_count(self) -> int:
        """Return the total number of rows."""
        result = self._con.execute(f"SELECT COUNT(*) FROM {self.alias}").fetchone()
        return result[0] if result else 0

    def columns(self) -> list[str]:
        """Return the list of column names."""
        result = self._con.execute(f"SELECT * FROM {self.alias} LIMIT 0")
        return [desc[0] for desc in result.description]

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._con.close()

    def __repr__(self) -> str:
        return f"ParquetStore({self.parquet_path}, rows={self.row_count()})"
