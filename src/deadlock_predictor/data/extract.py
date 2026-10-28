"""Download ranked matches since a start time, one local Parquet file per partition.

Each partition is read from its base file plus every delta file that holds rows for it,
de-duplicated on (match_id, player_slot). Finished partitions are skipped on rerun; the
newest ones, which are still filling up, are always refreshed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

from deadlock_predictor.data.columns import select_list
from deadlock_predictor.data.dumps import match_player_urls

PARTITION_SIZE = 1_000_000


def partitions(manifest: dict) -> list[int]:
    """Partitions that have a base file or delta rows, ascending."""
    found = set()
    for file in manifest["tables"]["match_player"]["files"]:
        if file["kind"] == "base":
            found.add(file["partition"])
        else:
            found.update(int(p) for p in file.get("rows_by_partition", {}))
    return sorted(found)


def partition_path(out_dir: Path, partition: int) -> Path:
    return out_dir / f"partition={partition}.parquet"


def extract_query(urls: list[str], partition: int, columns: str | None = None) -> str:
    """SQL for one partition's ranked rows, base and delta combined, without duplicates.

    Parameters: $match_mode, $game_mode, $start_time. `urls` are inlined because a
    delta file holds many partitions, so the match_id range filter is what keeps it small.
    """
    columns = columns or select_list()
    files = ", ".join(f"'{u}'" for u in urls)
    lo, hi = partition * PARTITION_SIZE, (partition + 1) * PARTITION_SIZE
    return f"""
        SELECT {columns}
        FROM read_parquet([{files}], union_by_name = true)
        WHERE match_id >= {lo} AND match_id < {hi}
          AND match_mode = $match_mode AND game_mode = $game_mode
          AND start_time >= $start_time
        QUALIFY row_number() OVER (PARTITION BY match_id, player_slot ORDER BY created_at DESC) = 1
        ORDER BY match_id, player_slot
    """


def extract_partition(
    con: duckdb.DuckDBPyConnection,
    urls: list[str],
    partition: int,
    out_path: Path,
    start_time: datetime,
    match_mode: str = "Ranked",
    game_mode: str = "Normal",
    columns: str | None = None,
) -> int:
    """Write one partition's ranked rows to `out_path`; returns the row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = {"match_mode": match_mode, "game_mode": game_mode, "start_time": start_time}
    tmp = out_path.with_suffix(".tmp")
    con.sql(extract_query(urls, partition, columns), params=params).write_parquet(str(tmp))
    tmp.replace(out_path)  # only a complete file ever has the final name
    return con.sql(f"SELECT count(*) FROM '{out_path.as_posix()}'").fetchall()[0][0]


def max_start_time(con: duckdb.DuckDBPyConnection, urls: list[str]) -> datetime | None:
    files = ", ".join(f"'{u}'" for u in urls)
    return con.sql(
        f"SELECT max(start_time) FROM read_parquet([{files}], union_by_name = true)"
    ).fetchall()[0][0]


def first_partition(con: duckdb.DuckDBPyConnection, manifest: dict, start_time: datetime) -> int:
    """Oldest partition with any match starting at or after `start_time`.

    Walks back from the newest partition, reading only start_time from base files.
    """
    parts = partitions(manifest)
    first = parts[-1]
    for partition in reversed(parts):
        urls = match_player_urls(manifest, partition, partition, include_deltas=False)
        if not urls:
            continue
        latest = max_start_time(con, urls)
        if latest is not None and latest < start_time:
            break
        first = partition
    return first


def extract(
    con: duckdb.DuckDBPyConnection,
    manifest: dict,
    out_dir: Path,
    start_time: datetime,
    match_mode: str = "Ranked",
    game_mode: str = "Normal",
    refresh_latest: int = 2,
    force: bool = False,
) -> list[Path]:
    """Extract every partition from the first one after `start_time`; returns the files."""
    first = first_partition(con, manifest, start_time)
    todo = [p for p in partitions(manifest) if p >= first]
    refresh = set(todo[-refresh_latest:]) if refresh_latest else set()
    paths = []
    for partition in todo:
        path = partition_path(out_dir, partition)
        paths.append(path)
        if path.exists() and not force and partition not in refresh:
            print(f"partition {partition}: exists, skipped")
            continue
        urls = match_player_urls(manifest, partition, partition)
        rows = extract_partition(
            con, urls, partition, path, start_time, match_mode=match_mode, game_mode=game_mode
        )
        print(f"partition {partition}: {rows:,} rows from {len(urls)} files")
    return paths
