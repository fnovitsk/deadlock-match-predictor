"""Access to the deadlock-api.com Parquet data dumps.

The manifest lists every Parquet file per table. `match_player` is partitioned by
``match_id // 1_000_000`` (higher partition = newer matches) and has two kinds of files:
``base`` files (one per partition) and hourly ``delta`` files spanning many partitions.
"""

from __future__ import annotations

import duckdb
import requests

MANIFEST_URL = "https://data.deadlock-api.com/v1/manifest.json"

# The dump server rejects Python's default user agent with 403 Forbidden.
USER_AGENT = "deadlock-match-predictor (github.com/fnovitsk/deadlock-match-predictor)"


def fetch_manifest(url: str = MANIFEST_URL, timeout: float = 60) -> dict:
    """Download the dump manifest."""
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _in_range(partition: int, min_partition: int | None, max_partition: int | None) -> bool:
    if min_partition is not None and partition < min_partition:
        return False
    return max_partition is None or partition <= max_partition


def match_player_urls(
    manifest: dict,
    min_partition: int | None = None,
    max_partition: int | None = None,
    include_deltas: bool = True,
) -> list[str]:
    """URLs of the match_player files covering partitions in [min_partition, max_partition].

    A delta file is included if it holds rows for any partition in the range.
    """
    base_url = manifest["public_url"].rstrip("/")
    urls = []
    for file in manifest["tables"]["match_player"]["files"]:
        if file["kind"] == "base":
            keep = _in_range(file["partition"], min_partition, max_partition)
        elif file["kind"] == "delta" and include_deltas:
            keep = any(
                _in_range(int(p), min_partition, max_partition)
                for p in file.get("rows_by_partition", {})
            )
        else:
            keep = False
        if keep:
            urls.append(f"{base_url}/{file['key']}")
    return urls


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """DuckDB connection that sends our user agent on HTTP(S) reads."""
    con = duckdb.connect(database)
    con.sql(
        f"CREATE SECRET http (TYPE http, EXTRA_HTTP_HEADERS MAP {{'User-Agent': '{USER_AGENT}'}})"
    )
    return con
