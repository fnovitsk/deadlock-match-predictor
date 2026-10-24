from datetime import UTC, datetime

import duckdb
import pytest

from deadlock_predictor.data.extract import (
    extract,
    extract_partition,
    first_partition,
    partitions,
)

START = datetime(2026, 9, 29, tzinfo=UTC)
COLUMNS = '"match_id", "player_slot", "created_at"'


def write_rows(con, path, rows):
    """rows: (match_id, player_slot, match_mode, game_mode, start_time, created_at)."""
    values = ", ".join(
        f"({m}, {s}, '{mode}', '{game}', TIMESTAMPTZ '{t}', TIMESTAMPTZ '{c}')"
        for m, s, mode, game, t, c in rows
    )
    con.sql(
        f"""COPY (SELECT * FROM (VALUES {values})
            t(match_id, player_slot, match_mode, game_mode, start_time, created_at))
        TO '{path.as_posix()}' (FORMAT parquet)"""
    )


@pytest.fixture
def dump(tmp_path):
    """A fake dump: base files for partitions 1 and 2, one delta touching both."""
    con = duckdb.connect()
    old, new = "2026-09-01 00:00:00+00", "2026-10-01 00:00:00+00"
    c1, c2 = "2026-10-01 01:00:00+00", "2026-10-02 01:00:00+00"
    write_rows(con, tmp_path / "base1.parquet", [(1_000_001, 1, "Ranked", "Normal", old, c1)])
    write_rows(
        con,
        tmp_path / "base2.parquet",
        [
            (2_000_001, 1, "Ranked", "Normal", new, c1),
            (2_000_001, 2, "Ranked", "Normal", new, c1),
            (2_000_002, 1, "Unranked", "Normal", new, c1),
            (2_000_003, 1, "Ranked", "StreetBrawl", new, c1),
        ],
    )
    write_rows(
        con,
        tmp_path / "delta.parquet",
        [
            (2_000_001, 2, "Ranked", "Normal", new, c2),  # duplicate of a base row
            (2_000_004, 1, "Ranked", "Normal", new, c2),  # only in the delta
            (1_000_002, 1, "Ranked", "Normal", old, c2),  # another partition, too old
        ],
    )
    manifest = {
        "public_url": tmp_path.as_posix(),
        "tables": {
            "match_player": {
                "files": [
                    {"kind": "base", "partition": 1, "key": "base1.parquet"},
                    {"kind": "base", "partition": 2, "key": "base2.parquet"},
                    {
                        "kind": "delta",
                        "key": "delta.parquet",
                        "rows_by_partition": {"1": 1, "2": 2},
                    },
                ]
            }
        },
    }
    return con, manifest, tmp_path


def test_partitions_include_delta_only_partitions():
    manifest = {
        "tables": {
            "match_player": {
                "files": [
                    {"kind": "base", "partition": 5, "key": "a"},
                    {"kind": "delta", "key": "b", "rows_by_partition": {"6": 1}},
                ]
            }
        }
    }
    assert partitions(manifest) == [5, 6]


def test_extract_partition_filters_and_deduplicates(dump):
    con, _, root = dump
    urls = [(root / f).as_posix() for f in ("base2.parquet", "delta.parquet")]
    out = root / "out" / "partition=2.parquet"
    rows = extract_partition(con, urls, 2, out, START, columns=COLUMNS)
    got = con.sql(f"SELECT match_id, player_slot, created_at FROM '{out.as_posix()}'").fetchall()
    assert rows == 3
    assert [(m, s) for m, s, _ in got] == [(2_000_001, 1), (2_000_001, 2), (2_000_004, 1)]
    # The newer (delta) copy of the duplicate wins.
    assert got[1][2] == datetime(2026, 10, 2, 1, tzinfo=UTC)


def test_first_partition_skips_partitions_before_start(dump):
    con, manifest, _ = dump
    assert first_partition(con, manifest, START) == 2


def test_extract_skips_finished_partitions(dump, capsys):
    con, manifest, root = dump
    out_dir = root / "out"
    out_dir.mkdir()
    (out_dir / "partition=2.parquet").write_bytes(b"not parquet")
    paths = extract(con, manifest, out_dir, START, refresh_latest=0)
    assert paths == [out_dir / "partition=2.parquet"]
    assert "exists, skipped" in capsys.readouterr().out
