from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from deadlock_predictor.data.split import split_boundaries, split_case

DAY0 = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.fixture
def con():
    """100 matches, one per hour, each with 9 snapshot rows."""
    con = duckdb.connect()
    con.sql(f"""
        CREATE TABLE matches AS
        SELECT i AS match_id, TIMESTAMPTZ '{DAY0.isoformat()}' + to_hours(i) AS start_time
        FROM range(100) r(i)
    """)
    con.sql("""
        CREATE TABLE snapshots AS
        SELECT m.*, t FROM matches m, range(180, 1800, 180) s(t)
    """)
    return con


def assign(con):
    case = split_case(split_boundaries(con))
    return con.sql(f"SELECT *, {case} AS split FROM snapshots")


def test_split_sizes(con):
    counts = dict(assign(con).aggregate("split, count(DISTINCT match_id)", "split").fetchall())
    assert counts == {"train": 70, "validation": 15, "test": 15}


def test_no_match_in_two_splits(con):
    multi = assign(con).aggregate("match_id, count(DISTINCT split) AS n", "match_id")
    assert multi.filter("n > 1").fetchall() == []


def test_splits_are_in_date_order(con):
    spans = dict(
        (s, (lo, hi))
        for s, lo, hi in assign(con)
        .aggregate("split, min(start_time), max(start_time)", "split")
        .fetchall()
    )
    assert spans["train"][1] < spans["validation"][0]
    assert spans["validation"][1] < spans["test"][0]
    assert spans["test"][1] == DAY0 + timedelta(hours=99)


def test_rejects_bad_fractions(con):
    with pytest.raises(ValueError):
        split_boundaries(con, train=0.9, validation=0.2)
