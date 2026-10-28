"""Chronological train/validation/test split, by match start time.

Training on older matches and testing on newer ones mimics how the model is used: it is
always predicting games played after the ones it learned from. Splitting by match (all of a
match's snapshots share its start time) keeps snapshots of one game out of two splits.
"""

from __future__ import annotations

from datetime import datetime

import duckdb

SPLITS = ("train", "validation", "test")


def split_boundaries(
    con: duckdb.DuckDBPyConnection,
    matches: str = "matches",
    train: float = 0.70,
    validation: float = 0.15,
) -> tuple[datetime, datetime]:
    """Start times where validation and test begin, so that the oldest `train` share of
    matches is train, the next `validation` share validation, and the rest test."""
    if not (0 < train and 0 < validation and train + validation < 1):
        raise ValueError("need 0 < train, 0 < validation, train + validation < 1")
    # i is each match's 0-based position by start time; round() because 0.7 + 0.15 is
    # 0.8499999... in floating point.
    return con.sql(
        f"""WITH ordered AS (
                SELECT start_time,
                       row_number() OVER (ORDER BY start_time) - 1 AS i,
                       count(*) OVER () AS n
                FROM {matches}
            )
            SELECT min(start_time) FILTER (i >= round(n * {train})),
                   min(start_time) FILTER (i >= round(n * {train + validation}))
            FROM ordered"""
    ).fetchall()[0]


def split_case(boundaries: tuple[datetime, datetime], column: str = "start_time") -> str:
    """SQL expression naming the split of a row with the given start time."""
    validation_start, test_start = (b.isoformat() for b in boundaries)
    return f"""CASE
        WHEN {column} < TIMESTAMPTZ '{validation_start}' THEN 'train'
        WHEN {column} < TIMESTAMPTZ '{test_start}' THEN 'validation'
        ELSE 'test' END"""
