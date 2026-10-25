"""Build the hero table: which heroes each team picked, one row per (match, hero).

Picks are known before the game starts, so they are safe features at every t. A hero
appears at most once per match, so `sign` (+1 Team0, −1 Team1) says who has it.
"""

from __future__ import annotations

import duckdb


def build_heroes(con: duckdb.DuckDBPyConnection, raw: str = "raw") -> duckdb.DuckDBPyRelation:
    """Hero picks for every match in table `matches` (see `filter_matches`)."""
    return con.sql(f"""
        SELECT match_id, hero_id, CASE WHEN team = 'Team0' THEN 1 ELSE -1 END::TINYINT AS sign
        FROM {raw} SEMI JOIN matches USING (match_id)
        ORDER BY match_id, hero_id
    """)
