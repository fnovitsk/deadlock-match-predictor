"""Build the snapshot table: one row per (match, snapshot time), from Team0's view.

Every feature at time t uses only what was knowable at t:
- stats come from the snapshot at t; the final snapshot (taken as the match ends) is dropped,
- objectives and mid-boss kills count only if they happened at or before t,
- end-of-match columns (`objectives_mask_*`, final totals) are never read.

Diff features are Team0 minus Team1, so positive values favour Team0.
"""

from __future__ import annotations

import duckdb

STATS = [
    "net_worth",
    "kills",
    "deaths",
    "assists",
    "level",
    "player_damage",
    "boss_damage",
    "creep_kills",
    "denies",
]

# objectives.team_objective prefix -> feature group. Core is left out: destroying it ends
# the match, so it is never destroyed before the last (dropped) snapshot.
OBJECTIVE_GROUPS = {
    "Tier1": "tier1",
    "Tier2": "tier2",
    "BarrackBoss": "barracks",
    "TitanShieldGenerator": "shield_gen",
    "Titan": "titan",
}

FEATURES = (
    ["t"]
    + [f"{s}_diff" for s in STATS]
    + [f"obj_{g}_diff" for g in OBJECTIVE_GROUPS.values()]
    + ["mid_boss_claims_diff", "mid_boss_kills_diff", "badge_mean"]
)
LABEL = "team0_win"


def _sign(team_column: str) -> str:
    """+1 for a Team0 row, -1 for a Team1 row."""
    return f"CASE WHEN {team_column} = 'Team0' THEN 1 ELSE -1 END"


def _objective_group(type_column: str) -> str:
    # TitanShieldGenerator must be tested before Titan (shared prefix).
    cases = sorted(OBJECTIVE_GROUPS.items(), key=lambda kv: -len(kv[0]))
    whens = " ".join(f"WHEN {type_column} LIKE '{p}%' THEN '{g}'" for p, g in cases)
    return f"CASE {whens} END"


def filter_matches(con: duckdb.DuckDBPyConnection, raw: str = "raw") -> dict[str, int]:
    """Create table `matches` with the matches that pass every filter.

    Returns how many matches each filter removed (a match can fail several).
    """
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE match_checks AS
        SELECT
            match_id,
            any_value(start_time) AS start_time,
            any_value(duration_s) AS duration_s,
            any_value(winning_team) AS winning_team,
            count(*) = 12
                AND count(*) FILTER (team = 'Team0') = 6
                AND count(*) FILTER (team = 'Team1') = 6 AS full_teams,
            NOT bool_or(coalesce(abandon_match_time_s, 0) > 0) AS no_abandon,
            NOT bool_or(coalesce(not_scored, false)) AS scored,
            any_value(winning_team) IN ('Team0', 'Team1') AS has_winner
        FROM {raw}
        GROUP BY match_id
    """)
    checks = ["full_teams", "no_abandon", "scored", "has_winner"]
    removed = con.sql(
        "SELECT count(*), "
        + ", ".join(f"count(*) FILTER (NOT {c})" for c in checks)
        + " FROM match_checks"
    ).fetchone()
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE matches AS
        SELECT match_id, start_time, duration_s, winning_team = 'Team0' AS {LABEL}
        FROM match_checks
        WHERE {" AND ".join(checks)}
    """)
    kept = con.sql("SELECT count(*) FROM matches").fetchone()[0]
    return {"matches": removed[0], **dict(zip(checks, removed[1:], strict=True)), "kept": kept}


def build_snapshots(con: duckdb.DuckDBPyConnection, raw: str = "raw") -> duckdb.DuckDBPyRelation:
    """Snapshot features for every match in table `matches` (see `filter_matches`)."""
    unnest_stats = ", ".join(f'unnest("stats.{s}")::BIGINT AS {s}' for s in STATS)
    stat_diffs = ", ".join(f"sum({_sign('team')} * {s}) AS {s}_diff" for s in STATS)
    obj_diffs = ", ".join(
        f"coalesce(sum(o.sign) FILTER (o.grp = '{g}'), 0) AS obj_{g}_diff"
        for g in OBJECTIVE_GROUPS.values()
    )
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE player_snaps AS
        SELECT r.match_id, r.team, unnest("stats.time_stamp_s") AS t, {unnest_stats}
        FROM {raw} r SEMI JOIN matches USING (match_id)
    """)
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE team_snaps AS
        SELECT p.match_id, p.t, {stat_diffs}
        FROM player_snaps p JOIN matches m USING (match_id)
        WHERE p.t > 0 AND p.t < m.duration_s  -- drop the leaky end-of-match snapshot
        GROUP BY p.match_id, p.t
        HAVING count(*) = 12  -- every player has this snapshot
    """)
    # Objectives and mid-boss lists are match-level: identical on all 12 player rows.
    event_columns = [
        "objectives.team_objective",
        "objectives.team",
        "objectives.destroyed_time_s",
        "mid_boss.team_claimed",
        "mid_boss.team_killed",
        "mid_boss.destroyed_time_s",
    ]
    events = ", ".join(f'any_value("{c}") AS "{c}"' for c in event_columns)
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT match_id, {events}
        FROM {raw} SEMI JOIN matches USING (match_id)
        GROUP BY match_id
    """)
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE objectives AS
        SELECT match_id, {_objective_group("type")} AS grp,
               -{_sign("owner")} AS sign,  -- losing your own objective is bad for you
               destroyed_s
        FROM (
            SELECT match_id,
                   unnest("objectives.team_objective") AS type,
                   unnest("objectives.team") AS owner,
                   unnest("objectives.destroyed_time_s") AS destroyed_s
            FROM events
        )
        WHERE destroyed_s > 0 AND grp IS NOT NULL
    """)
    con.sql("""
        CREATE OR REPLACE TEMP TABLE mid_boss AS
        SELECT match_id,
               unnest("mid_boss.team_claimed") AS claimed_by,
               unnest("mid_boss.team_killed") AS killed_by,
               unnest("mid_boss.destroyed_time_s") AS destroyed_s
        FROM events
    """)
    # average_badge is the lobby's average rank, the same on every player row, so there
    # is no per-team rank gap to compute.
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE badges AS
        SELECT match_id, avg(average_badge) AS badge_mean
        FROM {raw} SEMI JOIN matches USING (match_id)
        WHERE average_badge > 0
        GROUP BY match_id
    """)
    return con.sql(f"""
        WITH obj AS (
            SELECT s.match_id, s.t, {obj_diffs}
            FROM team_snaps s
            LEFT JOIN objectives o
              ON o.match_id = s.match_id AND o.destroyed_s <= s.t
            GROUP BY s.match_id, s.t
        ),
        boss AS (
            SELECT s.match_id, s.t,
                   coalesce(sum({_sign("b.claimed_by")}) FILTER (b.claimed_by IS NOT NULL), 0)
                     AS mid_boss_claims_diff,
                   coalesce(sum({_sign("b.killed_by")}) FILTER (b.killed_by IS NOT NULL), 0)
                     AS mid_boss_kills_diff
            FROM team_snaps s
            LEFT JOIN mid_boss b
              ON b.match_id = s.match_id AND b.destroyed_s > 0 AND b.destroyed_s <= s.t
            GROUP BY s.match_id, s.t
        )
        SELECT s.match_id, m.start_time, m.duration_s,
               {", ".join(FEATURES)}, m.{LABEL}
        FROM team_snaps s
        JOIN matches m USING (match_id)
        JOIN obj USING (match_id, t)
        JOIN boss USING (match_id, t)
        LEFT JOIN badges USING (match_id)
        ORDER BY s.match_id, s.t
    """)
