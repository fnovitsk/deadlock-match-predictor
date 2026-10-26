"""Load one match as dump rows: from the local dump if it's there, else from the API.

The API (`/v1/matches/{id}/metadata`) has the same information as the dump, in a different
shape: one JSON document, with numeric codes where the dump has names (team 0/1 instead of
'Team0'/'Team1', objective ids instead of 'Tier1Lane1', ...). `api_rows` converts it to the
dump's columns and values, so the feature code runs unchanged on both. The code tables
below were read off matches present in both sources.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

from deadlock_predictor.data.columns import COLUMNS, select_list
from deadlock_predictor.data.dumps import USER_AGENT
from deadlock_predictor.data.extract import PARTITION_SIZE, partition_path

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "match_player"
API_URL = "https://api.deadlock-api.com/v1/matches/{match_id}/metadata"

TEAMS = {0: "Team0", 1: "Team1"}
MATCH_MODES = {4: "Ranked"}
GAME_MODES = {1: "Normal"}
OUTCOMES = {0: "TeamWin"}
# team_objective_id → the dump's name. Lanes are 1-4 (lane 2 is unused in 6v6).
OBJECTIVES = {
    0: "Core",
    **{i: f"Tier1Lane{i}" for i in range(1, 5)},
    **{4 + i: f"Tier2Lane{i}" for i in range(1, 5)},
    9: "Titan",
    10: "TitanShieldGenerator1",
    11: "TitanShieldGenerator2",
    **{11 + i: f"BarrackBossLane{i}" for i in range(1, 5)},
}
STAT_NAMES = [c.removeprefix("stats.") for c in COLUMNS if c.startswith("stats.")]
STRING_LISTS = [
    "objectives.team",
    "objectives.team_objective",
    "mid_boss.team_killed",
    "mid_boss.team_claimed",
]


def _name(table: dict, code) -> str:
    return table.get(code, str(code))


def lobby_badge(ranks: list[int]) -> int:
    """The dump's `average_badge`: the lobby's mean rank, as a badge.

    A badge is tier·10 + subtier (subtier 1-6), so 86 → 91 is one step, not five. The mean
    is taken on the linear scale tier·6 + subtier, without unranked players (0), rounded
    half up (Python's `round` rounds 62.5 to 62), and turned back into a badge. This
    matched the dump on all 41 sampled matches.
    """
    steps = [r // 10 * 6 + r % 10 for r in ranks if r > 0]
    if not steps:
        return 0
    mean = int(np.floor(np.mean(steps) + 0.5))
    tier, sub = divmod(mean - 1, 6)
    return tier * 10 + sub + 1


def api_rows(info: dict) -> pd.DataFrame:
    """The 12 player rows of an API `match_info`, with the dump's columns and values."""
    start = datetime.fromtimestamp(info["start_time"], tz=UTC)
    badge = lobby_badge([p["player_rank_data"]["initial_display_rank"] for p in info["players"]])
    objectives = info["objectives"]
    bosses = info["mid_boss"]
    match = {
        "match_id": info["match_id"],
        "start_time": start,
        "duration_s": info["duration_s"],
        "match_mode": _name(MATCH_MODES, info["match_mode"]),
        "game_mode": _name(GAME_MODES, info["game_mode"]),
        "game_mode_version": info["game_mode_version"],
        "average_badge_team0": info["average_badge_team0"],
        "average_badge_team1": info["average_badge_team1"],
        "winning_team": _name(TEAMS, info["winning_team"]),
        "match_outcome": _name(OUTCOMES, info["match_outcome"]),
        "objectives_mask_team0": info["objectives_mask_team0"],
        "objectives_mask_team1": info["objectives_mask_team1"],
        "not_scored": info["not_scored"],
        "low_pri_pool": info["low_pri_pool"],
        "new_player_pool": info["new_player_pool"],
        "created_at": None,
        "objectives.destroyed_time_s": [o["destroyed_time_s"] for o in objectives],
        "objectives.first_damage_time_s": [o["first_damage_time_s"] for o in objectives],
        "objectives.team_objective": [
            _name(OBJECTIVES, o["team_objective_id"]) for o in objectives
        ],
        "objectives.team": [_name(TEAMS, o["team"]) for o in objectives],
        "mid_boss.team_killed": [_name(TEAMS, b["team_killed"]) for b in bosses],
        "mid_boss.team_claimed": [_name(TEAMS, b["team_claimed"]) for b in bosses],
        "mid_boss.destroyed_time_s": [b["destroyed_time_s"] for b in bosses],
    }
    rows = []
    for p in info["players"]:
        stats = sorted(p["stats"], key=lambda s: s["time_stamp_s"])
        rows.append(
            {
                **match,
                "account_id": p["account_id"],
                "player_slot": p["player_slot"],
                "team": _name(TEAMS, p["team"]),
                "hero_id": p["hero_id"],
                "party": None,
                "assigned_lane": p["assigned_lane"],
                "kills": p["kills"],
                "deaths": p["deaths"],
                "assists": p["assists"],
                "net_worth": p["net_worth"],
                "player_level": p["level"],
                "average_badge": badge,
                "abandon_match_time_s": p["abandon_match_time_s"],
                **{f"stats.{s}": [snap[s] for snap in stats] for s in STAT_NAMES},
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("player_slot", ignore_index=True)


def local_rows(match_id: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    """The match's player rows from the local dump, or None if it isn't there.

    Extraction already de-duplicated each partition file on (match_id, player_slot).
    """
    path = partition_path(raw_dir, match_id // PARTITION_SIZE)
    if not path.exists():
        return None
    df = duckdb.sql(
        f"SELECT {select_list()} FROM read_parquet('{path.as_posix()}') "
        f"WHERE match_id = {int(match_id)} ORDER BY player_slot"
    ).df()
    return df if len(df) else None


def fetch_match(match_id: int, timeout: float = 30) -> dict:
    """The API's `match_info` for one match."""
    url = API_URL.format(match_id=int(match_id))
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.json()["match_info"]


def match_rows(match_id: int, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, str]:
    """(rows, source): the local dump if it has the match, else the API."""
    rows = local_rows(match_id, raw_dir)
    if rows is not None:
        return rows, "dump"
    return api_rows(fetch_match(match_id)), "api"


def register(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame, name: str = "raw") -> None:
    """Make `rows` a DuckDB table with the dump's list types.

    An empty Python list carries no element type, so string lists are cast explicitly.
    """
    con.register("_rows", rows)
    casts = ", ".join(f'"{c}"::VARCHAR[] AS "{c}"' for c in STRING_LISTS)
    con.sql(f"CREATE OR REPLACE TABLE {name} AS SELECT * REPLACE ({casts}) FROM _rows")
    con.unregister("_rows")
