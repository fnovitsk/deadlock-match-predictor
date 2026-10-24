"""Save a small, reproducible sample of ranked matches for exploration.

Run: python -m deadlock_predictor.data.sample
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import yaml

from deadlock_predictor.data.dumps import connect, fetch_manifest, match_player_urls

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "data.yaml"

# Only what exploration needs; the full table has 157 columns.
MATCH_COLUMNS = [
    "match_id",
    "start_time",
    "duration_s",
    "match_mode",
    "game_mode",
    "game_mode_version",
    "average_badge_team0",
    "average_badge_team1",
    "winning_team",
    "match_outcome",
    "objectives_mask_team0",
    "objectives_mask_team1",
    "not_scored",
    "low_pri_pool",
    "new_player_pool",
    "created_at",
]
PLAYER_COLUMNS = [
    "account_id",
    "player_slot",
    "team",
    "hero_id",
    "party",
    "assigned_lane",
    "kills",
    "deaths",
    "assists",
    "net_worth",
    "player_level",
    "average_badge",
    "abandon_match_time_s",
]
STATS_COLUMNS = [
    "stats.time_stamp_s",
    "stats.net_worth",
    "stats.kills",
    "stats.deaths",
    "stats.assists",
    "stats.level",
    "stats.player_damage",
    "stats.boss_damage",
    "stats.creep_kills",
    "stats.denies",
]
EVENT_COLUMNS = [
    "objectives.destroyed_time_s",
    "objectives.first_damage_time_s",
    "objectives.team_objective",
    "objectives.team",
    "mid_boss.team_killed",
    "mid_boss.team_claimed",
    "mid_boss.destroyed_time_s",
]
COLUMNS = MATCH_COLUMNS + PLAYER_COLUMNS + STATS_COLUMNS + EVENT_COLUMNS


def save_sample(
    con: duckdb.DuckDBPyConnection,
    urls: list[str],
    n_matches: int,
    out_path: Path,
    match_mode: str = "Ranked",
    game_mode: str = "Normal",
    seed: int = 0,
) -> int:
    """Write all player rows of `n_matches` randomly chosen matches to Parquet.

    Matches are picked by ordering on a hash of match_id, so the same seed always
    picks the same matches. Returns the number of rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ", ".join(f'"{c}"' for c in COLUMNS)
    query = f"""
        WITH picked AS (
            SELECT DISTINCT match_id
            FROM read_parquet($urls)
            WHERE match_mode = $match_mode AND game_mode = $game_mode
            ORDER BY hash(match_id + $seed)
            LIMIT $n_matches
        )
        SELECT {columns}
        FROM read_parquet($urls)
        WHERE match_id IN (SELECT match_id FROM picked)
        ORDER BY match_id, player_slot
    """
    params = {
        "urls": urls,
        "match_mode": match_mode,
        "game_mode": game_mode,
        "seed": seed,
        "n_matches": n_matches,
    }
    rel = con.sql(query, params=params)
    rel.write_parquet(str(out_path))
    return con.sql(f"SELECT count(*) FROM '{out_path.as_posix()}'").fetchone()[0]


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    sample = config["sample"]
    filters = config["filters"]
    urls = match_player_urls(
        fetch_manifest(config["source"]["manifest_url"]),
        min_partition=sample["min_partition"],
        max_partition=sample["max_partition"],
        include_deltas=False,
    )
    out_path = CONFIG_PATH.parents[1] / sample["path"]
    rows = save_sample(
        connect(),
        urls,
        n_matches=sample["n_matches"],
        out_path=out_path,
        match_mode=filters["match_mode"],
        game_mode=filters["game_mode"],
        seed=sample["seed"],
    )
    print(f"Wrote {rows:,} rows from {len(urls)} files to {out_path}")


if __name__ == "__main__":
    main()
