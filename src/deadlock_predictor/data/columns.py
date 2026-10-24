"""The match_player columns we read; the full table has 157."""

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


def select_list(columns: list[str] = COLUMNS) -> str:
    """Comma-separated, quoted column names for a SQL SELECT (names contain dots)."""
    return ", ".join(f'"{c}"' for c in columns)
