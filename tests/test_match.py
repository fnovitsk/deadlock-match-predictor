from datetime import UTC, datetime

import duckdb
import pandas as pd
import pytest
from test_snapshots import STATS, TIMES, match_rows

from deadlock_predictor.data.columns import COLUMNS
from deadlock_predictor.data.match import api_rows, lobby_badge, local_rows, register
from deadlock_predictor.features.heroes import build_heroes
from deadlock_predictor.features.snapshots import build_snapshots, filter_matches

OBJECTIVES = [("Team1", "Tier1Lane3", 400), ("Team0", "TitanShieldGenerator1", 700)]
MID_BOSS = [("Team0", "Team1", 650)]
TEAM_CODE = {"Team0": 0, "Team1": 1}
OBJECTIVE_CODE = {"Tier1Lane3": 3, "TitanShieldGenerator1": 10}


def api_match(match_id=1, duration_s=1000):
    """The API's `match_info` for the same match `match_rows` builds (6 v 6, Team0 ahead)."""
    players = []
    for slot in range(12):
        team = 0 if slot < 6 else 1
        worth = 100 if team == 0 else 50
        players.append(
            {
                "account_id": 1000 + slot,
                "player_slot": slot,
                "team": team,
                "hero_id": slot + 1,
                "assigned_lane": 1,
                "kills": 1,
                "deaths": 1,
                "assists": 1,
                "net_worth": worth,
                "level": 1,
                "abandon_match_time_s": 0,
                "player_rank_data": {"initial_display_rank": 51},
                # stats arrive unordered in the JSON; api_rows sorts them by time
                "stats": [
                    {"time_stamp_s": t, **{s: worth if s == "net_worth" else 1 for s in STATS}}
                    for t in reversed(TIMES)
                ],
            }
        )
    return {
        "match_id": match_id,
        "start_time": int(datetime(2026, 10, 1, tzinfo=UTC).timestamp()),
        "duration_s": duration_s,
        "match_mode": 4,
        "game_mode": 1,
        "game_mode_version": 2,
        "average_badge_team0": 0,
        "average_badge_team1": 0,
        "winning_team": 0,
        "match_outcome": 0,
        "objectives_mask_team0": 0,
        "objectives_mask_team1": 0,
        "not_scored": False,
        "low_pri_pool": False,
        "new_player_pool": False,
        "objectives": [
            {
                "team": TEAM_CODE[owner],
                "team_objective_id": OBJECTIVE_CODE[kind],
                "destroyed_time_s": t,
                "first_damage_time_s": t - 50,
            }
            for owner, kind, t in OBJECTIVES
        ],
        "mid_boss": [
            {"team_claimed": TEAM_CODE[c], "team_killed": TEAM_CODE[k], "destroyed_time_s": t}
            for c, k, t in MID_BOSS
        ],
        "players": players,
    }


def features(rows):
    con = duckdb.connect()
    register(con, rows)
    filter_matches(con)
    return build_snapshots(con).df(), build_heroes(con).df()


def test_api_rows_have_the_dump_columns_and_names():
    rows = api_rows(api_match())
    assert list(rows.columns) == COLUMNS and len(rows) == 12
    assert set(rows.team) == {"Team0", "Team1"} and set(rows.winning_team) == {"Team0"}
    assert set(rows.match_mode) == {"Ranked"} and set(rows.game_mode) == {"Normal"}
    assert rows["objectives.team_objective"][0] == ["Tier1Lane3", "TitanShieldGenerator1"]
    assert rows["mid_boss.team_claimed"][0] == ["Team0"]
    assert rows["stats.time_stamp_s"][0] == TIMES  # sorted, final snapshot kept
    assert set(rows.average_badge) == {51}


def test_api_and_dump_give_the_same_features():
    dump = match_rows(1, objectives=OBJECTIVES, mid_boss=MID_BOSS)
    snaps_dump, heroes_dump = features(pd.DataFrame(dump))
    snaps_api, heroes_api = features(api_rows(api_match()))
    # The test fixture's badge (50) isn't a real badge; compare everything else.
    cols = [c for c in snaps_dump.columns if c != "badge_mean"]
    pd.testing.assert_frame_equal(snaps_dump[cols], snaps_api[cols], check_dtype=False)
    pd.testing.assert_frame_equal(heroes_dump, heroes_api, check_dtype=False)


def test_empty_mid_boss_lists_work():
    info = api_match()
    info["mid_boss"] = []
    snaps, _ = features(api_rows(info))
    assert (snaps.mid_boss_claims_diff == 0).all()


@pytest.mark.parametrize(
    "ranks, badge",
    [
        # Real lobbies (dump average_badge on the right).
        ([0, 84, 85, 86, 86, 86, 86, 91, 92, 95, 95, 102], 92),
        ([85, 86, 86, 86, 86, 86, 91, 91, 92, 92, 92, 93], 91),
        ([94, 95, 96, 101, 101, 103, 103, 104, 104, 105, 106, 106], 103),  # mean 62.5
        ([0] * 12, 0),
    ],
)
def test_lobby_badge(ranks, badge):
    assert lobby_badge(ranks) == badge


def test_local_rows(tmp_path):
    rows = pd.concat([api_rows(api_match(1_000_005)), api_rows(api_match(1_000_007))])
    rows["created_at"] = datetime(2026, 10, 2, tzinfo=UTC)
    rows.to_parquet(tmp_path / "partition=1.parquet")
    found = local_rows(1_000_007, tmp_path)
    assert found.match_id.unique().tolist() == [1_000_007] and len(found) == 12
    assert local_rows(1_000_009, tmp_path) is None
    assert local_rows(2_000_000, tmp_path) is None  # no partition file
