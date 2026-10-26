from datetime import UTC, datetime

import duckdb
import pandas as pd
import pytest

from deadlock_predictor.features.snapshots import (
    FEATURES,
    LABEL,
    STATS,
    build_snapshots,
    filter_matches,
)

TIMES = [180, 360, 540, 720, 1000]  # the last one is the end-of-match snapshot


def match_rows(
    match_id,
    duration_s=1000,
    team0_net_worth=100,
    team1_net_worth=50,
    players=(6, 6),
    abandon_slot=None,
    objectives=(),
    mid_boss=(),
    times=TIMES,
):
    """Player rows for one match. objectives: (owner, type, destroyed_s);
    mid_boss: (claimed_by, killed_by, destroyed_s)."""
    rows = []
    teams = ["Team0"] * players[0] + ["Team1"] * players[1]
    for slot, team in enumerate(teams):
        worth = team0_net_worth if team == "Team0" else team1_net_worth
        row = {
            "match_id": match_id,
            "player_slot": slot,
            "team": team,
            "hero_id": slot + 1,
            "start_time": datetime(2026, 10, 1, tzinfo=UTC),
            "duration_s": duration_s,
            "winning_team": "Team0",
            "not_scored": False,
            "average_badge": 50,
            "abandon_match_time_s": 300 if slot == abandon_slot else 0,
            "stats.time_stamp_s": list(times),
            "objectives.team": [o[0] for o in objectives],
            "objectives.team_objective": [o[1] for o in objectives],
            "objectives.destroyed_time_s": [o[2] for o in objectives],
            "mid_boss.team_claimed": [b[0] for b in mid_boss],
            "mid_boss.team_killed": [b[1] for b in mid_boss],
            "mid_boss.destroyed_time_s": [b[2] for b in mid_boss],
        }
        for stat in STATS:
            row[f"stats.{stat}"] = [worth if stat == "net_worth" else 1 for _ in times]
        rows.append(row)
    return rows


def snapshots(*matches, index="t"):
    con = duckdb.connect()
    raw = pd.DataFrame([row for match in matches for row in match])  # noqa: F841
    # Empty Python lists carry no type; give the string lists the real data's VARCHAR[].
    strings = ["objectives.team", "objectives.team_objective"]
    strings += ["mid_boss.team_claimed", "mid_boss.team_killed"]
    casts = ", ".join(f'"{c}"::VARCHAR[] AS "{c}"' for c in strings)
    con.sql(f"CREATE TABLE raw AS SELECT * REPLACE ({casts}) FROM raw")
    report = filter_matches(con)
    df = build_snapshots(con).df()
    return report, df.set_index(index) if index else df


def test_final_snapshot_is_dropped():
    _, df = snapshots(match_rows(1))
    assert list(df.index) == [180, 360, 540, 720]
    assert (df.index < df.duration_s).all()


def test_diffs_are_team0_minus_team1():
    _, df = snapshots(match_rows(1, team0_net_worth=100, team1_net_worth=50))
    assert (df.net_worth_diff == 6 * 100 - 6 * 50).all()
    assert (df.kills_diff == 0).all()
    assert df[LABEL].all()


def test_objectives_count_only_once_destroyed():
    objectives = [
        ("Team1", "Tier1Lane1", 540),  # Team0 destroyed an enemy tower: +1
        ("Team0", "Tier2Lane3", 720),  # Team0 lost its own tower: -1
        ("Team1", "Tier1Lane4", 0),  # never destroyed
        ("Team1", "Core", 1000),  # ends the match; not a feature
    ]
    _, df = snapshots(match_rows(1, objectives=objectives))
    assert list(df.obj_tier1_diff) == [0, 0, 1, 1]
    assert list(df.obj_tier2_diff) == [0, 0, 0, -1]


def test_titan_and_shield_generators_are_separate_groups():
    objectives = [("Team1", "TitanShieldGenerator1", 360), ("Team1", "Titan", 720)]
    _, df = snapshots(match_rows(1, objectives=objectives))
    assert list(df.obj_shield_gen_diff) == [0, 1, 1, 1]
    assert list(df.obj_titan_diff) == [0, 0, 0, 1]


def test_mid_boss_counts_claims_and_kills_by_t():
    # Team1 killed the mid-boss at 400 but Team0 stole the claim.
    _, df = snapshots(match_rows(1, mid_boss=[("Team0", "Team1", 400)]))
    assert list(df.mid_boss_claims_diff) == [0, 0, 1, 1]
    assert list(df.mid_boss_kills_diff) == [0, 0, -1, -1]


@pytest.mark.parametrize(
    ("bad_match", "check"),
    [
        (dict(abandon_slot=3), "no_abandon"),
        (dict(players=(6, 5)), "full_teams"),
        (dict(players=(5, 7)), "full_teams"),
    ],
)
def test_bad_matches_are_dropped(bad_match, check):
    report, df = snapshots(match_rows(1), match_rows(2, **bad_match))
    assert report[check] == 1
    assert report["kept"] == 1
    assert set(df.match_id) == {1}


def test_snapshot_missing_for_a_player_is_dropped():
    rows = match_rows(1)
    rows[0]["stats.time_stamp_s"] = TIMES[:2] + TIMES[3:]  # no 540 for this player
    for stat in STATS:
        rows[0][f"stats.{stat}"] = rows[0][f"stats.{stat}"][:4]
    _, df = snapshots(rows)
    assert list(df.index) == [180, 360, 720]


def test_output_columns():
    _, df = snapshots(match_rows(1), index=None)
    assert list(df.columns) == (["match_id", "start_time", "duration_s"] + FEATURES + [LABEL])


def test_keep_failing_keeps_an_abandoned_match():
    con = duckdb.connect()
    raw = pd.DataFrame(match_rows(1, abandon_slot=3))  # noqa: F841
    strings = ["objectives.team", "objectives.team_objective"]
    strings += ["mid_boss.team_claimed", "mid_boss.team_killed"]
    casts = ", ".join(f'"{c}"::VARCHAR[] AS "{c}"' for c in strings)
    con.sql(f"CREATE TABLE raw AS SELECT * REPLACE ({casts}) FROM raw")
    assert filter_matches(con)["kept"] == 0
    report = filter_matches(con, keep_failing=True)
    assert report["kept"] == 1 and report["no_abandon"] == 1
    assert len(build_snapshots(con).df()) == 4
