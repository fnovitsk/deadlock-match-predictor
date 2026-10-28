import numpy as np
import pytest
from test_curve import models  # noqa: F401  (pytest fixture)
from test_match_detail import detail_match

from deadlock_predictor.data import match_detail as md
from deadlock_predictor.data.match import api_rows
from deadlock_predictor.features.snapshots import FEATURES
from deadlock_predictor.viz.curve import match_features
from deadlock_predictor.viz.timeline import event_points, features_at


def bounty_match():
    """`detail_match` where each Team0 player earns 100 objective souls between the 360 s
    and 540 s snapshots: the Lane Guardian Team0 takes at 400 s."""
    info = detail_match()
    for p in info["players"]:
        for s in p["stats"]:
            s["gold_boss"] = 100 if p["team"] == 0 and s["time_stamp_s"] >= 540 else 0
    return info


def parts(info):
    rows = api_rows(info)
    snaps, heroes, _ = match_features(rows)
    return snaps, heroes, rows, md.deaths(info), md.player_stats(info)


def swap_teams(info):
    """The same match with the two teams exchanged."""
    flip = {0: 1, 1: 0}
    out = {**info, "winning_team": flip[info["winning_team"]]}
    out["players"] = [{**p, "team": flip[p["team"]]} for p in info["players"]]
    out["objectives"] = [{**o, "team": flip[o["team"]]} for o in info["objectives"]]
    out["mid_boss"] = [
        {**b, "team_claimed": flip[b["team_claimed"]], "team_killed": flip[b["team_killed"]]}
        for b in info["mid_boss"]
    ]
    return out


@pytest.mark.parametrize("with_stats", [False, True])
def test_training_rows_at_snapshots(with_stats):
    snaps, _, rows, deaths, stats = parts(bounty_match())
    at = features_at(snaps, rows, deaths, snaps.t, stats if with_stats else None)
    np.testing.assert_array_equal(at[FEATURES].to_numpy(float), snaps[FEATURES].to_numpy(float))


def test_kills_change_at_the_kill():
    snaps, _, rows, deaths, _ = parts(detail_match())
    # Team0 kills at 300 and 310 s; the teamkill at 400 s counts for nobody. From the
    # 360 s snapshot on, the stats' kills (equal for both teams here) take over again.
    at = features_at(snaps, rows, deaths, [299, 300, 310, 359, 360, 401])
    assert at.kills_diff.tolist() == [0, 1, 2, 2, 0, 0]


def test_buildings_and_mid_boss_count_from_their_second():
    snaps, _, rows, deaths, _ = parts(detail_match())
    at = features_at(snaps, rows, deaths, [399, 400, 649, 650, 699, 700])
    # Team0 takes a Lane Guardian at 400 s, claims the mid-boss (killed by Team1) at 650 s,
    # and loses a Shrine at 700 s.
    assert at.obj_tier1_diff.tolist() == [0, 1, 1, 1, 1, 1]
    assert at.mid_boss_claims_diff.tolist() == [0, 0, 0, 1, 1, 1]
    assert at.mid_boss_kills_diff.tolist() == [0, 0, 0, -1, -1, -1]
    assert at.obj_shield_gen_diff.tolist() == [0, 0, 0, 0, 0, -1]


def test_building_souls_arrive_with_the_building():
    snaps, _, rows, deaths, stats = parts(bounty_match())
    at = features_at(snaps, rows, deaths, [399, 400], stats)
    # 6 players × 100 souls, all at 400 s. Net worth is flat in this match, so the rest
    # falls by 600 between snapshots: −600 / 180 s per second.
    assert at.net_worth_diff.diff().iloc[1] == pytest.approx(600 - 600 / 180)
    flat = features_at(snaps, rows, deaths, [399, 400])
    assert flat.net_worth_diff.diff().iloc[1] == pytest.approx(0)


def test_event_points(models):  # noqa: F811
    info = bounty_match()
    points = event_points(*parts(info), models, info["duration_s"])
    assert points.t.is_monotonic_increasing and points.t.is_unique
    events = points[points.kind == "event"]
    assert events.t.tolist() == [400, 650, 700]
    assert events.trigger.tolist() == [
        "Lane Guardian (Hidden King)",
        "Mid-boss (Hidden King)",
        "Shrine (Archmother)",
    ]
    assert set(points[points.kind == "before"].t) == {399, 649, 699}
    assert set(points[points.kind == "snapshot"].t) == {180, 360, 540, 720}


def test_event_points_are_symmetric(models):  # noqa: F811
    info = bounty_match()
    points = event_points(*parts(info), models, info["duration_s"])
    flipped = event_points(*parts(swap_teams(info)), models, info["duration_s"])
    np.testing.assert_array_equal(points.t, flipped.t)
    for name in ("p_final", "p_baseline"):
        np.testing.assert_allclose(points[name] + flipped[name], 1.0, atol=1e-9)
