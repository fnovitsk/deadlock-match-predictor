import json

import numpy as np
from test_match import api_match

from deadlock_predictor.data import match_detail as md


def detail_match(**kwargs):
    """`api_match` plus what only the API has: deaths, items, buffs and paths.

    Slot 0 (Team0) kills slot 6 (Team1) at 300 s and 310 s; slot 7 dies to its own
    team's slot 8 at 400 s (no enemy credit); slot 1 buys an item at 100 s and sells it
    at 500 s; slot 2 picks up a buff at 200 s.
    """
    info = api_match(**kwargs)
    deaths = {6: [(300, 0), (310, 0)], 7: [(400, 8)]}
    for p in info["players"]:
        p["death_details"] = [
            {
                "game_time_s": t,
                "killer_player_slot": killer,
                "death_pos": {"x": 10.0, "y": 20.0},
                "death_duration_s": 15,
            }
            for t, killer in deaths.get(p["player_slot"], [])
        ]
        p["items"] = (
            [{"game_time_s": 100, "item_id": 555, "sold_time_s": 500}]
            if p["player_slot"] == 1
            else []
        )
        p["power_up_buffs"] = (
            [{"type": "hp_permanent_pickup_lv2", "pickup_times_s": [200]}]
            if p["player_slot"] == 2
            else []
        )
    info["match_paths"] = {
        "interval_s": 1.0,
        "x_resolution": 100,
        "y_resolution": 100,
        "paths": [
            {
                "player_slot": slot,
                "x_min": -1000.0,
                "x_max": 1000.0,
                "y_min": 0.0,
                "y_max": 500.0,
                "x_pos": [0, 50, 100],
                "y_pos": [100, 100, 0],
                "health": [100, 80, 0],
            }
            for slot in range(12)
        ],
    }
    return info


def test_fetch_detail_uses_the_cache(tmp_path, monkeypatch):
    info = detail_match(match_id=5)
    (tmp_path / "5.json").write_text(json.dumps(info))
    monkeypatch.setattr(md, "fetch_match", lambda _: (_ for _ in ()).throw(AssertionError))
    assert md.fetch_detail(5, cache_dir=tmp_path)["match_id"] == 5


def test_fetch_detail_caches_the_api(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "fetch_match", lambda mid: detail_match(match_id=mid))
    md.fetch_detail(9, cache_dir=tmp_path)
    assert json.loads((tmp_path / "9.json").read_text())["match_id"] == 9


def test_players_team0_first():
    players = md.players(detail_match())
    assert players.team.tolist() == ["Team0"] * 6 + ["Team1"] * 6
    assert players.hero_id.tolist() == list(range(1, 13))


def test_player_stats_one_row_per_snapshot():
    stats = md.player_stats(detail_match())
    assert len(stats) == 12 * stats.t.nunique()
    assert (stats.groupby("slot").t.apply(lambda t: t.is_monotonic_increasing)).all()
    assert set(stats[stats.team == "Team0"].net_worth) == {100}


def test_deaths_credit_only_enemy_kills():
    deaths = md.deaths(detail_match())
    assert deaths.t.tolist() == [300, 310, 400]
    assert deaths.killer_team.tolist()[:2] == ["Team0", "Team0"]
    assert deaths.iloc[2][["killer", "killer_team"]].isna().all()  # killed by a teammate
    assert (deaths.victim_team == "Team1").all()


def test_items_and_buffs():
    items = md.items(detail_match())
    assert items[["slot", "t", "item_id", "sold_t"]].values.tolist() == [[1, 100, 555, 500]]
    buffs = md.buffs(detail_match())
    assert buffs[["slot", "t"]].values.tolist() == [[2, 200]]


def test_paths_in_world_units():
    paths = md.paths(detail_match())
    one = paths[paths.slot == 0]
    assert one.t.tolist() == [0, 1, 2]
    assert np.allclose(one.x, [-1000, 0, 1000]) and np.allclose(one.y, [500, 500, 0])
    assert (one.team == "Team0").all() and one.health.tolist() == [100, 80, 0]


def test_missing_sections_give_empty_frames():
    info = api_match()
    assert md.deaths(info).empty and md.items(info).empty and md.buffs(info).empty
    assert md.paths(info).empty
