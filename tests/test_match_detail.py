import json

from test_match import api_match

from deadlock_predictor.data import match_detail as md


def detail_match(**kwargs):
    """`api_match` plus what only the API has: the death list.

    Slot 0 (Team0) kills slot 6 (Team1) at 300 s and 310 s; slot 7 dies to its own
    team's slot 8 at 400 s (no enemy credit).
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


def test_missing_sections_give_empty_frames():
    info = api_match()
    assert md.deaths(info).empty
