import numpy as np
import pandas as pd
import requests
from test_match import api_match

from deadlock_predictor.data.match import api_rows
from deadlock_predictor.live import score


def growing_match(match_id=1):
    """api_match, with net worth growing over time: 10 souls per second per player."""
    info = api_match(match_id)
    for p in info["players"]:
        for s in p["stats"]:
            s["net_worth"] = 10 * s["time_stamp_s"] * (2 if p["team"] == 0 else 1)
    return info


def test_outcomes_report_label_and_filters():
    ok = api_rows(api_match(1))
    info = api_match(2)
    info["players"][0]["abandon_match_time_s"] = 100
    games = score.outcomes(pd.concat([ok, api_rows(info)])).set_index("match_id")
    assert games.loc[1, "team0_win"] and games.loc[1, "failed"] == []
    assert games.loc[2, "failed"] == ["no_abandon"]


def test_team_net_worth_sums_players():
    post = score.team_net_worth(api_rows(growing_match()))
    at_360 = post[post.t == 360].set_index("team").net_worth
    assert at_360["Team0"] == 6 * 7200 and at_360["Team1"] == 6 * 3600


def test_clock_check_finds_the_offset():
    post = score.team_net_worth(api_rows(growing_match()))
    # The live clock runs 60 s behind the game clock: at live t, the game is at t + 60.
    t_live = np.array([300.0, 420.0, 600.0])
    states = pd.DataFrame(
        {
            "match_id": 1,
            "t": t_live,
            "net_worth_team0": 6 * 20 * (t_live + 60),
            "net_worth_team1": 6 * 10 * (t_live + 60),
        }
    )
    clock = score.clock_check(states, post).iloc[:, 0]
    assert clock.idxmin() == 60 and clock.min() < 1e-9


def test_fetch_finished_skips_games_in_progress(monkeypatch):
    def fake(mid):
        if mid == 2:
            raise requests.HTTPError("404")
        return api_match(mid)

    monkeypatch.setattr(score, "fetch_match", fake)
    monkeypatch.setattr(score.time, "sleep", lambda s: None)
    rows = score.fetch_finished([1, 2])
    assert set(rows.match_id) == {1}
    again = score.fetch_finished([1, 2, 3], cached=rows)
    assert set(again.match_id) == {1, 3} and len(again) == 24
