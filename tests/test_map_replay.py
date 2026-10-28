import numpy as np
import pandas as pd
import pytest
from test_curve import models  # noqa: F401  (pytest fixture)
from test_explorer import match  # noqa: F401  (pytest fixture)
from test_match_detail import detail_match

from deadlock_predictor.data.assets import MapInfo
from deadlock_predictor.viz import map_replay

BUILDINGS = pd.DataFrame(
    {
        "owner": ["Team1", "Team0"],
        "name": ["Tier1Lane3", "Titan"],
        "x": [0.0, 10.0],
        "y": [5.0, -5.0],
    }
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(map_replay, "map_info", lambda: MapInfo(10752, None, None))
    monkeypatch.setattr(map_replay, "buildings", lambda: BUILDINGS)


def frame(fig, t):
    return next(f for f in fig.frames if f.name == str(t))


def test_one_frame_per_step(match):  # noqa: F811
    fig = map_replay.replay_figure(match, step_s=5)
    assert len(fig.frames) == int(match.duration_s) // 5 + 1
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)


def test_players_alive_then_dead(match):  # noqa: F811
    fig = map_replay.replay_figure(match, step_s=2)
    # In the synthetic paths everyone has 100% health at 0 s and 0% from 2 s on.
    assert len(frame(fig, 0).data[1].x) == 12
    assert len(frame(fig, 2).data[1].x) == 0


def test_buildings_grey_out_when_destroyed(match):  # noqa: F811
    # Team0 destroys Team1's Lane 3 Guardian at 400 s; the Titan never falls.
    fig = map_replay.replay_figure(match, step_s=5)
    standing, fallen = frame(fig, 395).data[3], frame(fig, 395).data[4]
    assert len(standing.x) == 2 and len(fallen.x) == 0
    standing, fallen = frame(fig, 400).data[3], frame(fig, 400).data[4]
    assert list(standing.text) == ["PD"] and list(fallen.x) == [0.0]  # the Patron


def test_recent_deaths_show_for_ten_seconds(match):  # noqa: F811
    fig = map_replay.replay_figure(match, step_s=5)
    # Deaths at 300 and 310 s.
    assert len(frame(fig, 305).data[2].x) == 1
    assert len(frame(fig, 315).data[2].x) == 1  # 300 s has gone, 310 s is shown
    assert len(frame(fig, 330).data[2].x) == 0


def test_opens_at_the_start_time(match):  # noqa: F811
    fig = map_replay.replay_figure(match, step_s=5, start_s=300)
    assert fig.layout.sliders[0].active == 60


def test_estimate_buildings():
    # Team0's six players stand at (1000, 2000) for the 10 s before Team1's Lane 3 Guardian
    # falls at 400 s; two matches agree.
    info = detail_match()
    xs, ys = np.full(500, 500.0), np.full(500, 600.0)  # quantized: x_min + x/res·(span)
    info["match_paths"] = {
        "interval_s": 1.0,
        "x_resolution": 1000,
        "y_resolution": 1000,
        "paths": [
            {
                "player_slot": s,
                "x_min": 0.0,
                "x_max": 2000.0,
                "y_min": 0.0,
                "y_max": 2000.0 / 0.6,
                "x_pos": xs.tolist(),
                "y_pos": ys.tolist(),
                "health": [100] * 500,
            }
            for s in range(12)
        ],
    }
    est = map_replay.estimate_buildings([info, info])
    lane = est[(est.owner == "Team1") & (est.name == "Tier1Lane3")].iloc[0]
    assert (lane.x, lane.y, lane.agree, lane.of) == (1000, 2000, 2, 2)
