import numpy as np
import pandas as pd
import pytest
from test_gbm import synthetic
from test_live import game

from deadlock_predictor.app.helpers import (
    history,
    live_states,
    live_table,
    pick_game,
    plot_history,
)
from deadlock_predictor.data.live import PREGAME_S
from deadlock_predictor.models.gbm import SymmetricLGBM, input_columns

HEROES = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def live_model():
    data = synthetic(3000, 0)
    cols = input_columns(HEROES, live=True)
    model = SymmetricLGBM(num_leaves=7, n_estimators=50).fit(data[cols], data.team0_win)
    return {"model": model, "columns": cols, "hero_ids": HEROES}


def test_live_table_lists_each_game(live_model):
    games = [game(7, start=0, nw=(40_000, 20_000), lost1=[1]), game(8, start=600)]
    states = live_states(games, live_model, now=1500 + PREGAME_S)
    table = live_table(states)
    assert table.match.tolist() == ["7", "8"]  # longest game first
    assert table.minute.tolist() == [25, 15]
    assert table["souls lead (Hidden King)"].tolist() == [20_000, 5_000]
    assert table["buildings lead (Hidden King)"].tolist() == [1, 0]
    assert ((table["P(Hidden King wins)"] > 0) & (table["P(Hidden King wins)"] < 1)).all()


def test_pick_game():
    assert pick_game(["5", "7"], "") == "5"  # nothing asked: the longest game
    assert pick_game(["5", "7"], "7") == "7"
    assert pick_game(["5", "7"], "8") is None  # not live
    assert pick_game([], "") is None


def test_no_games():
    assert live_table(live_states([], None)).empty


def test_history_merges_logs_without_duplicates(live_model):
    log = live_states([game(7, start=0)], live_model, now=600)
    session = pd.concat(
        [live_states([game(7, start=0)], live_model, now=t) for t in (600, 660, 720)]
    )
    past = history(7, log, session, pd.DataFrame())
    assert past.t.tolist() == [600 - PREGAME_S, 660 - PREGAME_S, 720 - PREGAME_S]
    assert history(9, log).empty


def test_plot_history(live_model):
    states = pd.concat(
        [live_states([game(7, start=0)], live_model, now=t) for t in np.arange(600, 900, 60)]
    )
    assert len(plot_history(states, 7).axes) == 2
