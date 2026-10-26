import numpy as np
import pandas as pd
import pytest
from test_gbm import synthetic
from test_match import api_match

from deadlock_predictor.data.match import api_rows
from deadlock_predictor.models import baseline, gbm
from deadlock_predictor.viz.curve import (
    GROUPS,
    label_levels,
    match_curve,
    match_events,
    match_features,
    plot_curve,
)

HEROES = [1, 2, 3, 4]
SWAP = {"Team0": "Team1", "Team1": "Team0"}


@pytest.fixture(scope="module")
def models():
    """Small baseline and final models fitted on synthetic snapshots."""
    data = synthetic(3000, 0)
    lr_cols = baseline.input_columns(3)
    gbm_cols = gbm.input_columns(HEROES)
    lr = baseline.make_model(3).fit(data[lr_cols], data.team0_win)
    trees = gbm.SymmetricLGBM(num_leaves=7, n_estimators=50).fit(data[gbm_cols], data.team0_win)
    return {
        "baseline": {"model": lr, "columns": lr_cols, "hero_ids": []},
        "final": {"model": trees, "columns": gbm_cols, "hero_ids": HEROES},
    }


def swapped(rows):
    """The same match with the teams' names exchanged."""
    out = rows.copy()
    out["team"] = out.team.map(SWAP)
    out["winning_team"] = out.winning_team.map(SWAP)
    for c in ["objectives.team", "mid_boss.team_killed", "mid_boss.team_claimed"]:
        out[c] = out[c].map(lambda teams: [SWAP[t] for t in teams])
    return out


def test_curve_is_symmetric(models):
    rows = api_rows(api_match())
    curve = match_curve(*match_features(rows)[:2], models)
    flipped = match_curve(*match_features(swapped(rows))[:2], models)
    for name in ("p_final", "p_baseline"):
        np.testing.assert_allclose(curve[name] + flipped[name], 1.0, atol=1e-9)


def test_groups_add_up_to_the_logit(models):
    curve = match_curve(*match_features(api_rows(api_match()))[:2], models)
    logit = np.log(curve.p_final / (1 - curve.p_final))
    np.testing.assert_allclose(curve[list(GROUPS)].sum(axis=1), logit, atol=1e-9)


def test_failed_filters_are_reported():
    info = api_match()
    info["players"][3]["abandon_match_time_s"] = 300
    snaps, _, failed = match_features(api_rows(info))
    assert failed == ["no_abandon"] and len(snaps) == 4


def test_events_name_the_scoring_team():
    events = match_events(api_rows(api_match()))
    # Team1's Tier 1 falls at 400 s (Team0 scored), the mid-boss claim at 650 s goes to
    # Team0, and Team0's shield generator falls at 700 s (Team1 scored).
    assert events.to_dict("list") == {
        "t": [400.0, 650.0, 700.0],
        "kind": ["T1", "MB", "SG"],
        "team": ["Team0", "Team0", "Team1"],
    }


def test_close_labels_get_separate_rows():
    events = pd.DataFrame(
        {"t": [600, 620, 640, 900, 610], "kind": ["T1"] * 5, "team": ["Team0"] * 4 + ["Team1"]}
    ).sort_values("t", ignore_index=True)
    # Team0 at 600, 620, 640 s (within a minute): rows 0, 1, 2; Team1 at 610 s: row 0;
    # Team0 at 900 s: back to row 0.
    assert label_levels(events) == [0, 0, 1, 2, 0]


def test_plot_draws(models):
    rows = api_rows(api_match())
    curve = match_curve(*match_features(rows)[:2], models)
    fig = plot_curve(curve, match_events(rows), 1, "Team0", 1000.0)
    assert len(fig.axes) == 2
