import numpy as np
import pandas as pd
import pytest

from deadlock_predictor.models.baseline import DIFFS
from deadlock_predictor.models.data import hero_column
from deadlock_predictor.models.gbm import SymmetricLGBM, input_columns, mirror_features

HEROES = [1, 2, 3, 4]


def synthetic(n, seed):
    """Team0's chance rises with its net worth lead and with picking hero 1."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({d: rng.normal(0, 3, n) for d in DIFFS})
    df["net_worth_diff"] = rng.normal(0, 5000, n)
    df["t"] = rng.choice([180, 540, 900, 1500, 2100, 3000], n).astype(float)
    df["badge_mean"] = rng.uniform(11, 116, n)
    picks = np.array([rng.permutation(HEROES)[:2] for _ in range(n)])
    for h in HEROES:
        df[hero_column(h)] = (picks[:, 0] == h).astype(int) - (picks[:, 1] == h)
    p = 1 / (1 + np.exp(-df.net_worth_diff / 4000 - 1.5 * df[hero_column(1)]))
    df["team0_win"] = rng.random(n) < p
    return df


@pytest.fixture(scope="module")
def model():
    train, val = synthetic(6000, 0), synthetic(2000, 1)
    cols = input_columns(HEROES)
    return SymmetricLGBM(num_leaves=7, learning_rate=0.1, n_estimators=300).fit(
        train[cols], train.team0_win, eval_set=(val[cols], val.team0_win)
    )


@pytest.fixture(scope="module")
def test_rows():
    return synthetic(1000, 2)[input_columns(HEROES)]


def test_model_is_symmetric(model, test_rows):
    p = model.predict_proba(test_rows)[:, 1]
    p_mirror = model.predict_proba(mirror_features(test_rows))[:, 1]
    np.testing.assert_allclose(p + p_mirror, 1.0, atol=1e-9)


def test_even_game_is_a_coin_flip(model, test_rows):
    even = test_rows.copy()
    even[[c for c in even.columns if c.endswith("_diff")]] = 0
    np.testing.assert_allclose(model.predict_proba(even)[:, 1], 0.5)


def test_learns_the_lead(model, test_rows):
    ahead = model.predict_proba(test_rows.assign(net_worth_diff=10_000.0))[:, 1].mean()
    behind = model.predict_proba(test_rows.assign(net_worth_diff=-10_000.0))[:, 1].mean()
    assert ahead > 0.75 > 0.25 > behind


def test_learns_the_strong_hero(model, test_rows):
    others = {hero_column(h): 0 for h in HEROES[1:]}
    with_hero = model.predict_proba(test_rows.assign(**others, hero_1_diff=1))[:, 1].mean()
    against = model.predict_proba(test_rows.assign(**others, hero_1_diff=-1))[:, 1].mean()
    assert with_hero > against + 0.2


def test_early_stopping_picks_fewer_trees(model):
    assert 0 < model.n_trees_ < 300


def test_contributions_sum_to_the_logit(model, test_rows):
    c = model.contributions(test_rows)
    assert list(c.columns) == input_columns(HEROES)
    np.testing.assert_allclose(c.sum(axis=1), model.decision_function(test_rows), atol=1e-9)


def test_mirror_features_keeps_whole_game_columns(test_rows):
    m = mirror_features(test_rows)
    assert (m.t == test_rows.t).all() and (m.badge_mean == test_rows.badge_mean).all()
    assert (m.net_worth_diff == -test_rows.net_worth_diff).all()


def test_live_columns_are_a_symmetric_subset():
    from deadlock_predictor.models.gbm import LIVE_COLUMNS

    live = input_columns(HEROES, live=True)
    assert set(LIVE_COLUMNS) < set(input_columns())
    assert "kills_diff" not in live and "badge_mean" not in live
    data, rows = synthetic(3000, 0), synthetic(500, 1)[live]
    model = SymmetricLGBM(num_leaves=7, n_estimators=50).fit(data[live], data.team0_win)
    p = model.predict_proba(rows)[:, 1] + model.predict_proba(mirror_features(rows))[:, 1]
    np.testing.assert_allclose(p, 1.0, atol=1e-9)
