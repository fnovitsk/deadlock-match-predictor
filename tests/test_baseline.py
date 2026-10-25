import numpy as np
import pandas as pd
import pytest

from deadlock_predictor.models.baseline import (
    DIFFS,
    RUNGS,
    hero_weights,
    input_columns,
    make_model,
    weight_curves,
)
from deadlock_predictor.models.data import hero_column, mirror

HEROES = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def data():
    """Synthetic snapshots where Team0's chance rises with its net worth lead."""
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({d: rng.normal(0, 3, n) for d in DIFFS})
    df["net_worth_diff"] = rng.normal(0, 5000, n)
    df["t"] = rng.choice([180, 540, 900, 1500, 2100, 3000], n)
    df["badge_mean"] = rng.uniform(11, 116, n)
    # Each team picks one of heroes 1-4 (never the same one); hero 1 is strong.
    picks = np.array([rng.permutation(HEROES)[:2] for _ in range(n)])
    for h in HEROES:
        df[hero_column(h)] = (picks[:, 0] == h).astype(int) - (picks[:, 1] == h)
    p = 1 / (1 + np.exp(-df.net_worth_diff / 4000 - 1.5 * df[hero_column(1)]))
    df["team0_win"] = rng.random(n) < p
    return df


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_model_is_symmetric(data, rung):
    model = make_model(rung, hero_ids=HEROES).fit(data[input_columns(rung, HEROES)], data.team0_win)
    p = model.predict_proba(data[input_columns(rung, HEROES)])[:, 1]
    p_mirror = model.predict_proba(mirror(data)[input_columns(rung, HEROES)])[:, 1]
    np.testing.assert_allclose(p + p_mirror, 1.0, atol=1e-9)


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_model_learns_the_lead(data, rung):
    model = make_model(rung, hero_ids=HEROES).fit(data[input_columns(rung, HEROES)], data.team0_win)
    ahead = data.assign(net_worth_diff=10_000.0)
    behind = data.assign(net_worth_diff=-10_000.0)
    p_ahead = model.predict_proba(ahead[input_columns(rung, HEROES)])[:, 1].mean()
    p_behind = model.predict_proba(behind[input_columns(rung, HEROES)])[:, 1].mean()
    assert p_ahead > 0.75 > 0.25 > p_behind


def test_even_game_is_a_coin_flip(data):
    model = make_model(3).fit(data[input_columns(3)], data.team0_win)
    even = data.assign(**{d: 0.0 for d in DIFFS})
    np.testing.assert_allclose(model.predict_proba(even[input_columns(3)])[:, 1], 0.5)


def test_weight_curves(data):
    model = make_model(3).fit(data[input_columns(3)], data.team0_win)
    curves = weight_curves(model, np.array([300, 1200, 2400]))
    assert list(curves.columns) == DIFFS
    assert (curves.net_worth_diff > 0).all()


def test_weight_curves_need_time_varying_rung(data):
    model = make_model(2).fit(data[input_columns(2)], data.team0_win)
    with pytest.raises(ValueError):
        weight_curves(model, np.array([300]))


def test_hero_rung_learns_the_strong_hero(data):
    model = make_model(5, hero_ids=HEROES).fit(data[input_columns(5, HEROES)], data.team0_win)
    weights = hero_weights(model)
    assert weights.idxmax() == hero_column(1)
    assert weights[hero_column(1)] > 1


def test_hero_rung_needs_hero_ids():
    with pytest.raises(ValueError):
        make_model(5)


def test_input_columns_add_heroes_only_for_hero_rungs():
    assert input_columns(3, HEROES) == input_columns(3)
    assert input_columns(6, HEROES)[-4:] == [hero_column(h) for h in HEROES]


def test_weight_curves_include_time_varying_heroes(data):
    model = make_model(6, hero_ids=HEROES).fit(data[input_columns(6, HEROES)], data.team0_win)
    curves = weight_curves(model, np.array([300, 1200, 2400]))
    assert list(curves.columns) == DIFFS + [hero_column(h) for h in HEROES]
    assert (curves[hero_column(1)] > 0.5).all()  # the strong hero helps at every time
