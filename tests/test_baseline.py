import numpy as np
import pandas as pd
import pytest

from deadlock_predictor.models.baseline import (
    DIFFS,
    RUNGS,
    input_columns,
    make_model,
    weight_curves,
)
from deadlock_predictor.models.data import mirror


@pytest.fixture(scope="module")
def data():
    """Synthetic snapshots where Team0's chance rises with its net worth lead."""
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({d: rng.normal(0, 3, n) for d in DIFFS})
    df["net_worth_diff"] = rng.normal(0, 5000, n)
    df["t"] = rng.choice([180, 540, 900, 1500, 2100, 3000], n)
    df["badge_mean"] = rng.uniform(11, 116, n)
    p = 1 / (1 + np.exp(-df.net_worth_diff / 4000))
    df["team0_win"] = rng.random(n) < p
    return df


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_model_is_symmetric(data, rung):
    model = make_model(rung).fit(data[input_columns(rung)], data.team0_win)
    p = model.predict_proba(data[input_columns(rung)])[:, 1]
    p_mirror = model.predict_proba(mirror(data)[input_columns(rung)])[:, 1]
    np.testing.assert_allclose(p + p_mirror, 1.0, atol=1e-9)


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_model_learns_the_lead(data, rung):
    model = make_model(rung).fit(data[input_columns(rung)], data.team0_win)
    ahead = data.assign(net_worth_diff=10_000.0)
    behind = data.assign(net_worth_diff=-10_000.0)
    p_ahead = model.predict_proba(ahead[input_columns(rung)])[:, 1].mean()
    p_behind = model.predict_proba(behind[input_columns(rung)])[:, 1].mean()
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
