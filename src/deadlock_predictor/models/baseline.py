"""Logistic-regression baseline: a ladder of increasingly rich models.

P(Team0 wins) = sigmoid(w · x). Every input x is a Team0 − Team1 diff, or a diff multiplied by
something that describes the whole game (a function of time, the lobby rank). There is no
intercept. So the logit is an odd function of the diffs, and the model is exactly symmetric:
swapping the teams (negating every diff) gives 1 − P. No column can learn a side advantage.

The time basis is a cubic spline of `t`. Multiplying a diff by it lets that diff's weight
change smoothly with game time (a "varying-coefficient" model).

Rungs 5 and 6 add the hero columns (+1 Team0 picked it, −1 Team1, 0 neither), which are
diffs too: a constant weight per hero (5), or a weight that changes with time (6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from deadlock_predictor.features.snapshots import FEATURES
from deadlock_predictor.models.data import hero_column

# deaths_diff is left out: it is almost exactly −kills_diff (correlation −0.99).
DIFFS = [f for f in FEATURES if f.endswith("_diff") and f != "deaths_diff"]
KNOTS = [180, 600, 1200, 1800, 2400, 3600]  # seconds; constant beyond the ends
BADGE_CENTER, BADGE_SCALE = 55.0, 27.0  # train mean and std of badge_mean

# heroes: None (no hero columns), "constant" or "by_time".
RUNGS = {
    1: {"diffs": ["net_worth_diff"], "by_time": True, "rank": False, "heroes": None},
    2: {"diffs": DIFFS, "by_time": False, "rank": False, "heroes": None},
    3: {"diffs": DIFFS, "by_time": True, "rank": False, "heroes": None},
    4: {"diffs": DIFFS, "by_time": True, "rank": True, "heroes": None},
    5: {"diffs": DIFFS, "by_time": True, "rank": False, "heroes": "constant"},
    6: {"diffs": DIFFS, "by_time": True, "rank": False, "heroes": "by_time"},
}


class DiffFeatures(BaseEstimator, TransformerMixin):
    """Build the model inputs from snapshot rows.

    by_time: each diff × each spline basis function of t (else the plain diffs).
    rank: add net_worth_diff × centred lobby rank × each basis function.
    heroes: hero columns, added as they are, or × each basis function if heroes_by_time.
    """

    def __init__(
        self,
        diffs=tuple(DIFFS),
        by_time=True,
        rank=False,
        heroes=(),
        heroes_by_time=False,
        knots=tuple(KNOTS),
    ):
        self.diffs = diffs
        self.by_time = by_time
        self.rank = rank
        self.heroes = heroes
        self.heroes_by_time = heroes_by_time
        self.knots = knots

    def fit(self, X: pd.DataFrame, y=None):
        self.spline_ = SplineTransformer(
            knots=np.asarray(self.knots, dtype=float).reshape(-1, 1),
            degree=3,
            extrapolation="constant",
        ).fit(X[["t"]])
        self.feature_names_ = self._names()
        return self

    def _basis(self, X: pd.DataFrame) -> np.ndarray:
        return self.spline_.transform(X[["t"]].astype(float))

    def _names(self) -> list[str]:
        n_basis = self.spline_.n_features_out_
        names = (
            [f"{d}*B{k}" for d in self.diffs for k in range(n_basis)]
            if self.by_time
            else list(self.diffs)
        )
        if self.rank:
            names += [f"net_worth_diff*rank*B{k}" for k in range(n_basis)]
        if self.heroes_by_time:
            names += [f"{h}*B{k}" for h in self.heroes for k in range(n_basis)]
        else:
            names += list(self.heroes)
        return names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        diffs = X[list(self.diffs)].to_numpy(dtype=float)
        basis = self._basis(X)
        parts = [_times_basis(diffs, basis) if self.by_time else diffs]
        if self.rank:
            rank = (X["badge_mean"].to_numpy(dtype=float) - BADGE_CENTER) / BADGE_SCALE
            nw = X["net_worth_diff"].to_numpy(dtype=float)
            parts.append((nw * rank)[:, None] * basis)
        if self.heroes:
            heroes = X[list(self.heroes)].to_numpy(dtype=float)
            parts.append(_times_basis(heroes, basis) if self.heroes_by_time else heroes)
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_)


def _times_basis(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Every column × every basis function: (rows, n) and (rows, k) → (rows, n·k)."""
    return (values[:, :, None] * basis[:, None, :]).reshape(len(values), -1)


def _hero_columns(rung: int, hero_ids) -> list[str]:
    return [hero_column(h) for h in hero_ids] if RUNGS[rung]["heroes"] else []


def input_columns(rung: int, hero_ids=()) -> list[str]:
    """Columns a rung reads. Hero rungs need the train hero ids (`data.hero_ids`)."""
    spec = RUNGS[rung]
    rank = ["badge_mean"] if spec["rank"] else []
    return ["t", *spec["diffs"], *rank, *_hero_columns(rung, hero_ids)]


def make_model(rung: int, C: float = 1.0, hero_ids=()) -> Pipeline:
    """Feature builder → scaling (divide by std; no centring, which would break the
    symmetry) → logistic regression without intercept."""
    spec = RUNGS[rung]
    if spec["heroes"] and not hero_ids:
        raise ValueError(f"rung {rung} uses heroes: pass hero_ids")
    features = DiffFeatures(
        tuple(spec["diffs"]),
        spec["by_time"],
        spec["rank"],
        heroes=tuple(_hero_columns(rung, hero_ids)),
        heroes_by_time=spec["heroes"] == "by_time",
    )
    return Pipeline(
        [
            ("features", features),
            ("scale", StandardScaler(with_mean=False)),
            ("logreg", LogisticRegression(C=C, fit_intercept=False, max_iter=1000)),
        ]
    )


def weight_curves(model: Pipeline, times: np.ndarray) -> pd.DataFrame:
    """Each diff's weight as a function of game time, per unit of that diff (raw units).

    Only for time-varying rungs. weight(t) = Σ_k w_k · B_k(t) / scale_k.
    """
    features: DiffFeatures = model.named_steps["features"]
    if not features.by_time:
        raise ValueError("rung has constant weights; read them from the coefficients")
    coef = model.named_steps["logreg"].coef_[0] / model.named_steps["scale"].scale_
    basis = features._basis(pd.DataFrame({"t": times}))
    n_basis = basis.shape[1]
    curves = {
        d: basis @ coef[i * n_basis : (i + 1) * n_basis] for i, d in enumerate(features.diffs)
    }
    return pd.DataFrame(curves, index=pd.Index(times, name="t"))


def hero_weights(model: Pipeline) -> pd.Series:
    """Each hero's weight on the logit (constant-hero rung only): picking it adds this."""
    features: DiffFeatures = model.named_steps["features"]
    if not features.heroes or features.heroes_by_time:
        raise ValueError("needs a rung with constant hero weights")
    coef = model.named_steps["logreg"].coef_[0] / model.named_steps["scale"].scale_
    return pd.Series(coef[-len(features.heroes) :], index=list(features.heroes))
