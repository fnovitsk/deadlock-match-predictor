"""Logistic-regression baseline: a ladder of increasingly rich models.

P(Team0 wins) = sigmoid(w · x). Every input x is a Team0 − Team1 diff, or a diff multiplied by
something that describes the whole game (a function of time, the lobby rank). There is no
intercept. So the logit is an odd function of the diffs, and the model is exactly symmetric:
swapping the teams (negating every diff) gives 1 − P. No column can learn a side advantage.

The time basis is a cubic spline of `t`. Multiplying a diff by it lets that diff's weight
change smoothly with game time (a "varying-coefficient" model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from deadlock_predictor.features.snapshots import FEATURES

# deaths_diff is left out: it is almost exactly −kills_diff (correlation −0.99).
DIFFS = [f for f in FEATURES if f.endswith("_diff") and f != "deaths_diff"]
KNOTS = [180, 600, 1200, 1800, 2400, 3600]  # seconds; constant beyond the ends
BADGE_CENTER, BADGE_SCALE = 55.0, 27.0  # train mean and std of badge_mean

RUNGS = {
    1: {"diffs": ["net_worth_diff"], "by_time": True, "rank": False},
    2: {"diffs": DIFFS, "by_time": False, "rank": False},
    3: {"diffs": DIFFS, "by_time": True, "rank": False},
    4: {"diffs": DIFFS, "by_time": True, "rank": True},
}


class DiffFeatures(BaseEstimator, TransformerMixin):
    """Build the model inputs from snapshot rows.

    by_time: each diff × each spline basis function of t (else the plain diffs).
    rank: add net_worth_diff × centred lobby rank × each basis function.
    """

    def __init__(self, diffs=tuple(DIFFS), by_time=True, rank=False, knots=tuple(KNOTS)):
        self.diffs = diffs
        self.by_time = by_time
        self.rank = rank
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
        return names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        diffs = X[list(self.diffs)].to_numpy(dtype=float)
        parts = []
        if self.by_time:
            basis = self._basis(X)
            parts.append((diffs[:, :, None] * basis[:, None, :]).reshape(len(X), -1))
        else:
            parts.append(diffs)
        if self.rank:
            rank = (X["badge_mean"].to_numpy(dtype=float) - BADGE_CENTER) / BADGE_SCALE
            nw = X["net_worth_diff"].to_numpy(dtype=float)
            parts.append((nw * rank)[:, None] * self._basis(X))
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_)


def input_columns(rung: int) -> list[str]:
    """Snapshot columns a rung reads."""
    spec = RUNGS[rung]
    return ["t", *spec["diffs"], *(["badge_mean"] if spec["rank"] else [])]


def make_model(rung: int, C: float = 1.0) -> Pipeline:
    """Feature builder → scaling (divide by std; no centring, which would break the
    symmetry) → logistic regression without intercept."""
    spec = RUNGS[rung]
    return Pipeline(
        [
            ("features", DiffFeatures(tuple(spec["diffs"]), spec["by_time"], spec["rank"])),
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
