"""Gradient-boosted trees (LightGBM), made exactly symmetric.

A tree model can't be made symmetric through its structure the way the logistic baseline
is: a split like "net_worth_diff > 2000" has no built-in mirror image. So symmetry comes
from two steps:
1. Train on every row plus its mirror (teams swapped), so the trees see both sides of
   every situation equally often.
2. Predict with the average of the two views: logit(P) = ½·(g(x) − g(mirror(x))), where g
   is the trees' raw score. Swapping the teams negates this exactly, so P(mirror) = 1 − P.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

from deadlock_predictor.models.baseline import DIFFS
from deadlock_predictor.models.data import diff_columns, hero_column

BASE_COLUMNS = ["t", *DIFFS, "badge_mean"]


def input_columns(hero_ids=()) -> list[str]:
    """Columns the trees read: the baseline's diffs, game time, lobby rank, and heroes."""
    return [*BASE_COLUMNS, *(hero_column(h) for h in hero_ids)]


def mirror_features(X: pd.DataFrame) -> pd.DataFrame:
    """Negate the diff columns of a feature table (`data.mirror` also flips the label)."""
    out = X.copy()
    diffs = diff_columns(X)
    out[diffs] = -out[diffs]
    return out


def _doubled(X: pd.DataFrame, y) -> tuple[pd.DataFrame, np.ndarray]:
    y = np.asarray(y, dtype=bool)
    X2 = pd.concat([X, mirror_features(X)], ignore_index=True).astype(np.float32)
    return X2, np.concatenate([y, ~y])


class SymmetricLGBM(BaseEstimator, ClassifierMixin):
    """LightGBM trained on mirrored data, predicting the symmetrized logit."""

    def __init__(
        self,
        num_leaves=31,
        min_child_samples=200,
        learning_rate=0.05,
        n_estimators=3000,
        early_stopping_rounds=100,
        n_jobs=-1,
        random_state=0,
    ):
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y, eval_set: tuple | None = None):
        """eval_set: (X_val, y_val) for early stopping; it is mirrored too, so the
        stopping signal doesn't favour either side."""
        self.columns_ = list(X.columns)
        self.classes_ = np.array([False, True])
        self.model_ = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=-1,
        )
        fit_args = {}
        if eval_set is not None:
            fit_args = {
                "eval_set": [_doubled(*eval_set)],
                "eval_metric": "binary_logloss",
                "callbacks": [lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            }
        self.model_.fit(*_doubled(X, y), **fit_args)
        self.n_trees_ = self.model_.best_iteration_ or self.n_estimators
        return self

    def _raw(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model_.predict(
            X[self.columns_].astype(np.float32),
            raw_score=True,
            num_iteration=self.n_trees_,
            **kwargs,
        )

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """The symmetrized logit of P(Team0 wins)."""
        return 0.5 * (self._raw(X) - self._raw(mirror_features(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.decision_function(X) > 0

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Each feature's share of the logit (SHAP values, symmetrized like the logit).

        The rows sum to `decision_function(X)`: the trees' constant ("bias") term cancels
        between x and mirror(x).
        """
        c = 0.5 * (
            self._raw(X, pred_contrib=True) - self._raw(mirror_features(X), pred_contrib=True)
        )
        return pd.DataFrame(c[:, :-1], columns=self.columns_, index=X.index)
