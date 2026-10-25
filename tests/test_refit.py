import numpy as np
import pandas as pd

from deadlock_predictor.models import baseline, gbm, refit
from deadlock_predictor.models.baseline import DIFFS


def fake_split(split, columns=None):
    rng = np.random.default_rng({"train": 0, "validation": 1}[split])
    n = 1000
    df = pd.DataFrame({d: rng.normal(0, 3, n) for d in DIFFS})
    df["net_worth_diff"] = rng.normal(0, 5000, n)
    df["t"] = rng.choice([180, 900, 2100], n).astype(float)
    df["badge_mean"] = 50.0
    df["match_id"] = np.arange(n) // 5
    df["team0_win"] = rng.random(n) < 1 / (1 + np.exp(-df.net_worth_diff / 4000))
    return df


def test_refit_uses_train_and_validation_only(monkeypatch):
    seen = []
    monkeypatch.setattr(refit, "load_split", lambda s, c=None: seen.append(s) or fake_split(s))
    saved = {"rung": 3, "C": 1.0, "model": None, "columns": [], "hero_ids": []}
    entry = refit.refit(saved, {}, ids=[1, 2])
    assert seen == ["train", "validation"]
    assert entry["fit_on"] == ("train", "validation")
    assert entry["hero_ids"] == []  # a model without heroes stays without heroes
    assert entry["columns"] == baseline.input_columns(3)


def test_refit_keeps_the_tree_count(monkeypatch):
    monkeypatch.setattr(refit, "load_split", lambda s, c=None: fake_split(s))
    data = fake_split("train")
    cols = gbm.input_columns()
    validated = gbm.SymmetricLGBM(num_leaves=7, n_estimators=40).fit(data[cols], data.team0_win)
    saved = {"params": {"num_leaves": 7}, "model": validated, "columns": cols, "hero_ids": []}
    entry = refit.refit(saved, {"learning_rate": 0.05, "n_jobs": 2}, ids=[])
    assert entry["model"].n_trees_ == validated.n_trees_ == 40
