"""Refit the baseline and the final model on train + validation, with their chosen settings.

Run: python -m deadlock_predictor.models.refit

Validation picked the settings (C, rung, tree settings, tree count). Once they are fixed,
validation's matches are just more data, and the newest data before test: refitting on
them helps a model keep up with drift (hero strength shifts over time, and hero 86 only
appears from validation on). No test data is read here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml

from deadlock_predictor.features.snapshots import LABEL
from deadlock_predictor.models import baseline, gbm
from deadlock_predictor.models.data import add_heroes, hero_ids, load_split, snapshot_columns

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "model.yaml"
FIT_SPLITS = ("train", "validation")


def refit_path(settings: dict) -> Path:
    path = ROOT / settings["model_path"]
    return path.with_name(f"{path.stem}_refit{path.suffix}")


def cells(config: dict) -> list[str]:
    """The cells scored on test: the baseline and the final choice."""
    final = json.loads((ROOT / config["final"]["path"]).read_text())
    return list(dict.fromkeys(["baseline", final["choice"]]))


def refit(saved: dict, settings: dict, ids: list[int]) -> dict:
    """A copy of a saved entry, refit on train + validation with the same settings."""
    ids = ids if saved["hero_ids"] else []
    if "rung" in saved:  # logistic regression
        model = baseline.make_model(saved["rung"], saved["C"], ids)
        columns = baseline.input_columns(saved["rung"], ids)
    else:  # LightGBM: the tree count early stopping chose, without stopping again
        model = gbm.SymmetricLGBM(
            learning_rate=settings["learning_rate"],
            n_estimators=saved["model"].n_trees_,
            n_jobs=settings["n_jobs"],
            **saved["params"],
        )
        columns = gbm.input_columns(ids)
    data = pd.concat(
        [load_split(s, snapshot_columns(columns)) for s in FIT_SPLITS], ignore_index=True
    )
    if ids:
        data = add_heroes(data, ids)
    model.fit(data[columns], data[LABEL])
    entry = {k: v for k, v in saved.items() if k not in ("model", "columns", "hero_ids")}
    return {**entry, "model": model, "columns": columns, "hero_ids": ids, "fit_on": FIT_SPLITS}


def run(config: dict) -> None:
    ids = sorted(set().union(*(hero_ids(s) for s in FIT_SPLITS)))
    for cell in cells(config):
        start = time.time()
        saved = joblib.load(ROOT / config[cell]["model_path"])
        entry = refit(saved, config[cell], ids)
        joblib.dump(entry, refit_path(config[cell]))
        print(f"{cell}: refit on {' + '.join(FIT_SPLITS)} ({time.time() - start:.0f}s)")


def main() -> None:
    run(yaml.safe_load(CONFIG_PATH.read_text()))


if __name__ == "__main__":
    main()
