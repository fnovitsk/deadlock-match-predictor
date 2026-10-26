"""Refit the baseline and the final model on train + validation, with their chosen settings.

Run: python -m deadlock_predictor.models.refit [--all]

With `--all`, the refit also uses the test split, giving the "deploy" models that draw the
win-probability curves (Milestone 5): the newest data, and every hero seen so far. They
can't be scored, since no unused data is left; the test score of the train + validation
refit is their expected quality.

Validation picked the settings (C, rung, tree settings, tree count). Once they are fixed,
validation's matches are just more data, and the newest data before test: refitting on
them helps a model keep up with drift (hero strength shifts over time, and hero 86 only
appears from validation on). No test data is read here.
"""

from __future__ import annotations

import argparse
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
ALL_SPLITS = ("train", "validation", "test")


def refit_path(settings: dict, kind: str = "refit") -> Path:
    """`models/<cell>_refit.joblib` (train + validation) or `_deploy` (all splits)."""
    path = ROOT / settings["model_path"]
    return path.with_name(f"{path.stem}_{kind}{path.suffix}")


def cells(config: dict) -> list[str]:
    """The cells scored on test: the baseline and the final choice."""
    final = json.loads((ROOT / config["final"]["path"]).read_text())
    return list(dict.fromkeys(["baseline", final["choice"]]))


def refit(saved: dict, settings: dict, ids: list[int], splits=FIT_SPLITS) -> dict:
    """A copy of a saved entry, refit on `splits` with the same settings."""
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
        columns = gbm.input_columns(ids, live=settings.get("live", False))
    data = pd.concat([load_split(s, snapshot_columns(columns)) for s in splits], ignore_index=True)
    if ids:
        data = add_heroes(data, ids)
    model.fit(data[columns], data[LABEL])
    entry = {k: v for k, v in saved.items() if k not in ("model", "columns", "hero_ids")}
    return {**entry, "model": model, "columns": columns, "hero_ids": ids, "fit_on": splits}


def run(config: dict, deploy: bool = False, only: str | None = None) -> None:
    """Refit the test cells (and, for deploy, the extra `deploy.cells`), or just `only`."""
    splits, kind = (ALL_SPLITS, "deploy") if deploy else (FIT_SPLITS, "refit")
    ids = sorted(set().union(*(hero_ids(s) for s in splits)))
    todo = cells(config) + (config["deploy"]["cells"] if deploy else [])
    for cell in [only] if only else todo:
        start = time.time()
        saved = joblib.load(ROOT / config[cell]["model_path"])
        entry = refit(saved, config[cell], ids, splits)
        joblib.dump(entry, refit_path(config[cell], kind))
        print(f"{cell}: refit on {' + '.join(splits)} ({time.time() - start:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="also fit on test: deploy models")
    parser.add_argument("--cell", help="refit only this config section")
    args = parser.parse_args()
    run(yaml.safe_load(CONFIG_PATH.read_text()), deploy=args.all, only=args.cell)


if __name__ == "__main__":
    main()
