"""Fit LightGBM on train with early stopping on validation, then pick the final model.

Run: python -m deadlock_predictor.models.train_gbm [--section NAME] [--select-only]

Each section of `configs/model.yaml` (`gbm`: no heroes, `gbm_heroes`: with heroes) tries a
small grid of tree settings and saves its best. Then `select_final` compares the four cells
(logistic regression or trees, with or without heroes) on validation and records the choice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import yaml

from deadlock_predictor.features.snapshots import LABEL
from deadlock_predictor.models.data import add_heroes, hero_ids, load_split, snapshot_columns
from deadlock_predictor.models.evaluate import scores
from deadlock_predictor.models.gbm import SymmetricLGBM, input_columns

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "model.yaml"
SECTIONS = ("gbm", "gbm_heroes")


def run(settings: dict) -> dict:
    """Fit one grid (a section of the model config) and save its best model."""
    ids = hero_ids() if settings["heroes"] else []
    cols = input_columns(ids)
    train = load_split("train", snapshot_columns(cols))
    validation = load_split("validation", snapshot_columns(cols))
    if ids:
        train, validation = add_heroes(train, ids), add_heroes(validation, ids)
    y_val = validation[LABEL].to_numpy()
    print(f"train {len(train):,} rows, validation {len(validation):,} rows, {len(cols)} columns")

    grid = settings["grid"]
    results, best = [], None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values, strict=True))
        start = time.time()
        model = SymmetricLGBM(
            learning_rate=settings["learning_rate"],
            n_estimators=settings["n_estimators"],
            early_stopping_rounds=settings["early_stopping_rounds"],
            **params,
        ).fit(train[cols], train[LABEL], eval_set=(validation[cols], y_val))
        s = scores(y_val, model.predict_proba(validation[cols])[:, 1])
        results.append({"params": params, "trees": model.n_trees_, "validation": s})
        print(
            f"{params} trees {model.n_trees_} log loss {s['log_loss']:.4f} "
            f"({time.time() - start:.0f}s)"
        )
        if best is None or s["log_loss"] < best[1]["log_loss"]:
            best = (params, s, model)

    params, s, model = best
    model_path = ROOT / settings["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"params": params, "model": model, "hero_ids": ids, "columns": cols}, model_path)
    metrics = {
        "built_at": datetime.now(UTC).isoformat(),
        "best": {"params": params, "trees": model.n_trees_, "validation": s},
        "hero_ids": ids,
        "grid": results,
    }
    (ROOT / settings["metrics_path"]).write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"best: {params}; saved {model_path}")
    return metrics


def select_final(config: dict) -> dict:
    """Pick the final model among the cells, on validation log loss.

    Cells are listed simplest first; the simplest within `tolerance` of the best wins, as
    in the baseline ladder.
    """
    settings = config["final"]
    losses = {}
    for cell in settings["cells"]:
        metrics = json.loads((ROOT / config[cell]["metrics_path"]).read_text())
        losses[cell] = metrics["best"]["validation"]["log_loss"]
    cutoff = min(losses.values()) + settings["tolerance"]
    choice = next(cell for cell in settings["cells"] if losses[cell] <= cutoff)
    final = {
        "built_at": datetime.now(UTC).isoformat(),
        "choice": choice,
        "model_path": config[choice]["model_path"],
        "validation_log_loss": losses,
    }
    (ROOT / settings["path"]).write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--section", choices=SECTIONS, help="fit only this grid")
    parser.add_argument("--select-only", action="store_true", help="skip fitting")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if not args.select_only:
        for section in [args.section] if args.section else SECTIONS:
            print(f"== {section}")
            run(config[section])
    select_final(config)


if __name__ == "__main__":
    main()
