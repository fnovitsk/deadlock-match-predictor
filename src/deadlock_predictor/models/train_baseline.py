"""Fit the baseline ladder on train, tune C on validation, save the best model.

Run: python -m deadlock_predictor.models.train_baseline
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import yaml

from deadlock_predictor.features.snapshots import LABEL
from deadlock_predictor.models.baseline import input_columns, make_model
from deadlock_predictor.models.data import load_split
from deadlock_predictor.models.evaluate import scores

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "model.yaml"


def run(config: dict) -> dict:
    settings = config["baseline"]
    columns = sorted({c for rung in settings["rungs"] for c in input_columns(rung)})
    train, validation = load_split("train", columns), load_split("validation", columns)
    y_val = validation[LABEL].to_numpy()
    print(f"train {len(train):,} rows, validation {len(validation):,} rows")

    results = {"0": {"C": None, "validation": scores(y_val, np.full(len(y_val), 0.5))}}
    models = {}  # rung -> model with that rung's best C
    for rung in settings["rungs"]:
        cols = input_columns(rung)
        by_C, fits = {}, {}
        for C in settings["C_grid"]:
            start = time.time()
            fits[C] = make_model(rung, C).fit(train[cols], train[LABEL])
            s = scores(y_val, fits[C].predict_proba(validation[cols])[:, 1])
            by_C[C] = s
            print(f"rung {rung} C={C:<6} log loss {s['log_loss']:.4f} ({time.time() - start:.0f}s)")
        best_C = min(by_C, key=lambda c: by_C[c]["log_loss"])
        results[str(rung)] = {"C": best_C, "validation": by_C[best_C], "by_C": by_C}
        models[rung] = fits[best_C]

    # Prefer the simplest rung within `tolerance` of the best: a gain smaller than that is
    # noise, not a reason to keep extra parameters.
    losses = {r: results[str(r)]["validation"]["log_loss"] for r in models}
    cutoff = min(losses.values()) + settings["tolerance"]
    rung = min(r for r, loss in losses.items() if loss <= cutoff)
    C, model = results[str(rung)]["C"], models[rung]
    model_path = ROOT / settings["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"rung": rung, "C": C, "model": model}, model_path)
    metrics = {
        "built_at": datetime.now(UTC).isoformat(),
        "best": {"rung": rung, "C": C},
        "rungs": results,
    }
    (ROOT / settings["metrics_path"]).write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"best: rung {rung}, C={C}; saved {model_path}")
    return metrics


def main() -> None:
    run(yaml.safe_load(CONFIG_PATH.read_text()))


if __name__ == "__main__":
    main()
