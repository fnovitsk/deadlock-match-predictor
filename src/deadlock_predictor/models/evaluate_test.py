"""Score the baseline and the final model on the test split, once.

Run: python -m deadlock_predictor.models.evaluate_test [--force]

Every decision so far (features, model family, settings) was made on validation. The test
split, the newest 15% of matches, was never read. Looking at it once gives an honest
estimate of how the chosen model does on games it could not have influenced. Looking again
after changing something would turn it into a second validation set, so this script
refuses to overwrite its results unless forced.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import yaml

from deadlock_predictor.features.snapshots import LABEL
from deadlock_predictor.models.data import load_split, predict_saved, snapshot_columns
from deadlock_predictor.models.evaluate import reliability, scores, scores_by_time

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "model.yaml"


def run(config: dict, force: bool = False) -> dict:
    out_path = ROOT / config["test"]["path"]
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} exists: the test split has been scored already. "
            "Pass --force only if you accept that the test score is no longer independent."
        )
    final = json.loads((ROOT / config["final"]["path"]).read_text())
    cells = dict.fromkeys(["baseline", final["choice"]])
    saved = {cell: joblib.load(ROOT / config[cell]["model_path"]) for cell in cells}
    columns = sorted({c for s in saved.values() for c in snapshot_columns(s["columns"])})
    test = load_split("test", columns)
    y = test[LABEL].to_numpy()
    print(f"test {len(test):,} rows, {test.match_id.nunique():,} matches")

    results = {}
    for cell, s in saved.items():
        p = predict_saved(s, test)
        metrics = json.loads((ROOT / config[cell]["metrics_path"]).read_text())
        results[cell] = {
            "validation": metrics["best"]["validation"],
            "test": scores(y, p),
            "test_by_time": scores_by_time(test.t.to_numpy(), y, p)
            .reset_index()
            .to_dict("records"),
            "test_reliability": reliability(y, p).reset_index(names="bin").to_dict("records"),
        }
        print(
            f"{cell}: test log loss {results[cell]['test']['log_loss']:.4f} "
            f"(validation {results[cell]['validation']['log_loss']:.4f})"
        )
    report = {
        "built_at": datetime.now(UTC).isoformat(),
        "final": final["choice"],
        "rows": len(test),
        "matches": int(test.match_id.nunique()),
        "cells": results,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="score the test split again")
    args = parser.parse_args()
    run(yaml.safe_load(CONFIG_PATH.read_text()), force=args.force)


if __name__ == "__main__":
    main()
