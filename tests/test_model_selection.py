import json

import pytest

from deadlock_predictor.models import evaluate_test, train_gbm

CONFIG = {
    "baseline": {"metrics_path": "baseline.json", "model_path": "baseline.joblib"},
    "baseline_heroes": {"metrics_path": "lr_heroes.json", "model_path": "lr_heroes.joblib"},
    "gbm": {"metrics_path": "gbm.json", "model_path": "gbm.joblib"},
    "gbm_heroes": {"metrics_path": "gbm_heroes.json", "model_path": "gbm_heroes.joblib"},
    "final": {
        "cells": ["baseline", "baseline_heroes", "gbm", "gbm_heroes"],
        "tolerance": 0.0005,
        "path": "final.json",
    },
    "test": {"path": "test_metrics.json"},
}


def write_losses(root, losses):
    for cell, loss in losses.items():
        metrics = {"best": {"validation": {"log_loss": loss}}}
        (root / CONFIG[cell]["metrics_path"]).write_text(json.dumps(metrics))


def cells(baseline, baseline_heroes, gbm, gbm_heroes):
    return dict(baseline=baseline, baseline_heroes=baseline_heroes, gbm=gbm, gbm_heroes=gbm_heroes)


@pytest.mark.parametrize(
    "losses, choice",
    [
        (cells(0.55, 0.54, 0.53, 0.52), "gbm_heroes"),
        # gbm_heroes beats gbm by less than the tolerance: keep the simpler gbm.
        (cells(0.55, 0.54, 0.5203, 0.52), "gbm"),
        (cells(0.55, 0.56, 0.57, 0.58), "baseline"),
    ],
)
def test_select_final_prefers_the_simplest_within_tolerance(tmp_path, monkeypatch, losses, choice):
    monkeypatch.setattr(train_gbm, "ROOT", tmp_path)
    write_losses(tmp_path, losses)
    final = train_gbm.select_final(CONFIG)
    assert final["choice"] == choice
    assert json.loads((tmp_path / "final.json").read_text())["choice"] == choice


def test_test_split_is_scored_once(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_test, "ROOT", tmp_path)
    (tmp_path / "test_metrics.json").write_text("{}")
    with pytest.raises(FileExistsError):
        evaluate_test.run(CONFIG)
