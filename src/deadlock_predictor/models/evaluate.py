"""Metrics for probability predictions. Every model is compared on these.

- Log loss: average −log(probability given to what happened). Punishes confident mistakes
  hard; 0.693 (= ln 2) is what always predicting 50% scores.
- Brier score: mean squared error of the probability. Gentler on confident mistakes;
  0.25 for always-50%.
- Calibration: when the model says 70%, does that side win ~70% of the time? Measured by
  the expected calibration error (ECE) and shown as a reliability table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Per probability bin: mean prediction, observed win rate, and row count."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table = (
        pd.DataFrame({"bin": idx, "p": p, "y": y})
        .groupby("bin")
        .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    )
    table.index = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in table.index]
    return table


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Row-weighted average gap between predicted and observed rates across bins."""
    table = reliability(y, p, bins)
    return float((table.n * (table.predicted - table.observed).abs()).sum() / table.n.sum())


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return {
        "log_loss": float(log_loss(y, p, labels=[False, True])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "accuracy": float(np.mean((p > 0.5) == np.asarray(y, dtype=bool))),
    }


def scores_by_time(
    t: np.ndarray, y: np.ndarray, p: np.ndarray, min_rows: int = 500
) -> pd.DataFrame:
    """`scores` at each snapshot time with at least `min_rows` rows."""
    df = pd.DataFrame({"t": t, "y": y, "p": p})
    rows = {
        t: {**scores(g.y.to_numpy(), g.p.to_numpy()), "n": len(g)}
        for t, g in df.groupby("t")
        if len(g) >= min_rows
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("t")
