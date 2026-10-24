import math

import numpy as np
import pytest

from deadlock_predictor.models.evaluate import (
    expected_calibration_error,
    reliability,
    scores,
    scores_by_time,
)


def test_coin_flip_scores():
    y = np.array([True, False, True, False])
    s = scores(y, np.full(4, 0.5))
    assert s["log_loss"] == pytest.approx(math.log(2))
    assert s["brier"] == pytest.approx(0.25)


def test_hand_computed_scores():
    y = np.array([True, False])
    p = np.array([0.8, 0.4])
    s = scores(y, p)
    assert s["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)
    assert s["brier"] == pytest.approx((0.2**2 + 0.4**2) / 2)
    assert s["accuracy"] == 1.0


def test_perfect_calibration_has_zero_ece():
    # In the 0.2 bin, 1 of 5 wins; in the 0.8 bin, 4 of 5 win.
    p = np.array([0.2] * 5 + [0.8] * 5)
    y = np.array([1, 0, 0, 0, 0] + [1, 1, 1, 1, 0], dtype=bool)
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_overconfidence_shows_in_ece_and_reliability():
    p = np.full(10, 0.9)
    y = np.array([1] * 6 + [0] * 4, dtype=bool)
    assert expected_calibration_error(y, p) == pytest.approx(0.3)
    table = reliability(y, p)
    assert list(table.index) == ["0.9–1.0"]
    assert table.observed.iloc[0] == pytest.approx(0.6)


def test_scores_by_time_skips_small_groups():
    t = np.array([180] * 3 + [360] * 1)
    y = np.array([True, False, True, False])
    out = scores_by_time(t, y, np.full(4, 0.5), min_rows=2)
    assert list(out.index) == [180]
    assert out.loc[180, "n"] == 3
