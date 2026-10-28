import pandas as pd

from deadlock_predictor.viz.moments import DRIFT, describe, key_moments, steps


def points(rows):
    """Curve points from (t, kind, trigger, p, net worth, objectives) rows."""
    df = pd.DataFrame(rows, columns=["t", "kind", "trigger", "p_final", "net worth", "objectives"])
    return df


CURVE = points(
    [
        (180, "snapshot", "stat snapshot", 0.50, 0.0, 0.0),
        (599, "before", "", 0.52, 0.1, 0.0),
        (600, "event", "Walker (Hidden King)", 0.60, 0.2, 0.3),
        (620, "event", "Lane Guardian (Hidden King)", 0.65, 0.3, 0.4),
        (900, "snapshot", "stat snapshot", 0.63, 0.25, 0.4),
        (1499, "before", "", 0.62, 0.2, 0.4),
        (1500, "event", "Shrine (Archmother)", 0.40, 0.0, -0.5),
    ]
)


def test_steps_label_each_change():
    s = steps(CURVE)
    assert len(s) == len(CURVE) - 1
    assert s.label.tolist()[1] == "Walker (Hidden King)"
    assert s.dp.round(2).tolist()[1] == 0.08


def test_biggest_moment_first_and_close_steps_merge():
    m = key_moments(CURVE, n=3)
    # The Shrine (-0.23 with the drift just before it) is the biggest; the Walker and Lane
    # Guardian 20 s apart merge, with the second before the Walker (+0.15 in all).
    assert m.labels.iloc[0] == ["Shrine (Archmother)"] and m.team.iloc[0] == "Team1"
    merged = m[m.labels.apply(lambda x: "Walker (Hidden King)" in x)].iloc[0]
    assert merged.labels == ["Walker (Hidden King)", "Lane Guardian (Hidden King)"]
    assert round(merged.dp, 2) == 0.15 and merged.team == "Team0"
    assert (m.dp.abs().diff().dropna() <= 0).all()


def test_causes_follow_the_direction():
    m = key_moments(CURVE, n=1).iloc[0]
    assert m.causes == ["objectives", "net worth"]
    assert describe(m).endswith("(mostly objectives, net worth)")
    assert "+23% for Archmother" in m.text


def test_drift_and_fights():
    fights = pd.DataFrame({"start": [1000.0], "end": [1499.0], "label": ["Fight 3-0 (Archmother)"]})
    s = steps(CURVE, fights)
    assert s.set_index("t").label[1499] == "Fight 3-0 (Archmother)"
    m = key_moments(CURVE.iloc[:5], n=5)
    assert any(DRIFT in labels for labels in m.labels)


def test_empty_curve():
    assert key_moments(CURVE.iloc[:1]).empty
