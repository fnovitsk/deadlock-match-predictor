import pandas as pd

from deadlock_predictor.viz.fights import fights


def deaths(*rows):
    """Deaths as (t, victim_team, x, y); victims 1, 2, ... and no killer credited."""
    return pd.DataFrame(
        [
            {"t": t, "victim": i + 1, "victim_team": team, "killer": None, "x": x, "y": y}
            for i, (t, team, x, y) in enumerate(rows)
        ]
    )


def test_close_deaths_make_one_fight():
    f = fights(deaths((100, "Team1", 0, 0), (115, "Team1", 500, 0), (130, "Team0", 0, 500)))
    assert len(f) == 1
    assert f.loc[0, ["start", "end", "deaths_team0", "deaths_team1"]].tolist() == [100, 130, 1, 2]
    assert f.loc[0, "winner"] == "Team0" and f.loc[0, "label"] == "Fight 2-1 (Hidden King)"


def test_a_long_gap_splits_fights():
    rows = [(100, "Team1", 0, 0), (110, "Team1", 0, 0), (120, "Team0", 0, 0)]
    rows += [(160, "Team0", 0, 0), (170, "Team0", 0, 0), (175, "Team0", 0, 0)]
    f = fights(deaths(*rows))
    assert f.start.tolist() == [100, 160]
    assert f.winner.tolist() == ["Team0", "Team1"]


def test_distant_deaths_are_separate_fights():
    # Two fights at the same time, in lanes 10,000 units apart.
    rows = [(100 + i, "Team1", 0, 0) for i in range(3)]
    rows += [(101 + i, "Team0", 10_000, 0) for i in range(3)]
    f = fights(deaths(*rows))
    assert len(f) == 2 and sorted(f.x.round()) == [0, 10_000]


def test_skirmishes_are_dropped():
    assert fights(deaths((100, "Team1", 0, 0), (105, "Team0", 0, 0))).empty


def test_even_fight():
    f = fights(
        deaths(
            (100, "Team1", 0, 0), (101, "Team0", 0, 0), (102, "Team1", 0, 0), (103, "Team0", 0, 0)
        )
    )
    assert f.loc[0, "winner"] is None and f.loc[0, "label"] == "Fight 2-2 (even)"
