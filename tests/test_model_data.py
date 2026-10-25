import pandas as pd
import pytest

from deadlock_predictor.models.data import (
    add_heroes,
    hero_column,
    hero_ids,
    load_split,
    mirror,
    with_mirror,
)


@pytest.fixture
def table(tmp_path):
    df = pd.DataFrame(
        {
            "match_id": [1, 1, 2, 3, 4],
            "t": [180, 360, 180, 180, 180],
            "net_worth_diff": [100.0, -50.0, 0.0, 10.0, 20.0],
            "badge_mean": [50.0, 50.0, 60.0, 70.0, 80.0],
            "team0_win": [True, True, False, True, False],
            "split": ["train", "train", "validation", "test", "train"],
        }
    )
    path = tmp_path / "snapshots.parquet"
    df.to_parquet(path)
    return path


def test_load_split_returns_only_that_split(table):
    for split in ("train", "validation", "test"):
        df = load_split(split, path=table)
        assert set(df.split) == {split}


def test_no_match_in_two_splits(table):
    ids = {s: set(load_split(s, path=table).match_id) for s in ("train", "validation", "test")}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]


def test_load_split_keeps_ids_and_label(table):
    df = load_split("train", columns=["t"], path=table)
    assert list(df.columns) == ["match_id", "t", "team0_win"]


def test_load_split_rejects_unknown_split(table):
    with pytest.raises(ValueError):
        load_split("everything", path=table)


def test_mirror_negates_diffs_and_flips_label(table):
    df = load_split("train", path=table)
    m = mirror(df)
    assert (m.net_worth_diff == -df.net_worth_diff).all()
    assert (m.team0_win == ~df.team0_win).all()
    assert (m.t == df.t).all() and (m.badge_mean == df.badge_mean).all()


def test_with_mirror_balances_the_label(table):
    df = with_mirror(load_split("train", path=table))
    assert len(df) == 6
    assert df.team0_win.mean() == 0.5


@pytest.fixture
def heroes(tmp_path):
    # Match 1: Team0 has hero 3, Team1 has hero 5. Match 4: Team1 has hero 3, hero 9 (rare).
    df = pd.DataFrame(
        {"match_id": [1, 1, 4, 4, 2], "hero_id": [3, 5, 3, 9, 7], "sign": [1, -1, -1, 1, 1]}
    )
    path = tmp_path / "heroes.parquet"
    df.to_parquet(path)
    return path


def test_hero_ids_come_from_the_split(table, heroes):
    assert hero_ids("train", path=heroes, snapshots_path=table) == [3, 5, 9]


def test_add_heroes_signs_each_row(table, heroes):
    df = add_heroes(load_split("train", path=table), [3, 5], path=heroes)
    assert [hero_column(3), hero_column(5)] == ["hero_3_diff", "hero_5_diff"]
    assert df.hero_3_diff.tolist() == [1, 1, -1]  # rows: match 1, match 1, match 4
    assert df.hero_5_diff.tolist() == [-1, -1, 0]
    assert df.hero_3_diff.dtype == "int8"


def test_hero_outside_the_list_is_dropped(table, heroes):
    df = add_heroes(load_split("train", path=table), [3, 5], path=heroes)
    assert "hero_9_diff" not in df.columns


def test_mirror_negates_heroes(table, heroes):
    df = add_heroes(load_split("train", path=table), [3, 5], path=heroes)
    assert (mirror(df).hero_3_diff == -df.hero_3_diff).all()
