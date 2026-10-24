import pandas as pd
import pytest

from deadlock_predictor.models.data import load_split, mirror, with_mirror


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
