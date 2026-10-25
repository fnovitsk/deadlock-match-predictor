"""Load the processed snapshot table for modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deadlock_predictor.data.split import SPLITS
from deadlock_predictor.features.snapshots import LABEL

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS_PATH = ROOT / "data" / "processed" / "snapshots.parquet"
HEROES_PATH = ROOT / "data" / "processed" / "heroes.parquet"


def diff_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_diff")]


def load_split(
    split: str, columns: list[str] | None = None, path: Path = SNAPSHOTS_PATH
) -> pd.DataFrame:
    """Rows of one split ("train", "validation" or "test").

    Only that split is read from disk, so a mistake elsewhere can't quietly mix splits.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if columns is not None:
        columns = list(dict.fromkeys(["match_id", *columns, LABEL]))
    return pd.read_parquet(path, columns=columns, filters=[("split", "==", split)])


def hero_column(hero_id: int) -> str:
    return f"hero_{hero_id}_diff"


def hero_ids(
    split: str = "train", path: Path = HEROES_PATH, snapshots_path: Path = SNAPSHOTS_PATH
) -> list[int]:
    """Sorted ids of the heroes picked in a split's matches.

    A model keeps the train list: a hero it never saw gets no column (all zeros) instead
    of a column whose weight was never learned.
    """
    matches = load_split(split, columns=[], path=snapshots_path).match_id.unique()
    heroes = pd.read_parquet(path, columns=["match_id", "hero_id"])
    return sorted(int(h) for h in heroes.hero_id[heroes.match_id.isin(matches)].unique())


def add_heroes(df: pd.DataFrame, ids: list[int], path: Path = HEROES_PATH) -> pd.DataFrame:
    """Add one int8 column per hero in `ids`: +1 if Team0 picked it, −1 if Team1, else 0.

    The columns end in `_diff`, so `mirror` negates them like the other diffs.
    """
    heroes = pd.read_parquet(path)
    wide = (
        heroes[heroes.match_id.isin(df.match_id.unique()) & heroes.hero_id.isin(ids)]
        .pivot(index="match_id", columns="hero_id", values="sign")
        .reindex(columns=ids)
        .reindex(df.match_id.unique())
        .fillna(0)
        .astype(np.int8)
    )
    columns = pd.DataFrame(
        wide.reindex(df.match_id).to_numpy(),
        columns=[hero_column(h) for h in ids],
        index=df.index,
    )
    return pd.concat([df, columns], axis=1)


def snapshot_columns(columns: list[str]) -> list[str]:
    """The columns that come from the snapshot table (hero columns come from `add_heroes`)."""
    return [c for c in columns if not c.startswith("hero_")]


def predict_saved(saved: dict, df: pd.DataFrame) -> np.ndarray:
    """P(Team0 wins) from a saved model entry (`model`, `columns`, `hero_ids`)."""
    if saved["hero_ids"]:
        df = add_heroes(df, saved["hero_ids"])
    return saved["model"].predict_proba(df[saved["columns"]])[:, 1]


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """The same snapshots seen from the other side: diffs negated, label flipped.

    Columns that describe the whole game (`t`, `badge_mean`, ids) are unchanged.
    """
    out = df.copy()
    diffs = diff_columns(df)
    out[diffs] = -out[diffs]
    out[LABEL] = ~out[LABEL].astype(bool)
    return out


def with_mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Original rows followed by their mirrored copies."""
    return pd.concat([df, mirror(df)], ignore_index=True)
