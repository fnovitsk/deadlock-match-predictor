"""Load the processed snapshot table for modeling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from deadlock_predictor.data.split import SPLITS
from deadlock_predictor.features.snapshots import LABEL

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS_PATH = ROOT / "data" / "processed" / "snapshots.parquet"


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
