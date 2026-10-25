import duckdb
import pandas as pd
from test_snapshots import match_rows

from deadlock_predictor.features.heroes import build_heroes
from deadlock_predictor.features.snapshots import filter_matches


def heroes(*matches):
    con = duckdb.connect()
    raw = pd.DataFrame([row for match in matches for row in match])  # noqa: F841
    con.sql("CREATE TABLE raw AS SELECT * FROM raw")
    filter_matches(con)
    return build_heroes(con).df()


def test_twelve_heroes_per_match():
    df = heroes(match_rows(1), match_rows(2))
    assert df.groupby("match_id").size().to_dict() == {1: 12, 2: 12}


def test_sign_is_plus_for_team0():
    df = heroes(match_rows(1)).set_index("hero_id")
    # match_rows puts slots 0-5 (heroes 1-6) on Team0 and slots 6-11 on Team1.
    assert (df.loc[1:6, "sign"] == 1).all()
    assert (df.loc[7:12, "sign"] == -1).all()


def test_only_kept_matches():
    df = heroes(match_rows(1), match_rows(2, abandon_slot=3))
    assert set(df.match_id) == {1}
