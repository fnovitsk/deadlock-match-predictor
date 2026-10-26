import pandas as pd

from deadlock_predictor.data.live import heroes_frame, live_state, objectives_lost
from deadlock_predictor.live.collect import append, load

ALL_STANDING = 0xFFFF


def knock_out(mask, *ids):
    for i in ids:
        mask &= ~(1 << i)
    return mask


def game(match_id=7, start=1000, nw=(30_000, 25_000), lost0=(), lost1=()):
    """An `/active` entry: Team0 (heroes 1-6) ahead; `lostX` are objective ids destroyed."""
    return {
        "match_id": match_id,
        "start_time": start,
        "team_stats": [
            {"team": 0, "net_worth": nw[0], "objectives_mask": knock_out(ALL_STANDING, *lost0)},
            {"team": 1, "net_worth": nw[1], "objectives_mask": knock_out(ALL_STANDING, *lost1)},
        ],
        "players": [{"team": s // 6, "hero_id": s + 1, "abandoned": None} for s in range(12)],
    }


def test_objectives_lost_counts_groups():
    # 1, 3 = Tier 1 lanes; 5 = Tier 2; 10 = a shield generator; 0 = Core (not a feature).
    lost = objectives_lost(knock_out(ALL_STANDING, 0, 1, 3, 5, 10))
    assert lost == {"tier1": 2, "tier2": 1, "barracks": 0, "shield_gen": 1, "titan": 0}
    assert set(objectives_lost(ALL_STANDING).values()) == {0}


def test_live_state_signs_and_time():
    state = live_state(game(lost0=[1], lost1=[1, 3, 5]), now=1900)
    assert state["t"] == 900
    assert state["net_worth_diff"] == 5000
    # Team1 lost two Tier 1s and a Tier 2, Team0 one Tier 1: good for Team0.
    assert state["obj_tier1_diff"] == 1 and state["obj_tier2_diff"] == 1
    assert state["heroes_team0"] == [1, 2, 3, 4, 5, 6]


def test_swapping_teams_negates_the_diffs():
    a = game(lost0=[1], lost1=[1, 3, 5])
    b = {**a, "team_stats": [{**s, "team": 1 - s["team"]} for s in a["team_stats"]]}
    b["players"] = [{**p, "team": 1 - p["team"]} for p in a["players"]]
    sa, sb = live_state(a, now=1900), live_state(b, now=1900)
    for c in ("net_worth_diff", "obj_tier1_diff", "obj_tier2_diff"):
        assert sa[c] == -sb[c]


def test_heroes_frame():
    states = pd.DataFrame([live_state(game(), now=1900), live_state(game(), now=1960)])
    h = heroes_frame(states)
    assert len(h) == 12 and h.sign.sum() == 0
    assert h[h.hero_id == 1].sign.item() == 1 and h[h.hero_id == 12].sign.item() == -1


def test_append_keeps_one_row_per_poll(tmp_path):
    path = tmp_path / "2026-10-25.parquet"
    first = pd.DataFrame([live_state(game(), now=1900), live_state(game(8), now=1900)])
    assert append(first, path) == 2
    again = pd.DataFrame([live_state(game(), now=1900), live_state(game(), now=1960)])
    assert append(again, path) == 3  # the repeated (match 7, 1900) row is kept once
    assert len(load(tmp_path)) == 3
