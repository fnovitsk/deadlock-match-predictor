import pytest
from test_curve import models  # noqa: F401  (pytest fixture)
from test_timeline import bounty_match

from deadlock_predictor.data.assets import Hero, Item
from deadlock_predictor.viz import explorer

CONFIG = {"explorer": {"fight_gap_s": 20, "fight_radius": 3000, "fight_min_deaths": 2}}
ICON = "data:image/png;base64,AAAA"


def fake_item(item_id):
    """Item 555 (the only purchase in the synthetic match) is a tier 3 weapon item."""
    if item_id == 555:
        return Item(555, "Big Gun", 3, 3200, "weapon", ICON, True)
    return Item(item_id, f"I{item_id}", 0, 0, "", None, False)


@pytest.fixture
def match(models, monkeypatch):  # noqa: F811
    info = bounty_match()
    for p in info["players"]:
        p["player_rank_data"] = {"initial_display_rank": 51}
    monkeypatch.setattr(explorer.md, "fetch_detail", lambda _: info)
    monkeypatch.setattr(explorer, "hero", lambda h: Hero(h, f"H{h}", "#123456", ICON))
    monkeypatch.setattr(explorer, "item", fake_item)
    return explorer.load_match(1, models, CONFIG)


def test_load_match(match):
    assert match.winner == "Team0" and not match.failed and match.badge == 51
    assert match.players.hero.tolist()[:2] == ["H1", "H2"]
    assert {"color", "portrait", "kills", "deaths", "assists"} <= set(match.players.columns)
    assert match.items[["name", "tier", "slot_type", "shop"]].values.tolist() == [
        ["Big Gun", 3, "weapon", True]
    ]
    # Team0's two kills at 300 and 310 s are a fight (2+ deaths with this config).
    assert match.fights.label.tolist() == ["Fight 2-0 (Hidden King)"]
    assert {"p_before", "p_after"} <= set(match.fights.columns)
    assert "Fight 2-0 (Hidden King)" in match.points.trigger.tolist()


def test_state_at(match):
    state = explorer.state_at(match, [299, 305, 311, 400])
    assert state.kills_Team0.tolist() == [0, 1, 2, 2]
    assert state.kills_Team1.tolist() == [0, 0, 0, 0]  # the teamkill counts for nobody
    # Team0 takes a Lane Guardian at 400 s.
    assert state.buildings_Team0.tolist() == [0, 0, 0, 1]
    assert (state.souls_lead > 0).all()  # Team0 players have 100 souls, Team1 50


def test_tables(match):
    assert explorer.fight_table(match).result.tolist() == ["Fight 2-0 (Hidden King)"]
    players = explorer.player_table(match)
    assert len(players) == 12 and players.team.tolist()[:1] == ["Hidden King"]
    assert players.kills.sum() == 12  # the game's own K/D/A: 1 each in the synthetic match
