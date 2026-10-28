import pytest
from test_curve import models  # noqa: F401  (pytest fixture)
from test_timeline import bounty_match

from deadlock_predictor.viz import explorer

CONFIG = {"explorer": {"fight_gap_s": 20, "fight_radius": 3000, "fight_min_deaths": 2}}


@pytest.fixture
def match(models, monkeypatch):  # noqa: F811
    info = bounty_match()
    monkeypatch.setattr(explorer.md, "fetch_detail", lambda _: info)
    monkeypatch.setattr(explorer, "hero_name", lambda h: f"H{h}")
    monkeypatch.setattr(explorer, "item_name", lambda i: f"I{i}")
    return explorer.load_match(1, models, CONFIG)


def test_load_match(match):
    assert match.winner == "Team0" and not match.failed
    assert match.players.hero.tolist()[:2] == ["H1", "H2"]
    # Team0's two kills at 300 and 310 s are a fight (2+ deaths with this config).
    assert match.fights.label.tolist() == ["Fight 2-0 (Hidden King)"]
    assert {"p_before", "p_after"} <= set(match.fights.columns)
    assert "Fight 2-0 (Hidden King)" in match.points.trigger.tolist()


def test_match_figure_has_every_layer(match):
    fig = explorer.match_figure(match)
    names = {t.name for t in fig.data}
    assert {"P(Hidden King wins)", "kill", "death", "item", "buff", "dead"} <= names
    assert "net worth" in names  # SHAP group bars
    player_lines = [t for t in fig.data if t.meta in ("net_worth", "level")]
    assert len(player_lines) == 2 * 12
    assert fig.layout.updatemenus[0].buttons[1].label == "Level"


def test_map_figure(match):
    fig = explorer.map_figure(match, t=1)
    dots = [t for t in fig.data if t.mode == "markers+text"]
    assert len(dots) == 12


def test_tables(match):
    assert explorer.fight_table(match).result.tolist() == ["Fight 2-0 (Hidden King)"]
    players = explorer.player_table(match)
    assert len(players) == 12 and players.kills.sum() == 2 and players.deaths.sum() == 3
