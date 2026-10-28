import numpy as np
import pytest
from test_curve import models  # noqa: F401  (pytest fixture)
from test_explorer import match  # noqa: F401  (pytest fixture)

from deadlock_predictor.data.assets import MapInfo
from deadlock_predictor.viz import charts
from deadlock_predictor.viz.theme import TEMPLATE


@pytest.fixture(autouse=True)
def offline_map(monkeypatch):
    monkeypatch.setattr(charts, "map_info", lambda: MapInfo(10752, None, None))


def test_win_probability(match):  # noqa: F811
    fig = charts.win_probability(match, xrange=(4, 12))
    names = {t.name for t in fig.data}
    assert {"P(Hidden King wins)", "stat snapshot (exact)", "event", "baseline model"} <= names
    assert list(fig.layout.xaxis.range) == [4, 12]
    assert fig.layout.template.layout.paper_bgcolor == "rgba(0,0,0,0)"
    # The hover carrier follows the curve every 10 s, with the state of the game.
    carrier = next(t for t in fig.data if t.marker.opacity == 0)
    assert "kills" in carrier.text[0] and len(carrier.x) > len(match.points)


def test_moves_bars_per_group(match):  # noqa: F811
    fig = charts.moves(match)
    assert {t.name for t in fig.data} >= {"net worth", "objectives"}


def test_player_timeline(match):  # noqa: F811
    fig = charts.player_timeline(match)
    kills = match.deaths.killer.notna().sum()
    web = [t for t in fig.data if t.name and t.name.startswith("kill web")]
    # One segment (three points: killer, victim, gap) per credited kill.
    assert sum(len(t.x) for t in web) == 3 * kills
    # The tier 3 item gets an icon, plus 12 portraits.
    icons = [i for i in fig.layout.images if i.xref == "x"]
    assert len(icons) == 1 and len(fig.layout.images) == 13


def test_souls_and_lead(match):  # noqa: F811
    souls = charts.souls(match, "net_worth")
    assert len(souls.data) == 12
    levels = charts.souls(match, "level")
    assert all(np.nanmax(t.y) <= 50 for t in levels.data)
    lead = charts.souls_lead(match)
    assert len(lead.data) == 2


def test_builds_and_matrix(match):  # noqa: F811
    builds = charts.builds(match)
    # The one item was sold, so no build has it.
    assert not [i for i in builds.layout.images if i.xref == "x"]
    matrix = charts.kill_matrix(match)
    assert np.asarray(matrix.data[0].z).sum() == match.deaths.killer.notna().sum()


def test_minimap(match):  # noqa: F811
    fig = charts.minimap(match, t=1)
    assert fig.layout.template.layout is not None
    assert fig.layout.yaxis.scaleanchor == "x"


def test_every_chart_uses_the_theme(match):  # noqa: F811
    for fig in (
        charts.win_probability(match),
        charts.player_timeline(match),
        charts.kill_matrix(match),
    ):
        assert fig.layout.template == charts.styled(fig).layout.template
    assert TEMPLATE == "deadlock_dark"
