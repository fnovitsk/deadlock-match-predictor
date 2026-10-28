import numpy as np
from test_curve import models  # noqa: F401  (pytest fixture)
from test_explorer import match  # noqa: F401  (pytest fixture)

from deadlock_predictor.viz import charts
from deadlock_predictor.viz.theme import TEMPLATE


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


def test_souls_and_lead(match):  # noqa: F811
    souls = charts.souls(match, "net_worth", xrange=(0, 5))
    assert len(souls.data) == 12 and list(souls.layout.xaxis.range) == [0, 5]
    levels = charts.souls(match, "level")
    assert all(np.nanmax(t.y) <= 50 for t in levels.data)
    lead = charts.souls_lead(match)
    assert len(lead.data) == 2


def test_every_chart_uses_the_theme(match):  # noqa: F811
    for fig in (charts.win_probability(match), charts.moves(match), charts.souls(match)):
        assert fig.layout.template == charts.styled(fig).layout.template
    assert TEMPLATE == "deadlock_dark"
