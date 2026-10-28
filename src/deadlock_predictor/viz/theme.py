"""Colours and the Plotly template shared by every dashboard chart (dark, like the page)."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

TEAM_COLORS = {"Team0": "#e8890c", "Team1": "#2f7fd8"}  # Hidden King amber, Archmother sapphire
TEAM_FILL = {"Team0": "rgba(232,137,12,0.10)", "Team1": "rgba(47,127,216,0.10)"}
BACKGROUND = "#0d0f14"
PANEL = "#171b23"
GRID = "#262c38"
TEXT = "#e3e6ec"
MUTED = "#8a93a3"
GROUP_COLORS = {
    "net worth": "#4c9be8",
    "fights": "#f0a63a",
    "levels & farm": "#5cb85c",
    "objectives": "#e5534b",
    "mid-boss": "#a371f7",
    "heroes": "#b08968",
    "time & rank": "#d16ba5",
}
TEMPLATE = "deadlock_dark"

pio.templates[TEMPLATE] = go.layout.Template(
    layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": TEXT, "family": "Source Sans Pro, Segoe UI, sans-serif", "size": 13},
        "title": {"font": {"size": 16}, "x": 0.01, "xanchor": "left"},
        "xaxis": {
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "linecolor": GRID,
            "tickcolor": GRID,
            "title": {"font": {"color": MUTED}},
        },
        "yaxis": {
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "linecolor": GRID,
            "tickcolor": GRID,
            "title": {"font": {"color": MUTED}},
        },
        "hoverlabel": {
            "bgcolor": PANEL,
            "bordercolor": GRID,
            "font": {"color": TEXT, "size": 13},
        },
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT}},
        "colorway": list(GROUP_COLORS.values()),
        "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
    }
)


def styled(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the dashboard template (and a height)."""
    fig.update_layout(template=TEMPLATE)
    if height:
        fig.update_layout(height=height)
    return fig
