"""The match page's charts (Plotly, dark template). Each takes a `viz.explorer.Match` and,
where it has a time axis, an optional range in minutes (so every chart zooms together)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from deadlock_predictor.viz.curve import EVENT_LABELS, GROUPS, TEAM_NAMES
from deadlock_predictor.viz.explorer import Match, clock, state_at
from deadlock_predictor.viz.theme import (
    GRID,
    GROUP_COLORS,
    MUTED,
    TEAM_COLORS,
    TEAM_FILL,
    TEXT,
    styled,
)


def _minutes(t):
    return np.asarray(t, dtype=float) / 60


def _range(fig: go.Figure, xrange, **where) -> None:
    if xrange is not None:
        fig.update_xaxes(range=list(xrange), **where)


def win_probability(m: Match, xrange=None, height: int = 520) -> go.Figure:
    """P(Hidden King wins) at every snapshot and event, with buildings, mid-boss and fights.

    Hovering anywhere along the curve shows the state of the game at that moment."""
    pts = m.points
    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=1, fillcolor=TEAM_FILL["Team0"], line_width=0, layer="below")
    fig.add_hrect(y0=0, y1=0.5, fillcolor=TEAM_FILL["Team1"], line_width=0, layer="below")
    fig.add_hline(y=0.5, line_color=GRID, line_width=1)
    for f in m.fights.itertuples():
        winner = f.winner if isinstance(f.winner, str) else None
        fig.add_vrect(
            x0=(f.start - 3) / 60,
            x1=(f.end + 3) / 60,
            fillcolor=TEAM_COLORS.get(winner, MUTED),
            opacity=0.22,
            line_width=0,
            layer="below",
        )
        kills = sorted((f.deaths_team0, f.deaths_team1), reverse=True)
        fig.add_annotation(
            x=(f.start + f.end) / 120,
            y=1.0,
            yanchor="bottom",
            text=f"⚔ {kills[0]}-{kills[1]}",
            showarrow=False,
            font={"size": 11, "color": TEAM_COLORS.get(winner, MUTED)},
        )
    fig.add_trace(
        go.Scatter(
            x=_minutes(pts.t),
            y=pts.p_baseline,
            mode="lines",
            line={"color": MUTED, "dash": "dot", "width": 1},
            name="baseline model",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_minutes(pts.t),
            y=pts.p_final,
            mode="lines",
            line={"color": TEXT, "width": 3},
            name="P(Hidden King wins)",
            hoverinfo="skip",
        )
    )
    # The state of the game every 10 s, invisible, carrying the hover.
    grid = np.unique(np.concatenate([np.arange(pts.t.min(), m.duration_s, 10), pts.t]))
    state = state_at(m, grid)
    p = np.interp(grid, pts.t, pts.p_final)
    lead = state.souls_lead
    hover = [
        f"<b>{clock(t)}</b>  P(Hidden King) {pi:.0%}<br>"
        f"souls lead {'Hidden King' if d >= 0 else 'Archmother'} +{abs(d):,.0f}<br>"
        f"kills {k0}–{k1}   buildings taken {b0}–{b1}"
        for t, pi, d, k0, k1, b0, b1 in zip(
            grid,
            p,
            lead,
            state.kills_Team0,
            state.kills_Team1,
            state.buildings_Team0,
            state.buildings_Team1,
            strict=True,
        )
    ]
    fig.add_trace(
        go.Scatter(
            x=_minutes(grid),
            y=p,
            mode="markers",
            marker={"size": 14, "opacity": 0},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    for kind, symbol, size, name in (
        ("snapshot", "circle", 9, "stat snapshot (exact)"),
        ("event", "circle-open", 9, "event"),
    ):
        sel = pts.kind == kind
        fig.add_trace(
            go.Scatter(
                x=_minutes(pts.t[sel]),
                y=pts.p_final[sel],
                mode="markers",
                marker={"color": TEXT, "symbol": symbol, "size": size, "line": {"width": 2}},
                name=name,
                text=[
                    f"<b>{clock(t)}</b>  {trig}<br>P(Hidden King) {pv:.0%}"
                    for t, trig, pv in zip(
                        pts.t[sel], pts.trigger[sel], pts.p_final[sel], strict=True
                    )
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    for team in TEAM_COLORS:
        e = m.events[m.events.team == team]
        y = 0.955 if team == "Team0" else 0.045
        fig.add_trace(
            go.Scatter(
                x=_minutes(e.t),
                y=[y] * len(e),
                mode="markers+text",
                marker={
                    "symbol": "triangle-down" if team == "Team0" else "triangle-up",
                    "size": 9,
                    "color": TEAM_COLORS[team],
                },
                text=[EVENT_LABELS[k] for k in e.kind],
                textposition="bottom center" if team == "Team0" else "top center",
                textfont={"color": TEAM_COLORS[team], "size": 10},
                hovertext=[
                    f"<b>{clock(t)}</b>  {k} ({TEAM_NAMES[team]})"
                    for t, k in zip(e.t, e.kind, strict=True)
                ],
                hoverinfo="text",
                name=f"{TEAM_NAMES[team]} objectives",
            )
        )
    fig.add_vline(x=m.duration_s / 60, line_color=TEAM_COLORS[m.winner], line_width=3)
    fig.add_annotation(
        x=m.duration_s / 60,
        y=0.5,
        xanchor="right",  # inside the plot, left of the end line
        text=f"{TEAM_NAMES[m.winner]} win ",
        showarrow=False,
        bgcolor="rgba(13,15,20,0.8)",
        font={"color": TEAM_COLORS[m.winner], "size": 13},
    )
    for text, y, team in (
        ("Hidden King ahead", 0.98, "Team0"),
        ("Archmother ahead", 0.02, "Team1"),
    ):
        fig.add_annotation(
            xref="paper",
            x=0.005,
            y=y,
            xanchor="left",
            yanchor="top" if y > 0.5 else "bottom",
            text=text,
            showarrow=False,
            font={"color": TEAM_COLORS[team], "size": 11},
        )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="P(Hidden King wins)")
    fig.update_xaxes(title_text="minute", range=[0, m.duration_s / 60 + 0.7])
    fig.update_layout(hovermode="closest", legend={"orientation": "h", "y": -0.18})
    _range(fig, xrange)
    return styled(fig, height)


def moves(m: Match, xrange=None, height: int = 200) -> go.Figure:
    """What moved the probability between consecutive curve points (log-odds, by group)."""
    pts = m.points
    names = [g for g in GROUPS if g in pts.columns]
    fig = go.Figure()
    if names:
        steps = pts[names].diff().fillna(pts[names])
        for name in names:
            fig.add_trace(
                go.Bar(
                    x=_minutes(pts.t),
                    y=steps[name],
                    name=name,
                    marker_color=GROUP_COLORS[name],
                    width=0.3,
                    hovertemplate=f"{name}: %{{y:+.2f}}<extra></extra>",
                )
            )
    fig.update_layout(
        barmode="relative", bargap=0, legend={"orientation": "h", "y": -0.35}, margin={"t": 10}
    )
    fig.update_yaxes(title_text="Δ log-odds")
    fig.update_xaxes(range=[0, m.duration_s / 60 + 0.7])
    _range(fig, xrange)
    return styled(fig, height)


def souls(m: Match, stat: str = "net_worth", xrange=None, height: int = 520) -> go.Figure:
    """Each player's souls (or level, or any per-snapshot stat) in their hero's colour."""
    fig = go.Figure()
    unit = {"net_worth": "souls", "level": "level"}.get(stat, stat.replace("_", " "))
    for p in m.players.itertuples():
        mine = m.stats[m.stats.slot == p.slot].sort_values("t")
        if stat not in mine:
            continue
        x = np.concatenate([[0.0], _minutes(mine.t)])
        y = np.concatenate([[0.0], mine[stat].to_numpy(float)])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=p.hero,
                line={
                    "color": p.color,
                    "width": 2.5,
                    "dash": "solid" if p.team == "Team0" else "dash",
                },
                marker={"size": 5},
                hovertemplate=(
                    f"<b>{p.hero}</b> ({TEAM_NAMES[p.team]})<br>"
                    f"%{{x:.1f}} min: %{{y:,.0f}} {unit}<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title_text="minute", range=[0, m.duration_s / 60 + 0.7])
    fig.update_yaxes(title_text=unit)
    fig.update_layout(
        hovermode="closest",
        legend={"orientation": "v", "x": 1.01, "y": 1},
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.06,
                "showarrow": False,
                "font": {"color": MUTED, "size": 11},
                "text": "solid: Hidden King · dashed: Archmother",
            }
        ],
    )
    _range(fig, xrange)
    return styled(fig, height)


def souls_lead(m: Match, xrange=None, height: int = 220) -> go.Figure:
    """The team souls lead, filled in the colour of whoever leads."""
    t = np.arange(0, m.duration_s + 1, 10)
    lead = state_at(m, t).souls_lead.to_numpy()
    fig = go.Figure()
    for team, sign in (("Team0", 1), ("Team1", -1)):
        fig.add_trace(
            go.Scatter(
                x=_minutes(t),
                y=np.clip(lead * sign, 0, None) * sign,
                fill="tozeroy",
                mode="lines",
                line={"color": TEAM_COLORS[team], "width": 1},
                fillcolor=TEAM_FILL[team].replace("0.10", "0.45"),
                name=f"{TEAM_NAMES[team]} ahead",
                hovertemplate="%{x:.1f} min: souls lead %{y:+,.0f}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="souls lead")
    fig.update_xaxes(range=[0, m.duration_s / 60 + 0.7])
    fig.update_layout(margin={"t": 10}, legend={"orientation": "h", "y": -0.3})
    _range(fig, xrange)
    return styled(fig, height)
