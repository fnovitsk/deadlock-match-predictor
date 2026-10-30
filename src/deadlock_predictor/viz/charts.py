"""The match page's charts (Plotly, dark template). Each takes a `viz.explorer.Match` and,
where it has a time axis, an optional range in minutes (so every chart zooms together)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from deadlock_predictor.viz.curve import GROUPS, TEAM_NAMES
from deadlock_predictor.viz.explorer import Match, clock, state_at
from deadlock_predictor.viz.icons import objective_icon
from deadlock_predictor.viz.theme import (
    BACKGROUND,
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


ICON_STEP = 0.1  # vertical spacing of stacked objective icons (in P units)


def _levels(times: np.ndarray, gap: float) -> list[int]:
    """A row for each icon so icons closer than `gap` (minutes) don't overlap."""
    last: list[float] = []
    levels = []
    for t in times:
        level = next((i for i, x in enumerate(last) if t - x >= gap), len(last))
        if level == len(last):
            last.append(t)
        last[level] = t
        levels.append(level)
    return levels


def win_probability(m: Match, xrange=None, height: int = 560) -> go.Figure:
    """P(Hidden King wins) at every snapshot and event: the area between the curve and 50%
    in the colour of the team ahead, objective icons for the team that took them, fights
    shaded. Hovering anywhere along the curve shows the state of the game at that moment."""
    pts = m.points
    span = (xrange[1] - xrange[0]) if xrange else m.duration_s / 60 + 0.7
    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=1, fillcolor=TEAM_FILL["Team0"], line_width=0, layer="below")
    fig.add_hrect(y0=0, y1=0.5, fillcolor=TEAM_FILL["Team1"], line_width=0, layer="below")
    fig.add_hline(y=0.5, line_color=MUTED, line_width=1, line_dash="dot")
    for f in m.fights.itertuples():
        winner = f.winner if isinstance(f.winner, str) else None
        fig.add_vrect(
            x0=(f.start - 3) / 60,
            x1=(f.end + 3) / 60,
            line_width=0,
            layer="below",
            fillcolor=TEAM_COLORS.get(winner, MUTED),
            opacity=0.16,
        )
        kills = sorted((f.deaths_team0, f.deaths_team1), reverse=True)
        fig.add_annotation(
            x=(f.start + f.end) / 120,
            yref="paper",  # above the chart, clear of the icon strip
            y=1.0,
            yanchor="bottom",
            showarrow=False,
            text=f"⚔ {kills[0]}-{kills[1]}",
            font={"size": 11, "color": TEAM_COLORS.get(winner, MUTED)},
        )

    # The curve, sampled every 10 s (plus every point) so the shading crosses 50% cleanly.
    grid = np.unique(np.concatenate([np.arange(pts.t.min(), m.duration_s, 10), pts.t]))
    p = np.interp(grid, pts.t, pts.p_final)
    x = _minutes(grid)
    for team, side in (("Team0", np.maximum(p, 0.5)), ("Team1", np.minimum(p, 0.5))):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.full_like(p, 0.5),
                mode="lines",
                hoverinfo="skip",
                line={"width": 0},
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=side,
                mode="lines",
                fill="tonexty",
                hoverinfo="skip",
                line={"width": 0},
                showlegend=False,
                fillcolor=TEAM_FILL[team].replace("0.10", "0.28"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=_minutes(pts.t),
            y=pts.p_baseline,
            mode="lines",
            line={"color": MUTED, "dash": "dot", "width": 1.2},
            name="baseline model",
            hoverinfo="skip",
        )
    )
    fig.add_trace(  # a soft glow under the curve
        go.Scatter(
            x=_minutes(pts.t),
            y=pts.p_final,
            mode="lines",
            hoverinfo="skip",
            line={"color": "rgba(227,230,236,0.18)", "width": 9},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_minutes(pts.t),
            y=pts.p_final,
            mode="lines",
            line={"color": TEXT, "width": 2.5},
            name="P(Hidden King wins)",
            hoverinfo="skip",
        )
    )

    # The state of the game every 10 s, invisible, carrying the hover.
    state = state_at(m, grid)
    hover = [
        f"<b>{clock(t)}</b>  P(Hidden King) {pi:.0%}<br>"
        f"souls lead {'Hidden King' if d >= 0 else 'Archmother'} +{abs(d):,.0f}<br>"
        f"kills {k0}–{k1}   buildings taken {b0}–{b1}"
        for t, pi, d, k0, k1, b0, b1 in zip(
            grid,
            p,
            state.souls_lead,
            state.kills_Team0,
            state.kills_Team1,
            state.buildings_Team0,
            state.buildings_Team1,
            strict=True,
        )
    ]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p,
            mode="markers",
            marker={"size": 14, "opacity": 0},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    snap = pts[pts.kind == "snapshot"]
    fig.add_trace(
        go.Scatter(
            x=_minutes(snap.t),
            y=snap.p_final,
            mode="markers",
            name="stat snapshot (exact)",
            marker={"color": TEXT, "size": 7, "line": {"color": BACKGROUND, "width": 1.5}},
            text=[
                f"<b>{clock(t)}</b>  stat snapshot<br>P(Hidden King) {v:.0%}"
                for t, v in zip(snap.t, snap.p_final, strict=True)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Objective icons in strips outside 0-100% (so they never cover the curve): the Hidden
    # King's takes above, the Archmother's below, stacked when close, each with a thin line
    # to the curve.
    ev = m.events.sort_values("t")
    size_x = span * 0.032
    reach = {"Team0": 1.0, "Team1": 0.0}  # how far each strip extends
    for team in TEAM_COLORS:
        e = ev[ev.team == team]
        if e.empty:
            continue
        top = team == "Team0"
        levels = _levels(_minutes(e.t), gap=size_x * 1.05)
        ys = [(1.08 + ICON_STEP * lv) if top else (-0.08 - ICON_STEP * lv) for lv in levels]
        reach[team] = max(ys) + 0.05 if top else min(ys) - 0.05
        at_curve = np.interp(e.t, pts.t, pts.p_final)
        lx, ly = [], []
        for t, y, c in zip(_minutes(e.t), ys, at_curve, strict=True):
            lx += [t, t, None]
            ly += [y + (-0.05 if top else 0.05), c, None]
        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
                line={"color": TEAM_COLORS[team], "width": 1},
                opacity=0.45,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=_minutes(e.t),
                y=at_curve,
                mode="markers",
                showlegend=False,
                hoverinfo="skip",
                marker={
                    "color": TEAM_COLORS[team],
                    "size": 8,
                    "line": {"color": BACKGROUND, "width": 1.5},
                },
            )
        )
        for t, y, kind in zip(_minutes(e.t), ys, e.kind, strict=True):
            fig.add_layout_image(
                source=objective_icon(kind, TEAM_COLORS[team]),
                xref="x",
                yref="y",
                x=t,
                y=y,
                sizex=size_x,
                sizey=0.095,
                xanchor="center",
                yanchor="middle",
                sizing="contain",
                layer="above",
            )
        fig.add_trace(
            go.Scatter(
                x=_minutes(e.t),
                y=ys,
                mode="markers",
                marker={"size": 22, "opacity": 0},
                name="event",
                showlegend=False,
                text=[
                    f"<b>{clock(t)}</b>  {k} taken by {TEAM_NAMES[team]}<br>P(Hidden King) {c:.0%}"
                    for t, k, c in zip(e.t, e.kind, at_curve, strict=True)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.add_vline(x=m.duration_s / 60, line_color=TEAM_COLORS[m.winner], line_width=3)
    fig.add_annotation(
        x=m.duration_s / 60,
        y=0.5,
        xanchor="right",
        showarrow=False,
        text=f"{TEAM_NAMES[m.winner]} win ",
        bgcolor="rgba(13,15,20,0.8)",
        font={"color": TEAM_COLORS[m.winner], "size": 13},
    )
    fig.update_yaxes(
        range=[min(reach["Team1"], -0.12), max(reach["Team0"], 1.12)],
        tickformat=".0%",
        title_text="P(Hidden King wins)",
        tickvals=[0, 0.25, 0.5, 0.75, 1],
        gridcolor="rgba(38,44,56,0.6)",
    )
    fig.update_xaxes(title_text="minute", range=[0, m.duration_s / 60 + 0.7], showgrid=False)
    fig.update_layout(
        hovermode="closest",
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 1,
            "xanchor": "right",
            "font": {"size": 11, "color": MUTED},
        },
        margin={"t": 50},
    )
    _range(fig, xrange)
    return styled(fig, height)


def moves(m: Match, xrange=None, height: int = 320) -> go.Figure:
    """What moved the probability, one bar per snapshot or event: the change in each
    contribution group since the previous one (log-odds; + = toward Hidden King), with the
    net change marked. Hover a bar for the full breakdown and what triggered it."""
    pts = m.points
    names = [g for g in GROUPS if g in pts.columns]
    fig = go.Figure()
    if names:
        shown = pts[pts.kind != "before"]
        steps = shown[names].diff().fillna(shown[names])
        x = _minutes(shown.t)
        width = max(0.2, (xrange[1] - xrange[0] if xrange else m.duration_s / 60) * 0.009)
        for name in names:
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=steps[name],
                    name=name,
                    marker_color=GROUP_COLORS[name],
                    width=width,
                    hovertemplate=f"{name}: %{{y:+.2f}}<extra></extra>",
                )
            )
        net = steps.sum(axis=1)
        labels = shown.trigger.replace("stat snapshot", "stat snapshot (souls and stats since)")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=net,
                mode="markers",
                name="net change",
                marker={
                    "symbol": "diamond",
                    "size": 8,
                    "color": TEXT,
                    "line": {"color": BACKGROUND, "width": 1},
                },
                text=[f"<b>{clock(t)}</b> {lab}" for t, lab in zip(shown.t, labels, strict=True)],
                hovertemplate="%{text}<br>net: %{y:+.2f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.update_layout(
        barmode="relative",
        bargap=0,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.14, "x": 0, "font": {"size": 11}},
        margin={"t": 40},
    )
    fig.update_yaxes(title_text="Δ log-odds<br>(+ toward Hidden King)", zeroline=False)
    fig.update_xaxes(title_text="minute", range=[0, m.duration_s / 60 + 0.7], showgrid=False)
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
