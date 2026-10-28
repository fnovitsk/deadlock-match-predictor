"""The match page's charts (Plotly, dark template). Each takes a `viz.explorer.Match` and,
where it has a time axis, an optional range in minutes (so every chart zooms together)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from deadlock_predictor.data.assets import map_info
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

SLOT_COLORS = {"weapon": "#d9822b", "vitality": "#5fa83c", "spirit": "#9a5fd6"}
BIG_ITEM_TIER = 3  # 3,200 souls and up get an icon on the timeline
ROW_GAP = 1.0  # timeline rows are 1 apart; Team0's at the top


def _has(image) -> bool:
    """An image data URI is there (missing ones are None, or NaN inside a DataFrame)."""
    return isinstance(image, str) and bool(image)


def _minutes(t):
    return np.asarray(t, dtype=float) / 60


def _range(fig: go.Figure, xrange, **where) -> None:
    if xrange is not None:
        fig.update_xaxes(range=list(xrange), **where)


def _rows(m: Match) -> dict[int, float]:
    """The y of each player's row in the timeline: Team0 on top."""
    n = len(m.players)
    return {slot: float(n - 1 - i) for i, slot in enumerate(m.players.slot)}


def _portrait_labels(fig: go.Figure, m: Match, rows: dict[int, float], size: float = 0.8):
    """Hero portraits and names in the left margin, one per row."""
    for p in m.players.itertuples():
        y = rows[p.slot]
        if _has(p.portrait):
            fig.add_layout_image(
                source=p.portrait,
                xref="paper",
                yref="y",
                x=-0.012,
                y=y,
                sizex=0.05,
                sizey=size,
                xanchor="right",
                yanchor="middle",
                sizing="contain",
                layer="above",
            )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=-0.075,
            y=y,
            xanchor="right",
            text=f"<b>{p.hero}</b>",
            showarrow=False,
            font={"color": TEAM_COLORS[p.team], "size": 12},
        )


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


def player_timeline(m: Match, xrange=None, show_buffs: bool = True) -> go.Figure:
    """One row per player: kills (with a line to the victim: the kill web), deaths and time
    dead, item purchases (icons for big items), buffs."""
    rows = _rows(m)
    team_of = dict(zip(m.players.slot, m.players.team, strict=True))
    name_of = dict(zip(m.players.slot, m.players.hero, strict=True))
    fig = go.Figure()
    for team in TEAM_COLORS:
        ys = [rows[s] for s, t in team_of.items() if t == team]
        fig.add_hrect(
            y0=min(ys) - 0.5,
            y1=max(ys) + 0.5,
            fillcolor=TEAM_FILL[team],
            line_width=0,
            layer="below",
        )
    de = m.deaths
    xs, ys = [], []
    for d in de.itertuples():
        xs += [d.t / 60, (d.t + d.dead_s) / 60, None]
        ys += [rows[d.victim], rows[d.victim], None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "#3a4150", "width": 9},
            name="dead",
            hoverinfo="skip",
        )
    )
    kills = de[de.killer.notna()]
    for team in TEAM_COLORS:
        k = kills[kills.killer_team == team]
        xs, ys = [], []
        for d in k.itertuples():
            xs += [d.t / 60, d.t / 60, None]
            ys += [rows[int(d.killer)], rows[d.victim], None]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                hoverinfo="skip",
                line={"color": TEAM_COLORS[team], "width": 1},
                opacity=0.45,
                name=f"kill web ({TEAM_NAMES[team]})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=_minutes(k.t),
                y=[rows[int(s)] for s in k.killer],
                mode="markers",
                marker={
                    "size": 10,
                    "color": TEAM_COLORS[team],
                    "line": {"color": "#0d0f14", "width": 1},
                },
                name=f"kill ({TEAM_NAMES[team]})",
                text=[
                    f"<b>{clock(t)}</b>  {name_of[int(s)]} killed {name_of[v]}"
                    for t, s, v in zip(k.t, k.killer, k.victim, strict=True)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=_minutes(de.t),
            y=[rows[v] for v in de.victim],
            mode="markers",
            marker={"symbol": "x-thin", "size": 11, "line": {"width": 2, "color": TEXT}},
            name="death",
            text=[
                f"<b>{clock(t)}</b>  {name_of[v]} died"
                + (f" to {name_of[int(k)]}" if pd.notna(k) else "")
                + f" ({s:.0f} s dead)"
                for t, v, k, s in zip(de.t, de.victim, de.killer, de.dead_s, strict=True)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    shop = m.items[m.items.shop]
    small = shop[shop.tier < BIG_ITEM_TIER]
    fig.add_trace(
        go.Scatter(
            x=_minutes(small.t),
            y=[rows[s] - 0.3 for s in small.slot],
            mode="markers",
            marker={
                "symbol": "line-ns",
                "size": 9,
                "line": {"width": 2, "color": [SLOT_COLORS.get(s, MUTED) for s in small.slot_type]},
            },
            name="items (tier 1-2)",
            hoverinfo="skip",
        )
    )
    big = shop[shop.tier >= BIG_ITEM_TIER]
    for it in big.itertuples():
        if _has(it.icon):
            fig.add_layout_image(
                source=it.icon,
                xref="x",
                yref="y",
                x=it.t / 60,
                y=rows[it.slot],
                sizex=0.9,
                sizey=0.62,
                xanchor="center",
                yanchor="middle",
                sizing="contain",
                opacity=0.45 if pd.notna(it.sold_t) else 1,
                layer="above",
            )
    item_hover = [
        f"<b>{clock(t)}</b>  {name_of[s]} bought {n} ({c:,} souls)"
        + (f", sold at {clock(sold)}" if pd.notna(sold) else "")
        for t, s, n, c, sold in zip(
            shop.t, shop.slot, shop.name, shop.cost, shop.sold_t, strict=True
        )
    ]
    fig.add_trace(
        go.Scatter(
            x=_minutes(shop.t),
            y=[
                rows[s] - (0 if tier >= BIG_ITEM_TIER else 0.3)
                for s, tier in zip(shop.slot, shop.tier, strict=True)
            ],
            mode="markers",
            marker={"size": 16, "opacity": 0},
            text=item_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    if show_buffs and len(m.buffs):
        b = m.buffs
        fig.add_trace(
            go.Scatter(
                x=_minutes(b.t),
                y=[rows[s] + 0.3 for s in b.slot],
                mode="markers",
                marker={"symbol": "diamond", "size": 4, "color": MUTED, "opacity": 0.6},
                name="buff pickups",
                text=[
                    f"{clock(t)}  {name_of[s]}: {k.replace('_', ' ')}"
                    for t, s, k in zip(b.t, b.slot, b.type, strict=True)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    _portrait_labels(fig, m, rows)
    fig.update_yaxes(
        showticklabels=False, range=[-0.6, len(rows) - 0.4], showgrid=False, zeroline=False
    )
    fig.update_xaxes(title_text="minute", range=[0, m.duration_s / 60 + 0.5])
    fig.update_layout(
        margin={"l": 175, "r": 20, "t": 30, "b": 40},
        hovermode="closest",
        legend={"orientation": "h", "y": -0.08},
    )
    _range(fig, xrange)
    return styled(fig, 70 * len(rows) + 120)


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


def builds(m: Match) -> go.Figure:
    """Each player's final items (bought, never sold), grouped by slot type, in buying order."""
    rows = _rows(m)
    shop = m.items[m.items.shop & m.items.sold_t.isna()]
    order = {"weapon": 0, "vitality": 1, "spirit": 2}
    fig = go.Figure()
    hover_x, hover_y, hover_text, width = [], [], [], 1
    for p in m.players.itertuples():
        mine = shop[shop.slot == p.slot].assign(o=lambda d: d.slot_type.map(order).fillna(3))
        mine = mine.sort_values(["o", "t"])
        for col, it in enumerate(mine.itertuples()):
            x = col + 0.5
            width = max(width, col + 1)
            if _has(it.icon):
                fig.add_layout_image(
                    source=it.icon,
                    xref="x",
                    yref="y",
                    x=x,
                    y=rows[p.slot],
                    sizex=0.85,
                    sizey=0.85,
                    xanchor="center",
                    yanchor="middle",
                    sizing="contain",
                    layer="above",
                )
            fig.add_shape(
                type="rect",
                x0=x - 0.46,
                x1=x + 0.46,
                y0=rows[p.slot] - 0.46,
                y1=rows[p.slot] + 0.46,
                line={"color": SLOT_COLORS.get(it.slot_type, GRID), "width": 2},
                fillcolor="rgba(255,255,255,0.03)",
                layer="below",
            )
            hover_x.append(x)
            hover_y.append(rows[p.slot])
            hover_text.append(
                f"<b>{it.name}</b> (tier {it.tier}, {it.cost:,} souls)<br>"
                f"{p.hero}, bought at {clock(it.t)}"
            )
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker={"size": 30, "opacity": 0},
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    _portrait_labels(fig, m, rows)
    fig.update_xaxes(range=[0, width], visible=False)
    fig.update_yaxes(range=[-0.6, len(rows) - 0.4], visible=False, scaleanchor="x")
    fig.update_layout(margin={"l": 175, "r": 10, "t": 10, "b": 10})
    return styled(fig, 62 * len(rows) + 30)


def kill_matrix(m: Match) -> go.Figure:
    """Who killed whom: killers down, victims across, Team0 first."""
    names = m.players.hero.tolist()
    slots = m.players.slot.tolist()
    counts = np.zeros((len(slots), len(slots)), dtype=int)
    index = {s: i for i, s in enumerate(slots)}
    for d in m.deaths[m.deaths.killer.notna()].itertuples():
        counts[index[int(d.killer)], index[d.victim]] += 1
    fig = go.Figure(
        go.Heatmap(
            z=counts,
            x=names,
            y=names,
            colorscale=[[0, "rgba(0,0,0,0)"], [0.01, "#2a2f3a"], [1, "#e5534b"]],
            text=np.where(counts > 0, counts.astype(str), ""),
            texttemplate="%{text}",
            hovertemplate="%{y} killed %{x} %{z} times<extra></extra>",
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    n0 = int((m.players.team == "Team0").sum())
    for v in (n0 - 0.5,):
        fig.add_hline(y=v, line_color=MUTED, line_width=1)
        fig.add_vline(x=v, line_color=MUTED, line_width=1)
    fig.update_yaxes(
        autorange="reversed",
        title_text="killer",
        tickfont={"size": 12},
        ticktext=[
            f"<span style='color:{TEAM_COLORS[t]}'>{n}</span>"
            for n, t in zip(names, m.players.team, strict=True)
        ],
        tickvals=names,
    )
    fig.update_xaxes(
        title_text="victim",
        side="top",
        tickangle=-40,
        ticktext=[
            f"<span style='color:{TEAM_COLORS[t]}'>{n}</span>"
            for n, t in zip(names, m.players.team, strict=True)
        ],
        tickvals=names,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin={"l": 120, "r": 10, "t": 110, "b": 10})
    return styled(fig, 560)


def minimap(m: Match, t: float, window_s: float = 20, height: int = 300) -> go.Figure:
    """A small map of one moment: everyone's position at t and the deaths around it."""
    info = map_info()
    r = info.radius
    fig = go.Figure()
    if info.dark:
        fig.add_layout_image(
            source=info.dark,
            xref="x",
            yref="y",
            x=-r,
            y=r,
            sizex=2 * r,
            sizey=2 * r,
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            layer="below",
            opacity=0.9,
        )
    at = m.paths[(m.paths.t == round(t)) & (m.paths.health > 0)].merge(
        m.players[["slot", "hero", "color"]], on="slot"
    )
    for team in TEAM_COLORS:
        q = at[at.team == team]
        fig.add_trace(
            go.Scatter(
                x=q.x,
                y=q.y,
                mode="markers",
                marker={
                    "size": 10,
                    "color": q.color,
                    "line": {"color": TEAM_COLORS[team], "width": 2},
                },
                text=q.hero,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    near = m.deaths[(m.deaths.t - t).abs() <= window_s]
    fig.add_trace(
        go.Scatter(
            x=near.x,
            y=near.y,
            mode="markers",
            marker={
                "symbol": "x",
                "size": 12,
                "color": [TEAM_COLORS[v] for v in near.victim_team],
                "line": {"width": 1, "color": TEXT},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_xaxes(range=[-r, r], visible=False)
    fig.update_yaxes(range=[-r, r], visible=False, scaleanchor="x")
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return styled(fig, height)
