"""Interactive match explorer: everything about one match on one time axis (Plotly).

`load_match` gathers the API metadata, the training snapshot rows, the win probability at
every snapshot, building, mid-boss and fight (`viz.timeline`), and the fights
(`viz.fights`). `match_figure` draws four rows sharing the time axis:
1. win probability, with buildings, mid-boss and fights marked;
2. what moved it between consecutive points (SHAP groups);
3. each player's souls (or level) and the team souls lead;
4. each player's timeline: kills, deaths (and time dead), items, buffs.
`map_figure` shows where everyone was at a chosen second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from deadlock_predictor.data import match_detail as md
from deadlock_predictor.data.assets import hero_name, item_name
from deadlock_predictor.data.match import api_rows
from deadlock_predictor.viz.curve import (
    CONFIG_PATH,
    EVENT_LABELS,
    GROUPS,
    TEAM_NAMES,
    load_models,
    match_curve,
    match_features,
)
from deadlock_predictor.viz.fights import fights
from deadlock_predictor.viz.timeline import event_points, features_at, objective_events

COLORS = {"Team0": "#e8890c", "Team1": "#2f7fd8"}  # amber Hidden King, sapphire Archmother
LIGHT = {"Team0": "rgba(232,137,12,0.08)", "Team1": "rgba(47,127,216,0.08)"}
UNITS = {"net_worth": "souls", "level": "level"}
MAX_STEP = 1500  # world units in one second: more is a respawn or teleport, not movement
GROUP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


@dataclass
class Match:
    match_id: int
    info: dict
    duration_s: float
    winner: str
    failed: list[str]
    players: pd.DataFrame  # slot, team, hero_id, hero (name), row (label in the timeline)
    stats: pd.DataFrame
    deaths: pd.DataFrame
    items: pd.DataFrame
    buffs: pd.DataFrame
    paths: pd.DataFrame
    events: pd.DataFrame  # buildings and mid-boss
    fights: pd.DataFrame  # with p_before / p_after
    points: pd.DataFrame  # the curve: t, kind, trigger, p_final, p_baseline, groups


def load_match(match_id: int, models: dict | None = None, config: dict | None = None) -> Match:
    config = config or yaml.safe_load(CONFIG_PATH.read_text())
    models = models or load_models(config)
    info = md.fetch_detail(match_id)
    rows = api_rows(info)
    snaps, heroes, failed = match_features(rows)
    if snaps.empty:
        raise ValueError(f"match {match_id} has no usable snapshots")
    players = md.players(info)
    players["hero"] = [hero_name(h) for h in players.hero_id]
    players["row"] = players.hero
    deaths, stats = md.deaths(info), md.player_stats(info)
    duration = float(info["duration_s"])

    settings = config.get("explorer", {})
    found = fights(
        deaths,
        gap_s=settings.get("fight_gap_s", 20),
        radius=settings.get("fight_radius", 3000),
        min_deaths=settings.get("fight_min_deaths", 3),
    )
    first_t = float(snaps.t.min())
    found = found[(found.end >= first_t) & (found.end < duration)].reset_index(drop=True)
    extra = pd.DataFrame({"t": found.end, "label": found.label})
    points = event_points(snaps, heroes, rows, deaths, stats, models, duration, extra)
    if len(found):
        edges = features_at(
            snaps, rows, deaths, np.concatenate([found.start - 1, found.end]), stats
        )
        p = match_curve(edges, heroes, {"final": models["final"]}).p_final.to_numpy()
        # features_at sorts its times: map them back.
        by_t = dict(zip(edges.t, p, strict=True))
        found["p_before"] = [by_t[t - 1] for t in found.start]
        found["p_after"] = [by_t[t] for t in found.end]
    return Match(
        match_id=int(match_id),
        info=info,
        duration_s=duration,
        winner=rows.winning_team.iloc[0],
        failed=failed,
        players=players,
        stats=stats,
        deaths=deaths,
        items=md.items(info),
        buffs=md.buffs(info),
        paths=md.paths(info),
        events=objective_events(rows),
        fights=found,
        points=points,
    )


def _minutes(t):
    return np.asarray(t, dtype=float) / 60


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def match_figure(m: Match) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.30, 0.14, 0.22, 0.34],
        subplot_titles=(
            "Win probability",
            "What moved it (change in log-odds, + = toward Hidden King)",
            "Souls",
            "Players",
        ),
    )
    end = m.duration_s / 60
    _win_probability(fig, m, row=1)
    _moves(fig, m, row=2)
    _souls(fig, m, row=3)
    _timelines(fig, m, row=4)
    for r in range(1, 5):
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="grey",
            range=[0, end + 0.5],
            row=r,
            col=1,
        )
    fig.update_xaxes(title_text="minute", row=4, col=1)
    fig.update_layout(
        height=1150,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        hovermode="closest",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.05, "font": {"size": 11}},
        title=f"Match {m.match_id}: {TEAM_NAMES[m.winner]} won in {_clock(m.duration_s)}",
        updatemenus=[_souls_toggle(fig)],
    )
    return fig


def _win_probability(fig, m: Match, row: int) -> None:
    pts = m.points
    # Shapes on a subplot without traces yet are dropped unless told otherwise.
    shape = {"row": row, "col": 1, "exclude_empty_subplots": False}
    fig.add_hrect(y0=0.5, y1=1, fillcolor=LIGHT["Team0"], line_width=0, layer="below", **shape)
    fig.add_hrect(y0=0, y1=0.5, fillcolor=LIGHT["Team1"], line_width=0, layer="below", **shape)
    fig.add_hline(y=0.5, line_color="grey", line_width=1, **shape)
    for f in m.fights.itertuples():
        color = COLORS.get(f.winner, "grey") if isinstance(f.winner, str) else "grey"
        fig.add_vrect(
            x0=(f.start - 2) / 60,
            x1=(f.end + 2) / 60,
            fillcolor=color,
            opacity=0.25,
            line_width=0,
            layer="below",
            **shape,
        )
    fig.add_trace(
        go.Scatter(
            x=_minutes(pts.t),
            y=pts.p_baseline,
            mode="lines",
            line={"color": "grey", "dash": "dot", "width": 1},
            name="baseline",
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    hover = [
        f"{_clock(t)}  {trigger or 'just before'}<br>P(Hidden King wins) = {p:.0%}"
        for t, trigger, p in zip(pts.t, pts.trigger, pts.p_final, strict=True)
    ]
    shown = pts.kind != "before"
    fig.add_trace(
        go.Scatter(
            x=_minutes(pts.t),
            y=pts.p_final,
            mode="lines",
            line={"color": "black", "width": 2},
            name="P(Hidden King wins)",
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    for kind, symbol, size in (("snapshot", "circle", 8), ("event", "circle-open", 7)):
        sel = shown & (pts.kind == kind)
        fig.add_trace(
            go.Scatter(
                x=_minutes(pts.t[sel]),
                y=pts.p_final[sel],
                mode="markers",
                marker={"color": "black", "symbol": symbol, "size": size},
                name="stat snapshot" if kind == "snapshot" else "event",
                text=[h for h, s in zip(hover, sel, strict=True) if s],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    # Buildings and mid-boss along the top (Hidden King) and bottom (Archmother) edges.
    ev = m.events
    for team in ("Team0", "Team1"):
        e = ev[ev.team == team]
        fig.add_trace(
            go.Scatter(
                x=_minutes(e.t),
                y=[0.96 if team == "Team0" else 0.04] * len(e),
                mode="text",
                text=[EVENT_LABELS[k] for k in e.kind],
                textfont={"color": COLORS[team], "size": 11},
                hovertext=[
                    f"{_clock(t)}  {k} ({TEAM_NAMES[team]})"
                    for t, k in zip(e.t, e.kind, strict=True)
                ],
                hoverinfo="text",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    for f in m.fights.itertuples():
        delta = getattr(f, "p_after", np.nan) - getattr(f, "p_before", np.nan)
        winner = f.winner if isinstance(f.winner, str) else None
        kills = sorted((f.deaths_team0, f.deaths_team1), reverse=True)
        fig.add_annotation(
            x=(f.start + f.end) / 120,
            y=1.0,
            yanchor="bottom",
            text=f"⚔ {kills[0]}-{kills[1]}",
            hovertext=f"{f.label}, {_clock(f.start)}–{_clock(f.end)}: P(Hidden King) {delta:+.0%}",
            showarrow=False,
            font={"size": 11, "color": COLORS.get(winner, "dimgrey")},
            row=row,
            col=1,
        )
    fig.add_vline(x=m.duration_s / 60, line_color=COLORS[m.winner], line_width=3, **shape)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="P(Hidden King)", row=row, col=1)


def _moves(fig, m: Match, row: int) -> None:
    pts = m.points
    names = [g for g in GROUPS if g in pts.columns]
    if not names:
        return
    moves = pts[names].diff().fillna(pts[names])
    keep = pts.kind != "before"  # the change up to each shown point, from the previous one
    sums = moves.cumsum()
    shown = sums[keep].diff().fillna(sums[keep])
    for name, color in zip(names, GROUP_COLORS, strict=False):
        fig.add_trace(
            go.Bar(
                x=_minutes(pts.t[keep]),
                y=shown[name],
                name=name,
                marker_color=color,
                width=0.35,
                legendgroup="moves",
                hovertemplate=f"{name}: %{{y:+.2f}}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="Δ log-odds", row=row, col=1)


def _souls(fig, m: Match, row: int) -> None:
    st = m.stats.merge(m.players[["slot", "hero"]], on="slot")
    lead = st.groupby(["t", "team"]).net_worth.sum().unstack().reindex(columns=["Team0", "Team1"])
    diff = (lead.Team0 - lead.Team1).reset_index(name="lead")
    zero = pd.DataFrame({"t": [0.0], "lead": [0.0]})
    diff = pd.concat([zero, diff], ignore_index=True)
    for team, sign in (("Team0", 1), ("Team1", -1)):
        fig.add_trace(
            go.Scatter(
                x=_minutes(diff.t),
                y=np.clip(diff.lead * sign, 0, None) * sign,
                fill="tozeroy",
                mode="none",
                fillcolor=LIGHT[team].replace("0.08", "0.30"),
                name=f"souls lead ({TEAM_NAMES[team]})",
                hovertemplate="team souls lead %{y:+,.0f}<extra></extra>",
                meta="souls",
            ),
            row=row,
            col=1,
        )
    for p in m.players.itertuples():
        mine = st[st.slot == p.slot]
        for stat in ("net_worth", "level"):
            fig.add_trace(
                go.Scatter(
                    x=_minutes(mine.t),
                    y=mine[stat],
                    mode="lines+markers",
                    marker={"size": 4},
                    line={"color": COLORS[p.team], "width": 1.5},
                    name=p.hero,
                    legendgroup=f"player{p.slot}",
                    showlegend=False,
                    visible=stat == "net_worth",
                    hovertemplate=f"{p.hero}: %{{y:,.0f}} {UNITS[stat]}<extra></extra>",
                    meta=stat,
                ),
                row=row,
                col=1,
            )
    fig.update_yaxes(title_text="souls", row=row, col=1)


def _souls_toggle(fig) -> dict:
    """Buttons that switch row 3 between souls and levels."""
    metas = [getattr(t, "meta", None) for t in fig.data]

    def visible(show: str) -> list[bool]:
        out = []
        for meta in metas:
            if meta in ("net_worth", "level"):  # a player's line
                out.append(meta == show)
            elif meta == "souls":  # the team lead: shown with souls only
                out.append(show == "net_worth")
            else:
                out.append(True)
        return out

    return {
        "type": "buttons",
        "direction": "right",
        "x": 0.0,
        "y": 0.505,
        "xanchor": "left",
        "showactive": True,
        "buttons": [
            {"label": "Souls", "method": "update", "args": [{"visible": visible("net_worth")}]},
            {"label": "Level", "method": "update", "args": [{"visible": visible("level")}]},
        ],
    }


def _timelines(fig, m: Match, row: int) -> None:
    rows = m.players.row.tolist()
    row_of = dict(zip(m.players.slot, m.players.row, strict=True))
    team_of = dict(zip(m.players.slot, m.players.team, strict=True))
    de = m.deaths
    # Time dead: a grey bar per death.
    xs, ys = [], []
    for d in de.itertuples():
        xs += [d.t / 60, (d.t + d.dead_s) / 60, None]
        ys += [row_of[d.victim], row_of[d.victim], None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "lightgrey", "width": 6},
            name="dead",
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    kills = de[de.killer.notna()]
    for name, frame, slot_col, symbol, size, text in (
        (
            "kill",
            kills,
            "killer",
            "circle",
            10,
            [
                f"{_clock(t)}  {row_of[int(k)]} killed {row_of[v]}"
                for t, k, v in zip(kills.t, kills.killer, kills.victim, strict=True)
            ],
        ),
        (
            "death",
            de,
            "victim",
            "x",
            9,
            [
                f"{_clock(t)}  {row_of[v]} died" + (f" to {row_of[int(k)]}" if pd.notna(k) else "")
                for t, v, k in zip(de.t, de.victim, de.killer, strict=True)
            ],
        ),
        (
            "item",
            m.items,
            "slot",
            "triangle-up",
            5,
            [
                f"{_clock(t)}  {row_of[s]} bought {item_name(i)}"
                for t, s, i in zip(m.items.t, m.items.slot, m.items.item_id, strict=True)
            ],
        ),
        (
            "buff",
            m.buffs,
            "slot",
            "diamond",
            4,
            [
                f"{_clock(t)}  {row_of[s]}: {b.replace('_', ' ')}"
                for t, s, b in zip(m.buffs.t, m.buffs.slot, m.buffs.type, strict=True)
            ],
        ),
    ):
        slots = frame[slot_col].astype(int)
        colors = [COLORS[team_of[s]] for s in slots]
        fig.add_trace(
            go.Scatter(
                x=_minutes(frame.t),
                y=[row_of[s] for s in slots],
                mode="markers",
                marker={
                    "symbol": symbol,
                    "size": size,
                    "color": "black" if name == "death" else colors,
                    "opacity": {"item": 0.45, "buff": 0.35}.get(name, 1),
                },
                name=name,
                text=text,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=rows[::-1],  # Hidden King on top
        tickfont={"size": 11},
        row=row,
        col=1,
    )
    # Color each hero's label by team.
    fig.update_yaxes(
        ticktext=[
            f"<span style='color:{COLORS[t]}'>{r}</span>"
            for r, t in zip(rows[::-1], m.players.team[::-1], strict=True)
        ],
        tickvals=rows[::-1],
        row=row,
        col=1,
    )


def map_figure(m: Match, t: float, trail_s: float = 30, window_s: float = 60) -> go.Figure:
    """Everyone's position at second `t` (dot size = health), their last `trail_s` seconds
    of movement, and deaths within `window_s` of t."""
    paths = m.paths
    fig = go.Figure()
    if paths.empty:
        fig.add_annotation(text="No position data for this match", showarrow=False)
        return fig
    names = dict(zip(m.players.slot, m.players.hero, strict=True))
    # Where anyone went during the match, faintly: it draws the lanes and bases.
    every = paths.iloc[::4]
    fig.add_trace(
        go.Scatter(
            x=every.x,
            y=every.y,
            mode="markers",
            marker={"size": 2, "color": "rgba(120,110,90,0.12)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for p in m.players.itertuples():
        mine = paths[(paths.slot == p.slot) & (paths.t <= t) & (paths.t >= t - trail_s)]
        if mine.empty:
            continue
        # Break the trail where the player jumps (respawn, teleport).
        jump = np.hypot(mine.x.diff(), mine.y.diff()) > MAX_STEP
        trail = pd.concat(
            [mine, pd.DataFrame({"x": np.nan, "y": np.nan}, index=mine.index[jump] - 0.5)]
        ).sort_index()
        fig.add_trace(
            go.Scatter(
                x=trail.x,
                y=trail.y,
                mode="lines",
                line={"color": COLORS[p.team], "width": 1},
                opacity=0.5,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        now = mine.iloc[-1]
        alive = now.health > 0
        fig.add_trace(
            go.Scatter(
                x=[now.x],
                y=[now.y],
                mode="markers+text",
                marker={
                    "size": 8 + 14 * now.health / 100,
                    "color": COLORS[p.team],
                    "line": {"color": "white", "width": 1},
                    "opacity": 1 if alive else 0.3,
                },
                text=[names[p.slot]],
                textposition="top center",
                textfont={"size": 10},
                name=names[p.slot],
                showlegend=False,
                hovertemplate=f"{names[p.slot]}: {now.health:.0f}% health<extra></extra>",
            )
        )
    near = m.deaths[(m.deaths.t - t).abs() <= window_s]
    if len(near):
        fig.add_trace(
            go.Scatter(
                x=near.x,
                y=near.y,
                mode="markers",
                marker={"symbol": "x", "size": 11, "color": [COLORS[v] for v in near.victim_team]},
                text=[f"{_clock(d.t)}  {names[d.victim]} died" for d in near.itertuples()],
                hovertemplate="%{text}<extra></extra>",
                name="deaths",
                showlegend=False,
            )
        )
    pad = 500
    fig.update_xaxes(range=[paths.x.min() - pad, paths.x.max() + pad], visible=False)
    fig.update_yaxes(
        range=[paths.y.min() - pad, paths.y.max() + pad], visible=False, scaleanchor="x"
    )
    fig.update_layout(
        height=620,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title=(
            f"Positions at {_clock(t)} "
            f"(trails: last {trail_s:.0f} s; ✕ deaths within {window_s:.0f} s)"
        ),
        plot_bgcolor="#f4f1ea",
    )
    return fig


def fight_table(m: Match) -> pd.DataFrame:
    f = m.fights
    if f.empty:
        return pd.DataFrame()
    names = dict(zip(m.players.slot, m.players.hero, strict=True))
    return pd.DataFrame(
        {
            "time": [f"{_clock(s)}–{_clock(e)}" for s, e in zip(f.start, f.end, strict=True)],
            "result": f.label,
            "P before": f.p_before.round(2),
            "P after": f.p_after.round(2),
            "change": (f.p_after - f.p_before).round(2),
            "died": [", ".join(names[v] for v in vs) for vs in f.victims],
        }
    )


def player_table(m: Match) -> pd.DataFrame:
    last = m.stats.sort_values("t").groupby("slot").tail(1).set_index("slot")
    p = m.players.set_index("slot")
    return pd.DataFrame(
        {
            "team": p.team.map(TEAM_NAMES),
            "hero": p.hero,
            "kills": m.deaths[m.deaths.killer.notna()]
            .killer.astype(int)
            .value_counts()
            .reindex(p.index, fill_value=0),
            "deaths": m.deaths.victim.value_counts().reindex(p.index, fill_value=0),
            "souls": last.net_worth.reindex(p.index),
            "level": last.get("level", pd.Series(dtype=float)).reindex(p.index),
            "hero damage": last.get("player_damage", pd.Series(dtype=float)).reindex(p.index),
            "healing": last.get("player_healing", pd.Series(dtype=float)).reindex(p.index),
        }
    ).reset_index(drop=True)
