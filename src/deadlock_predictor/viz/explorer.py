"""Everything about one match, ready to draw: `load_match` and the `Match` it returns.

`load_match` gathers the API metadata, the training snapshot rows, the win probability at
every snapshot, building, mid-boss and fight (`viz.timeline`), the fights (`viz.fights`),
and hero and item details (`data.assets`). The charts are in `viz.charts` and
the page that shows them in `app.match_page`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from deadlock_predictor.data import match_detail as md
from deadlock_predictor.data.assets import hero, item
from deadlock_predictor.data.match import TEAMS, api_rows, lobby_badge
from deadlock_predictor.viz.curve import (
    CONFIG_PATH,
    TEAM_NAMES,
    load_models,
    match_curve,
    match_features,
)
from deadlock_predictor.viz.fights import fights
from deadlock_predictor.viz.timeline import event_points, features_at, objective_events

BUILDINGS = ["tier1", "tier2", "barracks", "shield_gen", "titan"]  # groups, not mid-boss


@dataclass
class Match:
    match_id: int
    info: dict
    duration_s: float
    winner: str
    failed: list[str]
    badge: int  # the lobby's average rank (tier·10 + subtier; 0 if unknown)
    # One row per player, Team0 first: slot, team, hero_id, hero (name), color, portrait
    # (data URI or None), and the game's own kills, deaths, assists.
    players: pd.DataFrame
    stats: pd.DataFrame
    deaths: pd.DataFrame
    items: pd.DataFrame  # with name, tier, cost, slot_type, icon, shop
    buffs: pd.DataFrame
    paths: pd.DataFrame
    events: pd.DataFrame  # buildings and mid-boss
    fights: pd.DataFrame  # with p_before / p_after
    points: pd.DataFrame  # the curve: t, kind, trigger, p_final, p_baseline, groups


def _players(info: dict) -> pd.DataFrame:
    players = md.players(info)
    details = [hero(h) for h in players.hero_id]
    players["hero"] = [h.name for h in details]
    players["color"] = [h.color for h in details]
    players["portrait"] = [h.portrait for h in details]
    totals = {p["player_slot"]: p for p in info["players"]}
    for key in ("kills", "deaths", "assists"):
        players[key] = [int(totals[s].get(key) or 0) for s in players.slot]
    return players


def _items(info: dict) -> pd.DataFrame:
    items = md.items(info)
    details = [item(i) for i in items.item_id]
    for field in ("name", "tier", "cost", "icon", "shop"):
        items[field] = [getattr(d, field) for d in details]
    items["slot_type"] = [d.slot for d in details]  # weapon / vitality / spirit
    return items


def load_match(match_id: int, models: dict | None = None, config: dict | None = None) -> Match:
    config = config or yaml.safe_load(CONFIG_PATH.read_text())
    models = models or load_models(config)
    info = md.fetch_detail(match_id)
    rows = api_rows(info)
    snaps, heroes, failed = match_features(rows)
    if snaps.empty:
        raise ValueError(f"match {match_id} has no usable snapshots")
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
    ranks = [
        (p.get("player_rank_data") or {}).get("initial_display_rank", 0) for p in info["players"]
    ]
    return Match(
        match_id=int(match_id),
        info=info,
        duration_s=duration,
        winner=rows.winning_team.iloc[0],
        failed=failed,
        badge=lobby_badge(ranks) if any(ranks) else 0,
        players=_players(info),
        stats=stats,
        deaths=deaths,
        items=_items(info),
        buffs=md.buffs(info),
        paths=md.paths(info),
        events=objective_events(rows),
        fights=found,
        points=points,
    )


def clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def state_at(m: Match, times) -> pd.DataFrame:
    """What the scoreboard said at each time: souls lead (Team0 − Team1, linear between
    snapshots), kills and buildings destroyed by each team so far."""
    times = np.asarray(times, dtype=float)
    team = (
        m.stats.groupby(["t", "team"])
        .net_worth.sum()
        .unstack()
        .reindex(columns=list(TEAMS.values()))
    )
    lead_t = np.concatenate([[0.0], team.index.to_numpy(float)])
    lead = np.concatenate([[0.0], (team.Team0 - team.Team1).to_numpy(float)])
    credited = m.deaths[m.deaths.killer_team.notna()]
    taken = m.events[m.events.group.isin(BUILDINGS)]
    out = pd.DataFrame({"t": times, "souls_lead": np.interp(times, lead_t, lead)})
    for team_name in TEAMS.values():
        kt = credited.t[credited.killer_team == team_name].to_numpy()
        bt = taken.t[taken.team == team_name].to_numpy()
        out[f"kills_{team_name}"] = [(kt <= t).sum() for t in times]
        out[f"buildings_{team_name}"] = [(bt <= t).sum() for t in times]
    return out


def fight_table(m: Match) -> pd.DataFrame:
    f = m.fights
    if f.empty:
        return pd.DataFrame()
    names = dict(zip(m.players.slot, m.players.hero, strict=True))
    return pd.DataFrame(
        {
            "time": [f"{clock(s)}–{clock(e)}" for s, e in zip(f.start, f.end, strict=True)],
            "result": f.label,
            "P before": f.p_before.round(2),
            "P after": f.p_after.round(2),
            "change": (f.p_after - f.p_before).round(2),
            "died": [", ".join(names[v] for v in vs) for vs in f.victims],
        }
    )


def player_table(m: Match) -> pd.DataFrame:
    """One row per player (Team0 first): the game's K/D/A and the final stats."""
    last = m.stats.sort_values("t").groupby("slot").tail(1).set_index("slot")
    p = m.players.set_index("slot")

    def final(stat):
        return last[stat].reindex(p.index) if stat in last else pd.Series(np.nan, index=p.index)

    return pd.DataFrame(
        {
            "team": p.team.map(TEAM_NAMES),
            "hero": p.hero,
            "kills": p.kills,
            "deaths": p.deaths,
            "assists": p.assists,
            "souls": final("net_worth"),
            "level": final("level"),
            "hero damage": final("player_damage"),
            "healing": final("player_healing"),
        }
    ).reset_index(drop=True)
